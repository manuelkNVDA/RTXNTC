/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <libntc/ntc.h>

#include "OptixDecompressor.h"
#include "OptixDecompressionKernel.h"
#include "OptixDecompressionKernelPtx.h"

//
// Error checking
//
#define CHK_MSG( call, message ) if( !call ) return message
#define CUDA_CHK_MSG( call, message ) if( call != CUDA_SUCCESS ) return message
#define OPTIX_CHK_MSG( call, message ) if( call != OPTIX_SUCCESS ) return message

#define CHK( call ) if( !call ) return false
#define NTC_CHK( call ) if( call != ntc::Status::Ok ) return false
#define CUDA_CHK( call ) if( call != CUDA_SUCCESS ) return false
#define OPTIX_CHK( call ) if( call != OPTIX_SUCCESS ) return false

inline void optixCheckLog( OptixResult res,
    const char*  log,
    size_t       sizeof_log,
    size_t       sizeof_log_returned,
    const char*  call,
    const char*  file,
    unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
            << log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
        std::cerr << ss.str().c_str();
        exit(1);
    }
}

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call, __FILE__, __LINE__ ); \
    } while( false )

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

//
// Constants and data structures
//
#define NTC_HIDDEN2_OR_OUTPUT ((NTC_MLP_LAYERS == 4) ? NTC_MLP_HIDDEN2_CHANNELS : NTC_MLP_OUTPUT_CHANNELS)
const unsigned int layerChannels[5] = {NTC_MLP_INPUT_CHANNELS, NTC_MLP_HIDDEN0_CHANNELS, 
    NTC_MLP_HIDDEN1_CHANNELS, NTC_HIDDEN2_OR_OUTPUT, NTC_MLP_OUTPUT_CHANNELS}; 

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;

//
// Decompression code
//
OptixDecompressor::OptixDecompressor( ntc::IContext* ntcContext, ntc::ITextureSetMetadata* texSetMetadata, ntc::IStream* texSetStream ) :
    m_ntcContext( ntcContext ),
    m_texSetMetadata( texSetMetadata ),
    m_texSetStream( texSetStream )
{
}

OptixDecompressor::~OptixDecompressor()
{
    cuMemFree( m_sbt.raygenRecord );
    cuMemFree( m_sbt.missRecordBase );
    cuMemFree( m_sbt.hitgroupRecordBase );
    cuMemFree( m_inferenceData.d_mlpWeights );
    cuTexObjectDestroy( m_inferenceData.latentTexture );
}

const char* OptixDecompressor::prepareDecompression()
{
    CHK_MSG( MakeOptixContext(), "Failed to make OptiX context" );
    CHK_MSG( MakeOptixInferenceData(), "Failed to make OptiX inference data" );
    CHK_MSG( MakeOptixNetworkData(), "Failed to make OptiX network data" );
    CHK_MSG( MakeLatentTexture(), "Failed to make OptiX latent texture" );
    CHK_MSG( MakeOptixPipeline(), "Failed to make OptiX pipeline" );
    return nullptr;
}

bool OptixDecompressor::MakeOptixContext()
{
    // Initialize CUDA
    CUcontext cuCtx = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
    CUctxCreateParams params = {};
    CUDA_CHK( cuCtxCreate( &cuCtx, &params, 0, 0 ) );
#else
    CUDA_CHK( cuCtxCreate( &cuCtx, 0, 0 ) );
#endif

    // Create OptiX context
    OPTIX_CHK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHK( optixDeviceContextCreate( cuCtx, &options, &m_optixContext ) );

    return true;
}

bool OptixDecompressor::MakeOptixInferenceData()
{
    // Get standard ntc inference data
    ntc::InferenceData ntcInferenceData;
    ntc::InferenceWeightType weightType = ntc::InferenceWeightType::GenericFP8;
    int firstLatentMip = 0;
    NTC_CHK( m_ntcContext->MakeInferenceData( m_texSetMetadata, weightType, firstLatentMip, &ntcInferenceData ) );
    m_inferenceData.constants = ntcInferenceData.constants;

    // Get latent texture info needed for optix inference.
    ntc::LatentTextureDesc latentTextureDesc = m_texSetMetadata->GetLatentTextureDesc();
    m_inferenceData.latentFeatures = latentTextureDesc.arraySize * 4;
    m_inferenceData.latentWidth = latentTextureDesc.width;
    m_inferenceData.latentHeight = latentTextureDesc.height;
    m_inferenceData.numLatentMips = latentTextureDesc.mipLevels;

    // Get subtexture info.
    m_inferenceData.numTextures = m_texSetMetadata->GetTextureCount();
    m_inferenceData.numChannels = m_texSetMetadata->GetDesc().channels;
    for( int i = 0; i < m_inferenceData.numTextures; i++ )
    {
        m_inferenceData.texFirstChannel[i] = m_texSetMetadata->GetTexture(i)->GetFirstChannel();
        m_inferenceData.texNumChannels[i] = m_texSetMetadata->GetTexture(i)->GetNumChannels();
    }

    return true;
}

bool OptixDecompressor::MakeOptixNetworkData()
{
    // Get the host network data
    void const* h_srcNetwork = nullptr;
    size_t h_srcNetworkSize = 0;
    ntc::InferenceWeightType weightType = ntc::InferenceWeightType::GenericFP8;
    NTC_CHK( m_texSetMetadata->GetInferenceWeights( weightType, &h_srcNetwork, &h_srcNetworkSize, nullptr ) );

    // Allocate host and device memory for network copies
    CUdeviceptr d_srcNetwork = 0;
    CUdeviceptr d_dstNetwork = 0;
    CUDA_CHK( cuMemAlloc( &d_srcNetwork, NTC_NETWORK_MAX_SIZE ) );
    CUDA_CHK( cuMemAlloc( &d_dstNetwork, NTC_NETWORK_MAX_SIZE ) );

    // Copy source network to device, and convert to inferencing optimal
    CUDA_CHK( cuMemcpyHtoD( d_srcNetwork, h_srcNetwork, h_srcNetworkSize ) );
    CHK( ConvertNetworkToInferencingOptimal( d_srcNetwork, h_srcNetworkSize, d_dstNetwork, NTC_NETWORK_MAX_SIZE ) );
    CUDA_CHK( cuMemFree( d_srcNetwork ) );

    m_inferenceData.d_mlpWeights = d_dstNetwork;
    return true;
}

bool OptixDecompressor::ConvertNetworkToInferencingOptimal( CUdeviceptr d_srcNetworkData, int srcNetworkTotalSize, CUdeviceptr d_dstMatrix, int d_dstSize )
{
    const int numLayers = NTC_MLP_LAYERS;
    
    std::vector<OptixCoopVecMatrixDescription> srcLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );
    std::vector<OptixCoopVecMatrixDescription> dstLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );

    OptixCoopVecMatrixLayout srcMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    OptixCoopVecMatrixLayout dstMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    
    NtcTextureSetConstants& tsc = m_inferenceData.constants;
    const int optStride = 0;

    // Compute layer sizes
    for( int i = 0; i < numLayers; ++i )
    {
        unsigned int K = layerChannels[i];
        unsigned int N = layerChannels[i+1];

        size_t srcMatrixDataSize = 0;
        size_t dstMatrixDataSize = 0;

        OptixCoopVecElemType layerType = (i < numLayers - 1) ? OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 : OPTIX_COOP_VEC_ELEM_TYPE_INT8;

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            m_optixContext,
            N,
            K,
            layerType,
            srcMatrixLayout,
            optStride,
            &srcMatrixDataSize
            ) );

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            m_optixContext,
            N,
            K,
            layerType,
            dstMatrixLayout,
            optStride,
            &dstMatrixDataSize
            ) );
        
        OptixCoopVecMatrixDescription& srcLayer = srcLayerDesc[i];
        OptixCoopVecMatrixDescription& dstLayer = dstLayerDesc[i];
        srcLayer.N = dstLayer.N = N;
        srcLayer.K = dstLayer.K = K;
        srcLayer.offsetInBytes  = i == 0 ? 0 : srcLayerDesc[i - 1].offsetInBytes + srcLayerDesc[i - 1].sizeInBytes;
        dstLayer.offsetInBytes  = i == 0 ? 0 : dstLayerDesc[i - 1].offsetInBytes + dstLayerDesc[i - 1].sizeInBytes;
        srcLayer.elementType    = layerType;
        dstLayer.elementType    = layerType;
        srcLayer.layout         = srcMatrixLayout;
        dstLayer.layout         = dstMatrixLayout;
        srcLayer.rowColumnStrideInBytes = optStride;
        dstLayer.rowColumnStrideInBytes = optStride;
        srcLayer.sizeInBytes    = static_cast<unsigned int>( srcMatrixDataSize );
        dstLayer.sizeInBytes    = static_cast<unsigned int>( dstMatrixDataSize );

        // Put network data offsets in the texture set constants
        tsc.networkWeightOffsets[i] = dstLayer.offsetInBytes;
    }

    OptixNetworkDescription inputNetworkDescription = { srcLayerDesc.data(), static_cast<unsigned int>( srcLayerDesc.size() ) };
    OptixNetworkDescription outputNetworkDescription = { dstLayerDesc.data(), static_cast<unsigned int>( dstLayerDesc.size() ) };

    size_t dst_mats_size = dstLayerDesc.back().offsetInBytes + dstLayerDesc.back().sizeInBytes;  // trick to sum all dstLayer sizes
    size_t src_mats_size = srcLayerDesc.back().offsetInBytes + srcLayerDesc.back().sizeInBytes;  // trick to sum all srcLayer sizes
    size_t src_other_stuff_size = srcNetworkTotalSize - src_mats_size;
    size_t dst_total_size       = dst_mats_size + src_other_stuff_size;

    if( d_dstSize < (int)dst_total_size )
        return false;

    const int numNetworks = 1;
    OPTIX_CHK( optixCoopVecMatrixConvert(
        m_optixContext,
        CUstream{0},
        numNetworks,
        &inputNetworkDescription,
        d_srcNetworkData,
        optStride,
        &outputNetworkDescription,
        d_dstMatrix,
        optStride) );

    // Update scale and bias offsets to account for matrix size difference
    size_t diffSize = dst_mats_size - src_mats_size;
    for( int i = 0; i < numLayers; i++ )
    {
        if( tsc.networkScaleOffsets[i] > 0 )
            tsc.networkScaleOffsets[i] += diffSize;
        if( tsc.networkBiasOffsets[i] > 0 )
            tsc.networkBiasOffsets[i] += diffSize;
    }

    // copy the other stuff after the mats arrays from src to dest
    CUDA_CHK( cuMemcpyDtoD( d_dstMatrix + dst_mats_size, d_srcNetworkData + src_mats_size, src_other_stuff_size ) );
    return true;
}

bool OptixDecompressor::MakeLatentTexture()
{
    // Allocate mipmapped CUDA array
    int numLatentTextures = m_inferenceData.latentFeatures / 4;
    int pixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;
    int numMips = m_inferenceData.numLatentMips;
    
    // Create mipmapped array descriptor
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
    arrayDesc.Width = m_inferenceData.latentWidth;
    arrayDesc.Height = m_inferenceData.latentHeight;
    arrayDesc.Depth = 0;  // 2D texture
    arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
    arrayDesc.NumChannels = pixelStride;
    arrayDesc.Flags = 0;
    
    CUmipmappedArray mipmappedArray;
    CUDA_CHK( cuMipmappedArrayCreate( &mipmappedArray, &arrayDesc, numMips ) );
    
    // Fill each mip level
    for ( int mipLevel = 0; mipLevel < numMips; mipLevel++ )
    {
        // Get the source (latentSrc) and destination (levelArray)
        int mipWidth = m_inferenceData.latentWidth >> mipLevel;
        int mipHeight = m_inferenceData.latentHeight >> mipLevel;
        std::vector<uint16_t> latentSrc( mipWidth * mipHeight * pixelStride, 0 );
        CHK( ReadLatentMipLevelUshort( latentSrc, mipLevel, mipWidth, mipHeight ) );
        CUarray levelArray;
        CUDA_CHK( cuMipmappedArrayGetLevel( &levelArray, mipmappedArray, mipLevel ) );
        
        // Copy data to this mip level
        CUDA_MEMCPY2D copyDesc = {};
        copyDesc.srcMemoryType = CU_MEMORYTYPE_HOST;
        copyDesc.srcHost = latentSrc.data();
        copyDesc.srcPitch = mipWidth * pixelStride * sizeof(uint16_t);
        copyDesc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copyDesc.dstArray = levelArray;
        copyDesc.WidthInBytes = mipWidth * pixelStride * sizeof(uint16_t);
        copyDesc.Height = mipHeight;
        CUDA_CHK( cuMemcpy2D( &copyDesc ) );
    }

    // Create texture object
    CUDA_RESOURCE_DESC resDesc = {};
    resDesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    resDesc.res.mipmap.hMipmappedArray = mipmappedArray;
    
    CUDA_TEXTURE_DESC texDesc = {};
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_READ_AS_INTEGER;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = (float)( numMips - 1 );
    texDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    texDesc.maxAnisotropy = 16;
    
    CUtexObject texObject;
    CUDA_CHK( cuTexObjectCreate( &texObject, &resDesc, &texDesc, nullptr ) );
    m_inferenceData.latentTexture = texObject;
    return true;
}

bool OptixDecompressor::ReadLatentLayer( uint8_t* dest, int destSize, ntc::LatentTextureFootprint& layerFootprint )
{
    CHK( m_texSetStream->Seek( layerFootprint.buffer.rangeInStream.offset ) );

    const ntc::CompressionType compType = layerFootprint.buffer.compressionType;
    if( layerFootprint.buffer.compressionType == ntc::CompressionType::None )
    {
        CHK( m_texSetStream->Read( dest, destSize ) );
    }
    else
    {
        std::vector<uint8_t> compData( layerFootprint.buffer.rangeInStream.size );
        CHK( m_texSetStream->Read( compData.data(), compData.size() ) );
        const int crc32 = layerFootprint.buffer.uncompressedCrc32;
        NTC_CHK( m_ntcContext->DecompressBuffer( compType, compData.data(), compData.size(), dest, destSize, crc32 ) );
    }

    return true;
}

bool OptixDecompressor::ReadLatentMipLevelUshort( std::vector<uint16_t>& dest, int mipLevel, int mipWidth, int mipHeight )
{
    int numLatentTextures = m_inferenceData.latentFeatures / 4;
    int destPixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;

    // Read all of the latent texture layers into a single buffer
    int latentLayerSizeInBytes = mipWidth * mipHeight * sizeof(uint16_t);
    std::vector<uint8_t> latentData( latentLayerSizeInBytes * numLatentTextures, 0 );
    for( int layer = 0; layer < numLatentTextures; ++layer )
    {
        ntc::LatentTextureFootprint latentTextureFootprint;
        NTC_CHK( m_texSetMetadata->GetLatentTextureFootprint( mipLevel, layer, latentTextureFootprint ) );
        CHK( ReadLatentLayer( latentData.data() + latentLayerSizeInBytes * layer, latentLayerSizeInBytes, latentTextureFootprint ) );
    }

    // Copy the data to the destination, interleaving the layers
    uint16_t* src = (uint16_t*)latentData.data();
    for( int y = 0; y < mipHeight; ++y )
    {
        for( int x = 0; x < mipWidth; ++x )
        {
            uint16_t* pixelDest = &dest[( y * mipWidth + x ) * destPixelStride];
            for( int layer = 0; layer < numLatentTextures; ++layer )
            {
                int srcLayerOffset = mipWidth * mipHeight * layer;
                int srcPixelOffset = y * mipWidth + x;
                uint16_t* pixelSrc = &src[srcLayerOffset + srcPixelOffset];
                pixelDest[layer] = *pixelSrc;
            }
        }
    }

    return true;
}


bool OptixDecompressor::MakeOptixPipeline()
{
    //
    // Create module
    //
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};

        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues      = 2;
        pipeline_compile_options.numAttributeValues    = 2;
        pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        OPTIX_CHECK_LOG( optixModuleCreate(
            m_optixContext,
            &module_compile_options,
            &pipeline_compile_options,
            reinterpret_cast<const char*>( CUDA_PTX ), // From OptixDecompressionKernelPtx.h
            sizeof( CUDA_PTX ),
            LOG, &LOG_SIZE,
            &module
        ) );
    }

    //
    // Create program groups, including NULL miss and hitgroups
    //
    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    {
        OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc  = {}; //
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;

        const char* raygen = "__raygen__decompress_optix";

        raygen_prog_group_desc.raygen.entryFunctionName = raygen;

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            m_optixContext,
            &raygen_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &raygen_prog_group
        ) );

        // Leave miss group's module and entryfunc name null
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            m_optixContext,
            &miss_prog_group_desc,
            1,   // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &miss_prog_group
        ) );
    }

    //
    // Link pipeline
    //
    {
        const uint32_t    max_trace_depth  = 0;
        OptixProgramGroup program_groups[] = { raygen_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth            = max_trace_depth;
        OPTIX_CHECK_LOG( optixPipelineCreate(
            m_optixContext,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof( program_groups ) / sizeof( program_groups[0] ),
            LOG, &LOG_SIZE,
            &m_optixPipeline
        ) );

        OptixStackSizes stack_sizes = {};
        for( auto& prog_group : program_groups )
        {
            OPTIX_CHK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, m_optixPipeline ) );
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
            0,  // maxCCDepth
            0,  // maxDCDEpth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state, &continuation_stack_size
        ) );

        OPTIX_CHK( optixPipelineSetStackSize( m_optixPipeline, direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state, continuation_stack_size,
            2  // maxTraversableDepth
        ) );
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof( RayGenSbtRecord );
        CUDA_CHK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &raygen_record ), raygen_record_size ) );
        RayGenSbtRecord rg_sbt;
        OPTIX_CHK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        rg_sbt.data = 0;
        CUDA_CHK( cuMemcpyHtoD( raygen_record, &rg_sbt, raygen_record_size ) );

        CUdeviceptr miss_record;
        size_t      miss_record_size = sizeof( MissSbtRecord );
        CUDA_CHK( cuMemAlloc( reinterpret_cast<CUdeviceptr*>( &miss_record ), miss_record_size ) );
        MissSbtRecord ms_sbt;
        OPTIX_CHK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
        ms_sbt.data = 0;
        CUDA_CHK( cuMemcpyHtoD( miss_record, &ms_sbt, miss_record_size ) );

        m_sbt.raygenRecord                = raygen_record;
        m_sbt.missRecordBase              = miss_record;
        m_sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
        m_sbt.missRecordCount             = 1;
    }

    return true;
}

const char* OptixDecompressor::DecompressMipLevel( half* outputImage, int mipLevel, float& timeInMilliseconds )
{
    {
        OptixDecompressionParams optixParams{ m_inferenceData, outputImage, mipLevel };
        CUstream stream;
        CUDA_CHK_MSG( cuStreamCreate( &stream, 0 ), "cuStreamCreate failed." );

        CUdeviceptr d_param;
        CUDA_CHK_MSG( cuMemAlloc( &d_param, sizeof( OptixDecompressionParams ) ), "cuMemAlloc failed." );
        CUDA_CHK_MSG( cuMemcpyHtoD( d_param, &optixParams, sizeof( OptixDecompressionParams ) ), "cuMemcpyHtoD failed." );

        // Time decompression
        CUevent eventStart = nullptr;
        CUevent eventEnd = nullptr;
        CUDA_CHK_MSG( cuEventCreate( &eventStart, 0 ), "cuEventCreate failed." );
        CUDA_CHK_MSG( cuEventCreate( &eventEnd, 0 ), "cuEventCreate failed." );
        CUDA_CHK_MSG( cuEventRecord( eventStart, stream ), "cuEventRecord failed." );

        NtcTextureSetConstants& tsc = optixParams.inferenceData.constants;
        int launchWidth = std::max( 1, tsc.imageWidth >> mipLevel );
        int launchHeight = std::max( 1, tsc.imageHeight >> mipLevel );
        int launchDepth = 1;

        OPTIX_CHK_MSG( optixLaunch(
            m_optixPipeline,
            stream,
            d_param,
            sizeof( OptixDecompressionParams ),
            &m_sbt,
            launchWidth,
            launchHeight,
            launchDepth ),
            "OptiX launch failed."
        );

        // Time decompression
        CUDA_CHK_MSG( cuEventRecord( eventEnd, stream ), "cuEventRecord failed." );
        CUDA_CHK_MSG( cuEventSynchronize( eventEnd ), "cuEventSynchronize failed." );
        CUDA_CHK_MSG( cuEventElapsedTime( &timeInMilliseconds, eventStart, eventEnd ), "cuEventElapsedTime failed." );
        CUDA_CHK_MSG( cuEventDestroy( eventStart ), "cuEventDestroy failed." );
        CUDA_CHK_MSG( cuEventDestroy( eventEnd ), "cuEventDestroy failed." );

        CUDA_CHK_MSG( cuMemFree( d_param ), "cuMemFree failed." );
    }

    return nullptr;
}
