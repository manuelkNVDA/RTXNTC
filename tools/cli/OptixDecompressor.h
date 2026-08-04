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
#pragma once

#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <optix.h>

#include <libntc/ntc.h>
#include <libntc/shaders/InferenceConstants.h>
#include <libntc/shaders/InferenceDataOptix.h>

struct OptixDecompressionParams;

const uint32_t NTC_NETWORK_MAX_SIZE = 32768;

class OptixDecompressor
{
  public:
    /// Make an optix decompressor for the given texture set
    OptixDecompressor( ntc::IContext* ntcContext, ntc::ITextureSetMetadata* texSet, ntc::IStream* texSetStream );

    /// Destroy the decompressor
    virtual ~OptixDecompressor();

    /// Prepare the decompression. Return nullptr on success, or an error message on failure
    const char* prepareDecompression();

    /// Decompress a mip level and store the result outputImage.
    /// Return nullptr on success, or an error message on failure
    const char* DecompressMipLevel( half* outputImage, int mipLevel, float& timeInMilliseconds );

  private:
    ntc::IContext* m_ntcContext = nullptr;
    ntc::ITextureSetMetadata* m_texSetMetadata = nullptr;
    ntc::IStream* m_texSetStream = nullptr;
    OptixDeviceContext m_optixContext = 0;
    OptixPipeline m_optixPipeline = nullptr;
    OptixShaderBindingTable m_sbt = {};
    InferenceDataOptix m_inferenceData{};

    bool MakeOptixContext();
    bool MakeOptixInferenceData();
    bool MakeOptixNetworkData();
    bool ConvertNetworkToInferencingOptimal( CUdeviceptr d_srcNetworkData, int srcNetworkTotalSize, CUdeviceptr d_dstMatrix, int d_dstSize );
    bool MakeLatentTexture();
    bool ReadLatentLayer( uint8_t* dest, int destSize, ntc::LatentTextureFootprint& layerFootprint );
    bool ReadLatentMipLevelUshort( std::vector<uint16_t>& dest, int mipLevel, int width, int height );
    bool MakeOptixPipeline();
};
