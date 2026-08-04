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

#include <stdint.h>

#include <cuda.h>
#include <optix.h>
#include <optix_device.h>

#include <libntc/shaders/InferenceOptix.h>
#include "OptixDecompressionKernel.h"

extern "C" {
__constant__ OptixDecompressionParams params;
}

// Computes the address (offset into the textureData array for one mip) for a channel in the pixel.
// See RegressionKernels.h
inline __device__ int GetChannelAddress(int pixelBaseAddress, int channel, int width)
{
    return pixelBaseAddress + (channel & ~1) * width + (channel & 1);
}

// Computes the address (offset into the textureData array for one mip) for a given pixel in the texture data.
// See the comment to PitchLinearImageSlice structure for the texture data layout explanation.
inline __device__ int GetPixelBaseAddress(int x, int y, int width, int numChannels)
{
    return y * width * numChannels + x * 2;
}

// Optix Raygen program
extern "C" __global__ void __raygen__decompress_optix()
{
    uint3 launch_index = optixGetLaunchIndex();
    const int x = launch_index.x;
    const int y = launch_index.y;

    // Make sure the launch index is in the mip level bounds
    InferenceDataOptix& inf = params.inferenceData;
    const int mipWidth = inf.constants.imageWidth >> params.mipLevel;
    const int mipHeight = inf.constants.imageHeight >> params.mipLevel;
    if( x >= mipWidth || y >= mipHeight )
        return;

    // Infer the texel
    using T_VEC_OUT = OptixCoopVec<float, NTC_MLP_OUTPUT_CHANNELS>;
    T_VEC_OUT texelData;
    SampleTextureSet<T_VEC_OUT>( texelData, inf.constants, inf.latentTexture, inf.latentFeatures,
        inf.latentWidth, inf.latentHeight, inf.d_mlpWeights, x, y, params.mipLevel );

    const int pixelBaseAddress = GetPixelBaseAddress( x, mipHeight-1-y, mipWidth, inf.numChannels );

    // Copy the texel channels to the output image
#pragma unroll
    for ( int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; i++ )
    {
        if ( i < inf.numChannels )
        {
            int channelAddress = GetChannelAddress( pixelBaseAddress, i, mipWidth );
            params.outputImage[channelAddress] = texelData[i];
        }
    }
}

