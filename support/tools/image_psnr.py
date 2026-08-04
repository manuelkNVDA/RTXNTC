#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import numpy
from PIL import Image

def _loadImage(filename):
    image = Image.open(filename)
    return numpy.array(image, dtype=numpy.int16)

def computePSNR(img_path1, img_path2):
    "Loads two common format image files and returns PSNR between them in dB"
    
    # Load the images
    arr1 = _loadImage(img_path1)
    arr2 = _loadImage(img_path2)

    if len(arr1.shape) == 2: arr1 = numpy.expand_dims(arr1, 2)
    if len(arr2.shape) == 2: arr2 = numpy.expand_dims(arr2, 2)

    # Compute mean squared error (MSE) between the two images
    mse = numpy.mean((arr1 - arr2) ** 2)
    
    # Compute PSNR from MSE, assuming that image data is in 0-255 integers
    psnr = 10 * numpy.log10(255**2 / mse)

    return mse, psnr

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Calculates the MSE and PSNR between two image files of equal dimensions.')
        print('Usage: image_psnr.py <image1> <image2>')
        sys.exit(1)

    mse, psnr = computePSNR(sys.argv[1], sys.argv[2])
    
    print(f'MSE = {mse:.2f}, PSNR = {psnr:.2f} dB')
