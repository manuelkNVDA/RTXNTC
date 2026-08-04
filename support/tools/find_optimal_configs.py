#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument('--input', required = True, help = 'Path to the CSV file with input data')
args = parser.parse_args()

class Bucket:
    def __init__(self):
        self.configs = []

class Configuration:
    def __init__(self):
        self.psnrValues = []

bppBuckets = {}
allowedBppValues = [
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0
]

with open(args.input, 'r') as inputFile:
    headers = inputFile.readline().strip().split(',')

    ci_GridSizeScale = headers.index('GridSizeScale')
    ci_HighResFeatures = headers.index('HighResFeatures')
    ci_LowResFeatures = headers.index('LowResFeatures')
    ci_HighResQuantBits = headers.index('HighResQuantBits')
    ci_LowResQuantBits = headers.index('LowResQuantBits')
    ci_PSNR = []
    for index in range(len(headers)):
        if headers[index].startswith('PSNR'):
            ci_PSNR.append(index)

    lineIndex = 2
    for line in inputFile:
        items = line.strip().split(',')
        if len(items) != len(headers):
            print(f'ERROR: line {lineIndex} item count mismatch! Expected {len(headers)}, got {len(items)}.')
            lineIndex += 1
            continue
        
        config = Configuration()
        config.GridSizeScale = int(items[ci_GridSizeScale])
        config.HighResFeatures = int(items[ci_HighResFeatures])
        config.LowResFeatures = int(items[ci_LowResFeatures])
        config.HighResQuantBits = int(items[ci_HighResQuantBits])
        config.LowResQuantBits = int(items[ci_LowResQuantBits])
        config.psnrValues = [float(items[index]) for index in ci_PSNR]

        bpp = (config.HighResFeatures * config.HighResQuantBits 
            + config.LowResFeatures * config.LowResQuantBits * 0.25) / (config.GridSizeScale * config.GridSizeScale)
        if bpp not in allowedBppValues:
            continue

        bucket = bppBuckets.get(bpp)
        if bucket is None:
            bucket = Bucket()
            
        bucket.configs.append(config)
        bppBuckets[bpp] = bucket

        lineIndex += 1

for (bpp, bucket) in sorted(bppBuckets.items(), key=itemgetter(0)):
    print(f'{{ {bpp:6.3f}f', end='')
    for maxHrFeatures in range(4, 20, 4):
        maxPsnrValues = []
        for config in bucket.configs:
            if config.HighResFeatures > maxHrFeatures:
                continue
            if len(maxPsnrValues) == 0:
                maxPsnrValues = config.psnrValues
            else:
                maxPsnrValues = [max(old, new) for old, new in zip(maxPsnrValues, config.psnrValues)]
        
        if len(maxPsnrValues) == 0:
            print(', { 0,  0,  0, 0, 0 }', end='')
            continue
        
        bestConfig = None
        bestScore = None
        for config in bucket.configs:
            if config.HighResFeatures > maxHrFeatures:
                continue
            score = sum([psnr - maxPsnr for maxPsnr, psnr in zip(maxPsnrValues, config.psnrValues)])
            if bestScore is None or score > bestScore:
                bestConfig = config
                bestScore = score

        print(f', {{ {bestConfig.GridSizeScale}, {bestConfig.HighResFeatures:2}, {bestConfig.LowResFeatures:2}, '
            + f'{bestConfig.HighResQuantBits}, {bestConfig.LowResQuantBits} }}', end='')    
    print(' },')
   
