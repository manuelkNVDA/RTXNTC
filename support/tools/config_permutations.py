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
import csv
import itertools
import os
import sys

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

defaultTool = ntc.get_default_tool_path()

parser = argparse.ArgumentParser()
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
parser.add_argument('--images', required = True, help = 'Path to the directory with test images')
parser.add_argument('--output', required = True, help = 'Path to the output directory')
parser.add_argument('--only', type = int, help = 'Run only one test by ordinal')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use, such as --devices 0 1')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.images):
    print(f"The specified image path '{args.images}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.output):
    os.makedirs(args.output)

baseName = os.path.basename(args.images)

scales = [2, 4]
features = [4, 8, 12, 16]
quantBits = [1, 2, 4, 8]
ordinal = 1

tasks = []

for (gridSizeScale, highResFeatures, lowResFeatures, highResQuantBits, lowResQuantBits) in itertools.product(scales, features, features, quantBits, quantBits):

    if args.only is not None and ordinal != args.only:
        ordinal += 1
        continue

    outputFileName = f'{baseName}_SCALE{gridSizeScale}_HR{highResFeatures:02}_LR{lowResFeatures:02}_HRQ{highResQuantBits}_LRQ{lowResQuantBits}.ntc'
    outputFilePath = os.path.join(args.output, outputFileName)

    ntcArgs = ntc.Arguments(
        tool=args.tool,
        loadImages=args.images,
        compress=True,
        decompress=True,
        saveCompressed=outputFilePath,
        latentShape=ntc.LatentShape(
            gridSizeScale=gridSizeScale,
            highResFeatures=highResFeatures,
            lowResFeatures=lowResFeatures,
            highResQuantBits=highResQuantBits,
            lowResQuantBits=lowResQuantBits)
    )
    
    tasks.append((ntcArgs, ordinal, outputFileName))
    ordinal = 1

writer = csv.writer(sys.stdout)

writer.writerow(['Ordinal', 'FileName', 'BPP', 'GridSizeScale', 'HighResFeatures',
                 'LowResFeatures', 'HighResQuantBits', 'LowResQuantBits', 'PSNR'])

def ready(task, result: ntc.Result, originalTaskCount: int, tasksCompleted: int):
    ntcArgs, ordinal, outputFileName = task
    latentShape: ntc.LatentShape = ntcArgs.latentShape
    writer.writerow([ ordinal, outputFileName, result.savedFileBpp, latentShape.gridSizeScale, 
                     latentShape.highResFeatures, latentShape.lowResFeatures, latentShape.highResQuantBits,
                     latentShape.lowResQuantBits, result.overallPsnr])

terminated = ntc.process_concurrent_tasks(tasks, args.devices, ready)
