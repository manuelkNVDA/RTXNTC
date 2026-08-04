#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import sys
import PIL
import PIL.Image
import csv

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

root = ntc.get_sdk_root_path()
defaultTool = ntc.get_default_tool_path()

parser = argparse.ArgumentParser()
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
parser.add_argument('--dataset', required = True, help = 'Path to the directory with test materials')
parser.add_argument('--trainingSteps', type = int, default = 100000, help = 'Number of training steps')
parser.add_argument('--randomSeed', type = int, default = 0, help = 'Random seed for training')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.dataset):
    print(f"The specified dataset path '{args.dataset}' does not exist.", file = sys.stderr)
    sys.exit(1)
    
ordinal = 0
count = 0
tasks = []
for (dirname, subdirs, files) in os.walk(args.dataset):
    if dirname.startswith('.\\'):
        dirname = dirname[2:]

    for filename in files:
        if os.path.splitext(filename)[1] not in ('.png', '.jpg', '.jpeg', '.tga', '.bmp'):
            continue

        filename = os.path.join(dirname, filename)

        try:
            # Get the number of channels in the original image
            with PIL.Image.open(filename) as img:
                numChannels = len(img.getbands())
        except:
            print(f'Failed to open {filename}, skipping.')
            continue

        if numChannels == 1:
            bcFormat = 'bc4'
        elif numChannels == 2:
            bcFormat = 'bc5'
        else:
            bcFormat = 'bc7'
        
        task = ntc.Arguments(tool=args.tool)
        task.customArguments = filename
        task.compress = True
        task.decompress = True
        task.matchBcPsnr = True
        task.graphicsApi = 'vk'
        task.bcFormat = bcFormat
        task.trainingSteps = args.trainingSteps
        task.randomSeed = args.randomSeed
        tasks.append(task)
    
writer = csv.writer(sys.stdout)
writer.writerow(['Ordinal', 'Name', 'Format', 'BC PSNR', 'BC BPP', 'NTC PSNR', 'NTC BPP'])

def task_ready(task: ntc.Arguments, result: ntc.Result, originalTaskCount: int, completedTaskCount: int):
    writer.writerow([
        completedTaskCount,
        task.customArguments,
        task.bcFormat.upper(),
        result.combinedBcPsnr,
        result.combinedBcBitsPerPixel,
        result.overallPsnrFP8,
        result.bitsPerPixel
    ])
    
terminated = ntc.process_concurrent_tasks(tasks, args.devices, task_ready)
if terminated:
    sys.exit(1)
