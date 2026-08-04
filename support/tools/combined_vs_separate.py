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
import time
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
parser.add_argument('--limit', type = int, help = 'Maximum number of materials to test')
parser.add_argument('--skip', type = int, help = 'Skip the first N materials')
parser.add_argument('--stride', type = int, help = 'Test every Nth material in the dataset')
parser.add_argument('--combinedBitsPerPixel', type = float, default = 15, help = 'Bitrate for the combined texture sets')
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
    # We only want directories that contain image files and no other subdirectories
    if len(files) == 0 or len(subdirs) != 0:
        continue
    
    ordinal += 1
    if args.stride and (ordinal % args.stride != 0):
        continue
    if args.skip and (ordinal < args.skip):
        continue

    if dirname.startswith('.\\'):
        dirname = dirname[2:]

    task = ntc.Arguments(tool=args.tool)
    task.loadImages = dirname
    task.compress = True
    task.decompress = True
    task.describe = True
    task.keepFileNames = True
    task.bitsPerPixel = args.combinedBitsPerPixel
    task.trainingSteps = args.trainingSteps
    task.randomSeed = args.randomSeed
    tasks.append(task)

    count += 1
    if args.limit and count >= args.limit:
        break
    
print('Collecting combined PSNR values...')

combined_psnr_values = []

def combined_task_ready(task: ntc.Arguments, result: ntc.Result, originalTaskCount: int, completedTaskCount: int):
    print(f'[{completedTaskCount}/{originalTaskCount}] {task.loadImages}: {result.dimensions[0]}x{result.dimensions[1]}')
    for name, psnr in result.perTexturePsnr.items():
        print(f'  {name}: {psnr} dB')
    combined_psnr_values.append((task.loadImages, result.dimensions, result.perTexturePsnr))
    
terminated = ntc.process_concurrent_tasks(tasks, args.devices, combined_task_ready)
if terminated:
    sys.exit(1)

print()

tasks = []
for dirname, dimensions, perTexturePsnr in combined_psnr_values:
    for texture, psnr in perTexturePsnr.items():
        task = ntc.Arguments(tool=args.tool)
        task.customArguments = os.path.join(dirname, texture)
        task.dimensions = f'{dimensions[0]}x{dimensions[1]}'
        task.compress = True
        task.decompress = True
        task.targetPsnr = psnr
        task.trainingSteps = args.trainingSteps
        task.randomSeed = args.randomSeed
        tasks.append((task, dirname, texture))

print('Searching for optimal per-texture BPP values...')

split_psnr_values = {}

def split_task_ready(taskTuple: ntc.Arguments, result: ntc.Result, originalTaskCount: int, completedTaskCount: int):
    task, dirname, texture = taskTuple
    if dirname not in split_psnr_values:
        split_psnr_values[dirname] = {}
    split_psnr_values[dirname][texture] = (task.targetPsnr, result.overallPsnrFP8, result.bitsPerPixel)
    print(f'[{completedTaskCount}/{originalTaskCount}] {dirname}/{texture}: expected {task.targetPsnr} dB, got {result.overallPsnrFP8} dB, bitrate {result.bitsPerPixel} bpp')

terminated = ntc.process_concurrent_tasks(tasks, args.devices, split_task_ready)
if terminated:
    sys.exit(1)

print()

for dirname, textures in split_psnr_values.items():
    sum_of_bpps = 0
    for name, (expected_psnr, actual_psnr, bpp) in textures.items():
        sum_of_bpps += bpp
    ratio = sum_of_bpps / args.combinedBitsPerPixel
    print(f'{dirname}: combined bitrate is {sum_of_bpps} bpp, ratio {ratio:.2f}x')
