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

import os
import sys
import argparse
import csv

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

defaultTool = ntc.get_default_tool_path()

parser = argparse.ArgumentParser()
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
input_group = parser.add_mutually_exclusive_group(required = True)
input_group.add_argument('--dataset', help = 'Path to the directory with compressed files in *.ntc format')
input_group.add_argument('--input', help = 'Path to the input .ntc file')
parser.add_argument('--api', required = True, help = 'Graphics API(s) to use, comma separated: dx12, vk, cuda')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of graphics adapters to use, such as --devices 0 1')
parser.add_argument('--args', help = 'Extra command line arguments for ntc-cli')
parser.add_argument('--tag', help = 'Tag to identify experiment results in a summary table')
parser.add_argument('--benchmark', type = int, default = 21, help = 'Benchmark iterations for each experiment, defaults to 21')
parser.add_argument('--math-versions', action = 'store_true', help = 'Run experiments on all math versions (FP8 etc.)')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if args.dataset is not None:
    if not os.path.isdir(args.dataset):
        print(f"The specified dataset path '{args.dataset}' does not exist.", file = sys.stderr)
    input_files = [os.path.join(args.dataset, filename) for filename in os.listdir(args.dataset) if filename.endswith('.ntc')]
else:
    if not os.path.isfile(args.input):
        print(f"The specified input file path '{args.input}' does not exist.", file = sys.stderr)
        sys.exit(1)
    input_files = [args.input]

apis = args.api.split(',')

writer = csv.writer(sys.stdout)
writer.writerow(['File', 'GPU', 'API', 'Experiment', 'Time'])

if args.math_versions:
    experiments = [
        ('', 'FP8'),
        ('--no-coopVec', 'DP4a'),
    ]
else:
    experiments = [('', args.tag)]

tasks = []

for input_file in input_files:
    for api in apis:
        for arg, tag in experiments:
            ntcArgs = ntc.Arguments(
                tool=args.tool,
                loadCompressed=input_file,
                decompress=True,
                graphicsApi='' if api == 'cuda' else api,
                benchmark=args.benchmark,
                customArguments=arg + ' ' + (args.args or '')
            )
            tasks.append((ntcArgs, tag))


def ready(task, result: ntc.Result, originalTaskCount: int, tasksCompleted: int):
    ntcArgs, tag = task
    path, filename = os.path.split(ntcArgs.loadCompressed)
    writer.writerow([filename, result.gpuName, result.graphicsApi, tag, result.decompressionTime])

terminated = ntc.process_concurrent_tasks(tasks, args.devices, ready)

if terminated:
    sys.exit(2)
