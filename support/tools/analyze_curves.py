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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# NOTE: This script was used in the development of the adaptive compression feature to analyze PSNR dependencies
# and experiment with the BPP search algorithm.
# It is not intended for any other use.


matplotlib.use('TkAgg')

use_log_model = True

@dataclass
class Experiment:
    material : str
    bpp : float
    curve : np.array
    final : float
    time : float = None

# Loads a list of experiment results from a CSV produced by psnr_study.py
def load_data_from_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rowIndex = 0
        rows = []
        indexBias = 0
        finalColumn = -1
        timeColumn = None
        for row in reader:
            if rowIndex > 0:
                row = row[indexBias:]
                rows.append(Experiment(
                    material=row[0],
                    bpp=float(row[1][:-3]) if row[1].endswith('bpp') else float(row[1]),
                    curve=np.array([float(x) for x in row[2:-1]]),
                    final=float(row[finalColumn]),
                    time=float(row[timeColumn]) if timeColumn is not None else None
                ))
            else:
                indexBias = 1 if row[0] == "Ordinal" else 0
                if row[-1] == "Time(s)":
                    finalColumn = -2
                    timeColumn = -1
            rowIndex += 1
    return rows
    
# Model for PSNR vs. time or PSNR vs. BPP dependencies
def model(x, a, b):
    if use_log_model:
        return a * np.log(x) + b
    else:
        return a * x + b

# Inverse of the 'model' function
def inverse_model(y, a, b):
    if use_log_model:
        return np.exp((y - b) / a)
    else:
        return (y - b) / a
    
# Find the values of (a, b) for the model from two points
def get_model_params(x1, x2, y1, y2) -> Tuple[float, float]:
    if use_log_model:
        x1 = np.log(x1)
        x2 = np.log(x2)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return (a, b)
    
    
def build_approximate_training_curve(x, y, pstart, fraction):
    pend = int(len(y) * fraction)
    p0 = (2.0, y[pend-1])
    param, param_cov = opt.curve_fit(model, x[pstart:pend], y[pstart:pend], p0=p0)
    result = model(x, *param)
    finalDiff = result[-1] - y[-1]
    return result, finalDiff


def predict_training_curves(rows: list[Experiment]):
    fraction = .25
    rowIndex = 0
    dataToPlot = []
    x = None
    for experiment in rows:
        if x is None:
            x = np.linspace(0.01, 1.0, len(experiment.curve))
        approximation, finalDiff = build_approximate_training_curve(x, experiment.curve, pstart=10, fraction=fraction)
        print(f'{experiment.material:40} {experiment.bpp:4.1f} bpp: final = {experiment.curve[-1]:.2f} dB, error = {finalDiff:.2f} dB')
        dataToPlot.append((experiment, approximation))

        rowIndex += 1
        if rowIndex > 6:
            break
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axvline(x=fraction)
    for experiment, approximation in dataToPlot:
        ax.semilogx(x, experiment.curve)
        ax.semilogx(x, approximation)
    
    plt.xlabel('Training time')
    plt.ylabel('PSNR')
    plt.title(f'Analytic functions fitted to the first {fraction*100:.0f}% of training curves')
    #plt.savefig('data/plot.png')
    plt.show()


# Extracts materials and their sorted experiment lists from the overall list of experiments
def get_unique_materials(rows: list[Experiment]) -> Dict[str, List[Experiment]]:
    materials = {}
    for experiment in rows:
        if experiment.material not in materials:
            materials[experiment.material] = []

    for material in materials:
        mat_experiments = [experiment for experiment in rows if experiment.material == material ]
        mat_experiments.sort(key=lambda experiment: experiment.bpp)
        materials[material] = mat_experiments

    return materials

# Plots per-material PSNR vs. BPP curves and their approximations
def plot_bpp_curves(materials: Dict[str, List[Experiment]]):
    fig, ax = plt.subplots(figsize=(16, 12))
    index = 0
    for material, mat_experiments in materials.items():
        x = [e.bpp for e in mat_experiments]
        y = [e.final for e in mat_experiments]
        first_point = 0
        second_point = x.index(4)
        third_point = x.index(8)
        #param, param_cov = opt.curve_fit(model, [x[first_point], x[second_point]], [y[first_point], y[second_point]])
        param, param_cov = opt.curve_fit(model, [x[second_point], x[third_point]], [y[second_point], y[third_point]])
        approximation = model(x, *param)
        ax.plot(x, y)
        ax.plot(x, approximation)

        print(f'{material:40}: {param[0]:4.2f}, {param[1]:5.2f}')
        #print(f'{material}: {[e.final for e in mat_experiments]}')
        index += 1
        #if index > 3:
        #    break
    
    plt.xlabel('BPP')
    plt.ylabel('PSNR')
    plt.title('')
    plt.show()
    
    return

# Prototype for the Adaptive Compression BPP search algorithm.
# Simulates making a small number of experiments to search for the right BPP given a target PSNR,
# validates the results against a simulated brute force search.
def find_bpps(materials: Dict[str, List[Experiment]]):
    total_experiments = 0
    total_time = 0

    for material, mat_experiments in materials.items():
        for target_psnr in range(30, 51, 5):
        
            # Finds the experiment result at the BPP most closely matching the given one.
            # If the exclude_left or exclude_right parameters are given, avoids finding bpp <= left or >= right.
            def get_result_at_closest_bpp(
                bpp: float,
                exclude_left: Optional[float] = None,
                exclude_right: Optional[float] = None
            ) -> Experiment:
                best = None
                for e in mat_experiments:
                    if exclude_left is not None and e.bpp <= exclude_left:
                        continue
                    if exclude_right is not None and e.bpp >= exclude_right:
                        break
                    if best is None or abs(e.bpp - bpp) < abs(best.bpp - bpp):
                        best = e
                return best
            
            # Finds the ideal BPP for the current target_psnr using brute force linear search.
            def get_ideal_result() -> Experiment:
                best = None
                for e in mat_experiments:
                    if e.final >= target_psnr:
                        best = e
                        break
                if best is None:
                    best = mat_experiments[-1]
                return best
                
            # Start the interpolation search alrogithm.
            # Begin at some *fast* midpoint to see if we need to go up or down from that.
            midpoint = get_result_at_closest_bpp(3.5) # 3.5 is significantly faster than 4.0
            if midpoint.final >= target_psnr:
                right = midpoint
                left = get_result_at_closest_bpp(0.5)
            else:
                left = midpoint
                right = get_result_at_closest_bpp(20.0)

            # Run the interpolation search loop, count our experiments and the time it would take us to make them.
            experiment_count = 2
            experiment_time = left.time + right.time
            while right.bpp > left.bpp:
                # Fit a model curve to the current boundaries
                param = get_model_params(left.bpp, right.bpp, left.final, right.final)
                
                # Predict the optimal BPP using the fitted model
                expected_bpp = inverse_model(target_psnr, *param)

                # Find a real BPP value most closely matching the predicted BPP, but excluding the left and right points.
                expected_point = get_result_at_closest_bpp(expected_bpp, exclude_left=left.bpp, exclude_right=right.bpp)

                # If the prediction is not matching any real point between left and right, stop the search.
                if expected_point is None:
                    if left.final >= target_psnr:
                        right = left
                    else:
                        left = right
                    break

                # Now we 'make' the experiment and measure the PSNR
                experiment_count += 1
                experiment_time += expected_point.time

                # Update the boundaries according to the experiment result
                if expected_point.final >= target_psnr:
                    right = expected_point
                else:
                    left = expected_point
                
            # Validate the interpolation search result using the brute force result
            ideal_point = get_ideal_result()
            if ideal_point.bpp == right.bpp:
                result = 'OK'
            elif ideal_point.bpp > right.bpp:
                result = 'FAIL'
            else:
                result = 'SUBOPT'

            # Print out the status
            print(f'{material:45}: target = {target_psnr:.2f} dB, found {right.final:.2f} dB at {right.bpp:5.2f} bpp, '
                f'ideal {ideal_point.final:.2f} dB at {ideal_point.bpp:5.2f} bpp, '
                f'{experiment_count} experiments in {experiment_time:5.1f} seconds - {result}')
            
            # Update overall statistics
            total_experiments += experiment_count
            total_time += experiment_time

    print(f'Total: {total_experiments} experiments in {total_time:.2f} seconds')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_curves.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    rows = load_data_from_csv(filename)
    materials = get_unique_materials(rows)
    find_bpps(materials)
    #plot_bpp_curves(materials)
    #predict_training_curves(rows)
    