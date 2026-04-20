#!/usr/bin/env python3

import os
import re
import json
import csv
from pathlib import Path

# ============================================================
# Fit diagonal coefficients from isolation tests
#
# Model for each task i:
#   RT_i = a_i + b_ii * T_i
#
# where:
#   RT_i = avg_response_ms
#   T_i  = task period in ms
#
# Reads directories like:
#   logs/diag/mm/
#   logs/diag/bfs/
# etc.
# ============================================================

BASE_DIR = Path("logs/diag")
OUTPUT_CSV = BASE_DIR / "diag_coefficients.csv"

TASKS = ["mm", "stereo", "quasi", "hist", "particle", "bfs"]
TASKS = ["bfs", "hist", "mm", "particle", "quasi", "stereo"]


def linear_fit(x, y):
    """
    Fit y = a + b x
    Returns:
        intercept, slope, r2
    """
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for a linear fit")

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))

    if sxx == 0:
        raise ValueError("All x values are identical; cannot compute slope")

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    y_pred = [intercept + slope * xi for xi in x]
    ss_res = sum((yi - yhat) ** 2 for yi, yhat in zip(y, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return intercept, slope, r2


def parse_period_ms(filename):
    """
    Example:
      mm_70ms__mm_sweep_70.json  -> 70
      bfs_100ms__bfs_sweep_100.json -> 100
    """
    m = re.search(r"_(\d+)ms__", filename)
    if m:
        return int(m.group(1))

    m = re.search(r"_sweep_(\d+)", filename)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse period from filename: {filename}")


def read_avg_response_ms(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "avg_response_ms" not in data:
        raise KeyError(f"avg_response_ms missing in {json_path}")

    return float(data["avg_response_ms"])


def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    results = []

    for task in TASKS:
        task_dir = BASE_DIR / task
        if not task_dir.exists():
            print(f"Skipping {task}: directory not found ({task_dir})")
            continue

        x_period_ms = []
        y_rt_ms = []

        json_files = sorted(task_dir.glob("*.json"))
        if not json_files:
            print(f"Skipping {task}: no JSON files found")
            continue

        for jf in json_files:
            try:
                period_ms = parse_period_ms(jf.name)
                avg_rt_ms = read_avg_response_ms(jf)
            except Exception as e:
                print(f"Skipping file {jf}: {e}")
                continue

            x_period_ms.append(period_ms)
            y_rt_ms.append(avg_rt_ms)

        if len(x_period_ms) < 2:
            print(f"Skipping {task}: not enough valid points")
            continue

        try:
            intercept, slope, r2 = linear_fit(x_period_ms, y_rt_ms)
        except Exception as e:
            print(f"Fit failed for {task}: {e}")
            continue

        results.append({
            "task": task,
            "num_points": len(x_period_ms),
            "intercept_ms": intercept,
            "diagonal_coefficient_slope_ms_per_ms": slope,
            "r2": r2,
            "x_points_ms": x_period_ms,
            "y_points_ms": y_rt_ms,
            "directory": str(task_dir),
        })

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "num_points",
            "intercept_ms",
            "diagonal_coefficient_slope_ms_per_ms",
            "r2",
            "directory",
        ])
        for row in results:
            writer.writerow([
                row["task"],
                row["num_points"],
                row["intercept_ms"],
                row["diagonal_coefficient_slope_ms_per_ms"],
                row["r2"],
                row["directory"],
            ])

    # Print results
    print("\nDiagonal coefficients:")
    print("=" * 80)
    for row in results:
        print(
            f"{row['task']:<10} : "
            f"slope = {row['diagonal_coefficient_slope_ms_per_ms']:+.6f}   "
            f"intercept = {row['intercept_ms']:.6f}   "
            f"R^2 = {row['r2']:.4f}"
        )

    print("\nDiagonal entries of B_RT<-task:")
    print("=" * 80)
    for row in results:
        print(f"b[{row['task']},{row['task']}] = {row['diagonal_coefficient_slope_ms_per_ms']:+.6f}")

    print(f"\nSaved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
