#!/usr/bin/env python3

import os
import re
import json
import csv
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("logs/offdiag")
OUTPUT_CSV = BASE_DIR / "offdiag_coefficients.csv"

# If True, use only the observed/static task logs in each directory.
# For mm_from_bfs, that means use the mm json files to measure RT_i.
USE_STATIC_LOGS_ONLY = True

# ============================================================
# Helpers
# ============================================================

def linear_fit(x, y):
    """
    Fit y = a + b x
    Returns (intercept, slope, r2)
    """
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points for linear fit")

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))

    if sxx == 0:
        raise ValueError("All x values are identical; cannot fit slope")

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


def parse_pair_dirname(dirname):
    """
    Parse 'mm_from_bfs' -> ('mm', 'bfs')
    """
    m = re.fullmatch(r"(.+)_from_(.+)", dirname)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_swept_period_from_filename(filename):
    """
    Example:
      bfs_100ms__static_100_sweep_100.json -> 100
      bfs_70ms__static_100_sweep_70.json   -> 70
    Prefer the '_sweep_<N>' part if present.
    """
    m = re.search(r"_sweep_(\d+)", filename)
    if m:
        return int(m.group(1))

    m = re.search(r"_(\d+)ms", filename)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not parse swept period from filename: {filename}")


def read_avg_response_ms(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "avg_response_ms" not in data:
        raise KeyError(f"avg_response_ms missing in {json_path}")
    return float(data["avg_response_ms"])


def get_task_from_filename(filename):
    """
    Example:
      mm_100ms__static_100_sweep_70.json -> mm
      bfs_70ms__static_100_sweep_70.json -> bfs
    """
    m = re.match(r"([A-Za-z0-9_]+)_\d+ms__", filename)
    if not m:
        return None
    return m.group(1)


# ============================================================
# Main extraction
# ============================================================

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    results = []
    all_tasks = set()

    for pair_dir in sorted(BASE_DIR.iterdir()):
        if not pair_dir.is_dir():
            continue

        task_i, task_j = parse_pair_dirname(pair_dir.name)
        if task_i is None:
            continue

        all_tasks.add(task_i)
        all_tasks.add(task_j)

        x_period_ms = []
        y_rt_ms = []

        json_files = sorted(pair_dir.glob("*.json"))
        if not json_files:
            print(f"Skipping {pair_dir}: no json files found")
            continue

        for jf in json_files:
            fname = jf.name
            task_in_file = get_task_from_filename(fname)

            if USE_STATIC_LOGS_ONLY:
                # For offdiag mm_from_bfs, use only mm logs
                if task_in_file != task_i:
                    continue

            swept_period_ms = parse_swept_period_from_filename(fname)
            avg_rt_ms = read_avg_response_ms(jf)

            x_period_ms.append(swept_period_ms)
            y_rt_ms.append(avg_rt_ms)

        if len(x_period_ms) < 2:
            print(f"Skipping {pair_dir}: not enough usable points")
            continue

        try:
            intercept, slope, r2 = linear_fit(x_period_ms, y_rt_ms)
        except Exception as e:
            print(f"Skipping {pair_dir}: fit failed: {e}")
            continue

        results.append({
            "observed_task": task_i,
            "swept_task": task_j,
            "num_points": len(x_period_ms),
            "intercept_ms": intercept,
            "coefficient_slope_ms_per_ms": slope,
            "r2": r2,
            "x_points_ms": x_period_ms,
            "y_points_ms": y_rt_ms,
            "directory": str(pair_dir),
        })

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "observed_task",
            "swept_task",
            "num_points",
            "intercept_ms",
            "coefficient_slope_ms_per_ms",
            "r2",
            "directory",
        ])
        for row in results:
            writer.writerow([
                row["observed_task"],
                row["swept_task"],
                row["num_points"],
                row["intercept_ms"],
                row["coefficient_slope_ms_per_ms"],
                row["r2"],
                row["directory"],
            ])

    # Print per-pair results
    print("\nOff-diagonal coefficients:")
    print("=" * 72)
    for row in sorted(results, key=lambda r: (r["observed_task"], r["swept_task"])):
        print(
            f"{row['observed_task']:<10} from {row['swept_task']:<10} : "
            f"slope = {row['coefficient_slope_ms_per_ms']:+.6f}   "
            f"intercept = {row['intercept_ms']:.6f}   "
            f"R^2 = {row['r2']:.4f}"
        )

    # Build matrix-style view
    tasks = sorted(all_tasks)
    coeff_map = {
        (r["observed_task"], r["swept_task"]): r["coefficient_slope_ms_per_ms"]
        for r in results
    }

    print("\nMatrix-style off-diagonal slope table (rows=observed RT_i, cols=swept task_j):")
    print("=" * 72)
    header = ["task_i\\task_j"] + tasks
    print("".join(f"{h:>14}" for h in header))

    for ti in tasks:
        row_str = f"{ti:>14}"
        for tj in tasks:
            if ti == tj:
                row_str += f"{'--':>14}"
            else:
                val = coeff_map.get((ti, tj), None)
                if val is None:
                    row_str += f"{'NA':>14}"
                else:
                    row_str += f"{val:>14.6f}"
        print(row_str)

    print(f"\nSaved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
