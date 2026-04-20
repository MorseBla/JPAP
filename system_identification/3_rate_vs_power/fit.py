#usage:
#python3 fit.py --csv period_vs_power.csv --min-period 0.06 --max-period 0.15
import argparse
import pandas as pd
import numpy as np


def fit_line(x, y):
    """
    Fit y = m*x + b using least squares.
    Returns slope, intercept, r^2.
    """
    if len(x) < 2:
        return None, None, None

    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs

    y_pred = m * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if np.isclose(ss_tot, 0.0):
        r2 = 1.0 if np.isclose(ss_res, 0.0) else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return m, b, r2


def main():
    parser = argparse.ArgumentParser(
        description="Fit linear period vs power curves per workload."
    )
    parser.add_argument(
        "--csv",
        default="period_vs_power.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--min-period",
        type=float,
        default=None,
        help="Minimum period (s) to include"
    )
    parser.add_argument(
        "--max-period",
        type=float,
        default=None,
        help="Maximum period (s) to include"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = {"workload", "period_s", "power_w"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if args.min_period is not None:
        df = df[df["period_s"] >= args.min_period]
    if args.max_period is not None:
        df = df[df["period_s"] <= args.max_period]

    if df.empty:
        print("No data left after applying the period filter.")
        return

    r2_values = []

    print("\nLinear fit results:")
    print("-" * 72)

    for workload in sorted(df["workload"].unique()):
        sub = df[df["workload"] == workload].sort_values("period_s")

        x = sub["period_s"].to_numpy(dtype=float)
        y = sub["power_w"].to_numpy(dtype=float)

        if len(x) < 2:
            print(f"{workload:12s} | skipped (need at least 2 points)")
            continue

        m, b, r2 = fit_line(x, y)

        if m is None:
            print(f"{workload:12s} | skipped")
            continue

        r2_values.append(r2)

        print(
            f"{workload:12s} | "
            f"n={len(x):2d} | "
            f"Power = ({m:.6f})*period + ({b:.6f}) | "
            f"R^2 = {r2:.6f}"
        )

    print("-" * 72)

    if r2_values:
        avg_r2 = float(np.mean(r2_values))
        print(f"Average R^2 across workloads: {avg_r2:.6f}")
    else:
        print("No workloads had enough points to fit.")


if __name__ == "__main__":
    main()
