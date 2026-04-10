#usage
#python3 fit_freq_rt_linear.py --csv freq_vs_rt.csv --min-freq 306 --max-freq 714
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
        description="Fit linear frequency vs response-time curves per workload."
    )
    parser.add_argument(
        "--csv",
        default="freq_vs_rt.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=None,
        help="Minimum frequency (MHz) to include"
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=None,
        help="Maximum frequency (MHz) to include"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = {"workload", "frequency_mhz", "response_time_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # apply frequency range filter
    if args.min_freq is not None:
        df = df[df["frequency_mhz"] >= args.min_freq]
    if args.max_freq is not None:
        df = df[df["frequency_mhz"] <= args.max_freq]

    if df.empty:
        print("No data left after applying the frequency filter.")
        return

    r2_values = []

    print("\nLinear fit results:")
    print("-" * 72)

    for workload in sorted(df["workload"].unique()):
        sub = df[df["workload"] == workload].sort_values("frequency_mhz")

        x = sub["frequency_mhz"].to_numpy(dtype=float)
        y = sub["response_time_ms"].to_numpy(dtype=float)

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
            f"RT = ({m:.6f})*freq + ({b:.6f}) | "
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
