import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit linear RT vs frequency curves for each workload."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="freq_vs_rt.csv",
        help="Input CSV file with columns: workload, frequency_mhz, response_time_ms"
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=None,
        help="Global minimum frequency (MHz) to include"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Global maximum frequency (MHz) to include"
    )
    parser.add_argument(
        "--ranges-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file specifying per-workload frequency ranges. "
            'Example: {"bfs":[306,714],"mm":[408,918]}'
        )
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots for each workload with fitted line"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as PNG files"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="linear_fit_results",
        help="Directory for saved plots/results"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save fit summary to CSV"
    )
    return parser.parse_args()


def compute_linear_fit(x, y):
    """
    Fit y = m*x + b and compute R^2.
    """
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs[0], coeffs[1]

    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if np.isclose(ss_tot, 0.0):
        r2 = 1.0 if np.isclose(ss_res, 0.0) else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return m, b, r2, y_pred


def load_ranges_json(path):
    with open(path, "r") as f:
        raw = json.load(f)

    parsed = {}
    for workload, bounds in raw.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(
                f"Range for workload '{workload}' must be a 2-element list like [fmin, fmax]"
            )
        parsed[workload] = (float(bounds[0]), float(bounds[1]))
    return parsed


def filter_by_range(df, fmin, fmax):
    mask = np.ones(len(df), dtype=bool)
    if fmin is not None:
        mask &= df["frequency_mhz"] >= fmin
    if fmax is not None:
        mask &= df["frequency_mhz"] <= fmax
    return df.loc[mask].copy()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    if args.save_plots or args.save_summary:
        outdir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"workload", "frequency_mhz", "response_time_ms"}
    if not required_cols.issubset(df.columns):
        print(
            f"ERROR: CSV must contain columns {sorted(required_cols)}. "
            f"Found {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    df["frequency_mhz"] = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    df = df.dropna(subset=["workload", "frequency_mhz", "response_time_ms"]).copy()

    per_workload_ranges = {}
    if args.ranges_json is not None:
        try:
            per_workload_ranges = load_ranges_json(args.ranges_json)
        except Exception as e:
            print(f"ERROR: failed to parse ranges JSON: {e}", file=sys.stderr)
            sys.exit(1)

    workloads = sorted(df["workload"].unique())
    summary_rows = []

    print("\n=== Linear fit: response_time_ms = m * frequency_mhz + b ===\n")

    for workload in workloads:
        wdf = df[df["workload"] == workload].copy()
        wdf = wdf.sort_values("frequency_mhz")

        if workload in per_workload_ranges:
            fmin, fmax = per_workload_ranges[workload]
        else:
            fmin, fmax = args.fmin, args.fmax

        wdf_fit = filter_by_range(wdf, fmin, fmax)

        if len(wdf_fit) < 2:
            print(
                f"[{workload}] skipped: need at least 2 points after filtering "
                f"(got {len(wdf_fit)})"
            )
            continue

        x = wdf_fit["frequency_mhz"].to_numpy(dtype=float)
        y = wdf_fit["response_time_ms"].to_numpy(dtype=float)

        m, b, r2, y_pred = compute_linear_fit(x, y)

        fit_min = float(np.min(x))
        fit_max = float(np.max(x))
        n = len(x)

        print(f"[{workload}]")
        print(f"  range used      : {fit_min:.1f} to {fit_max:.1f} MHz")
        print(f"  points used     : {n}")
        print(f"  slope           : {m:.8f} ms/MHz")
        print(f"  intercept       : {b:.8f} ms")
        print(f"  R^2             : {r2:.6f}")
        print(f"  model           : RT = {m:.8f} * freq + {b:.8f}")
        print()

        summary_rows.append({
            "workload": workload,
            "fmin_used_mhz": fit_min,
            "fmax_used_mhz": fit_max,
            "points_used": n,
            "slope_ms_per_mhz": m,
            "intercept_ms": b,
            "r2": r2,
        })

        if args.plot or args.save_plots:
            plt.figure(figsize=(7, 5))

            # all original data
            plt.scatter(
                wdf["frequency_mhz"],
                wdf["response_time_ms"],
                label="All data"
            )

            # filtered data used in fit
            plt.scatter(
                wdf_fit["frequency_mhz"],
                wdf_fit["response_time_ms"],
                label="Data used for fit"
            )

            # fitted line across fit range
            x_line = np.linspace(fit_min, fit_max, 200)
            y_line = m * x_line + b
            plt.plot(
                x_line,
                y_line,
                label=f"Linear fit (R²={r2:.4f})"
            )

            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Response Time (ms)")
            plt.title(f"{workload}: Frequency vs Response Time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if args.save_plots:
                plot_path = outdir / f"{workload}_linear_fit.png"
                plt.savefig(plot_path, dpi=200)

            if args.plot:
                plt.show()
            else:
                plt.close()

    if args.save_summary and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = outdir / "linear_fit_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
