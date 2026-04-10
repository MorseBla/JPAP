import os
import json
import re
import matplotlib.pyplot as plt

BASE_DIR = "logs"

def extract_period_from_filename(filename):
    """
    Extract period from filenames like:
    bfs_period_0p60.json → 0.60
    """
    match = re.search(r'_period_(\d+)p(\d+)\.json$', filename)
    if match:
        whole = match.group(1)
        frac = match.group(2)
        return float(f"{whole}.{frac}")
    return None

def load_workload_data(workload_path):
    periods = []
    powers = []

    for fname in os.listdir(workload_path):
        if not fname.endswith(".json"):
            continue

        period = extract_period_from_filename(fname)
        if period is None:
            continue

        full_path = os.path.join(workload_path, fname)

        with open(full_path, 'r') as f:
            data = json.load(f)

        power = data.get("avg_power_w", None)
        if power is None:
            continue

        periods.append(period)
        powers.append(power)

    # sort by period
    pairs = sorted(zip(periods, powers))
    if pairs:
        periods, powers = zip(*pairs)
    else:
        periods, powers = [], []

    return periods, powers

def main():
    plt.figure(figsize=(8, 5))

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        periods, powers = load_workload_data(workload_path)

        if len(periods) == 0:
            continue

        plt.plot(periods, powers, marker='o', label=workload)

    plt.xlabel("Period (s)")
    plt.ylabel("Avg Power (W)")
    plt.title("Period vs Power")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("period_vs_power.png")
    plt.show()

if __name__ == "__main__":
    main()
