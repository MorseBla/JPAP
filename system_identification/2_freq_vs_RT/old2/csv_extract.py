import os
import json
import re
import csv

BASE_DIR = "logs"
OUTPUT_FILE = "freq_vs_rt.csv"

def extract_freq_from_filename(filename):
    # Example: bfs_306000000.json → 306 MHz
    match = re.search(r'_(\d+)\.json$', filename)
    if match:
        return int(match.group(1)) / 1e6  # Hz → MHz
    return None

def main():
    rows = []

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        for fname in os.listdir(workload_path):
            if not fname.endswith(".json"):
                continue

            freq = extract_freq_from_filename(fname)
            if freq is None:
                continue

            full_path = os.path.join(workload_path, fname)

            with open(full_path, 'r') as f:
                data = json.load(f)

            rt = data.get("avg_response_ms", None)
            if rt is None:
                continue

            rows.append({
                "workload": workload,
                "frequency_mhz": freq,
                "response_time_ms": rt
            })

    # sort for cleanliness (by workload, then freq)
    rows.sort(key=lambda x: (x["workload"], x["frequency_mhz"]))

    # write CSV
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "workload", "frequency_mhz", "response_time_ms"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
