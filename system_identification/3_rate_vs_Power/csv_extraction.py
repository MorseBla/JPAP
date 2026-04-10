import os
import json
import csv

BASE_DIR = "logs"
OUTPUT_FILE = "period_vs_power.csv"


def main():
    rows = []

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        for fname in os.listdir(workload_path):
            if not fname.endswith(".json"):
                continue

            full_path = os.path.join(workload_path, fname)

            with open(full_path, "r") as f:
                data = json.load(f)

            period = data.get("period_s", None)
            power = data.get("avg_power_w", None)

            if period is None or power is None:
                continue

            rows.append({
                "workload": workload,
                "period_s": float(period),
                "power_w": float(power),
            })

    rows.sort(key=lambda x: (x["workload"], x["period_s"]))

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["workload", "period_s", "power_w"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
