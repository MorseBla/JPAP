import os
import json
import re
import matplotlib.pyplot as plt

BASE_DIR = "logs"

def extract_freq_from_filename(filename):
    # matches something like bfs_306000000.json
    match = re.search(r'_(\d+)\.json$', filename)
    if match:
        return int(match.group(1)) / 1e6  # convert to MHz
    return None

def load_workload_data(workload_path):
    freqs = []
    rts = []

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

        freqs.append(freq)
        rts.append(rt)

    # sort by frequency
    pairs = sorted(zip(freqs, rts))
    if pairs:
        freqs, rts = zip(*pairs)
    else:
        freqs, rts = [], []

    return freqs, rts

def main():
    plt.figure(figsize=(8, 5))

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        freqs, rts = load_workload_data(workload_path)

        if len(freqs) == 0:
            continue

        plt.plot(freqs, rts, marker='o', label=workload)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Avg Response Time (ms)")
    plt.title("Frequency vs Response Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("freq_vs_rt.png")
    plt.show()

if __name__ == "__main__":
    main()
