import os
import json
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib import font_manager
font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Arial"



BASE_DIR = "logs"

FIGSIZE = (6, 3.5)
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
LINEWIDTH = 2.0

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


def apply_styling(ax, ylabel=None, xlabel=None):
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    for tick in ax.get_xticklabels(): tick.set_fontweight("bold")
    for tick in ax.get_yticklabels(): tick.set_fontweight("bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)
    if ylabel: ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight='bold')




def main():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        freqs, rts = load_workload_data(workload_path)

        if len(freqs) == 0:
            continue

        ax.plot(freqs, rts, marker='o', label=workload, linewidth=LINEWIDTH)

    # Tick formatting
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["font.weight"] = "bold"

    # Bold tick labels (correct way)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid
    ax.grid(True, alpha=0.2)
    

    freq_ticks = [306, 408, 510, 612, 714, 816, 918, 1020]
    ax.xaxis.set_major_locator(FixedLocator(freq_ticks))


    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Labels
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("GPU Frequency (MHz)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel("Response Time (ms)", fontsize=LABEL_FONTSIZE, fontweight='bold')

    ax.legend(frameon=False, ncols=2, prop={'weight':'bold', 'size': 14})

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig("freq_vs_rt.pdf", bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    main()
