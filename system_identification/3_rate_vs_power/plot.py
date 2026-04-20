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
        period = period * 1000
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
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial']
    })

    for workload in os.listdir(BASE_DIR):
        workload_path = os.path.join(BASE_DIR, workload)

        if not os.path.isdir(workload_path):
            continue

        periods, powers = load_workload_data(workload_path)
        if len(periods) == 0:
            continue

        ax.plot(periods, powers, marker='o', label=workload, linewidth=LINEWIDTH)

    # Labels (bold, consistent)
    ax.set_xlabel("Task Period (ms)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel("Power (W)", fontsize=LABEL_FONTSIZE, fontweight='bold')

    # Tick styling
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # Increase y-axis tick density (but not hardcoded values)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid (subtle)
    ax.grid(True, alpha=0.2)

    ax.legend(frameon=False, ncols=2, prop={'weight':'bold', 'size': 14})

    fig.tight_layout()
    fig.savefig("period_vs_power.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
