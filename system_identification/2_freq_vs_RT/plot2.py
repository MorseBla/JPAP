import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
# ---------- FONT ----------
font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
arial_regular = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
font_manager.fontManager.addfont(font_path)
font_manager.fontManager.addfont(arial_regular)

# ---------- CONFIG ----------
root = Path("logs")
save_path = Path("figures")

# Set order here if you want a specific legend/plot order
task_names = ["hist", "mm", "particle", "quasi", "stereo"]

FIGSIZE = (6, 5)
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 16
LINEWIDTH = 2.0
MARKERSIZE = 7
LEGEND_FONTSIZE = 14

# ---------- HELPERS ----------
def is_valid_number(x):
    try:
        v = float(x)
        return v != -1
    except:
        return False

def read_results_csv(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows

def aggregate_task_data(task_dir: Path):
    """
    Returns:
        {
            'rt':   {freq_mhz: avg_rt},
            'power':{freq_mhz: avg_power},
            'temp': {freq_mhz: avg_temp},
        }
    """
    agg = {
        "rt": defaultdict(list),
        "power": defaultdict(list),
        "temp": defaultdict(list),
    }

    for freq_dir in sorted(task_dir.iterdir()):
        if not freq_dir.is_dir():
            continue

        csv_path = freq_dir / "results.csv"
        if not csv_path.exists():
            continue

        rows = read_results_csv(csv_path)
        for row in rows:
            if not is_valid_number(row.get("freq_mhz", "")):
                continue

            freq = float(row["freq_mhz"])

            if is_valid_number(row.get("avg_response_ms", "")):
                agg["rt"][freq].append(float(row["avg_response_ms"]))

            if is_valid_number(row.get("power_w", "")):
                agg["power"][freq].append(float(row["power_w"]) * 1000)

            if is_valid_number(row.get("temp_c", "")):
                agg["temp"][freq].append(float(row["temp_c"]))

    out = {"rt": {}, "power": {}, "temp": {}}
    for metric in ["rt", "power", "temp"]:
        for freq, vals in agg[metric].items():
            if len(vals) > 0:
                out[metric][freq] = float(np.mean(vals))

    return out

def get_xy(metric_dict):
    freqs = sorted(metric_dict.keys())
    vals = [metric_dict[f] for f in freqs]
    return np.asarray(freqs, dtype=float), np.asarray(vals, dtype=float)

# ---------- LOAD ALL TASK DATA ----------
all_data = {}

for task in task_names:
    task_dir = root / task
    if not task_dir.exists() or not task_dir.is_dir():
        print(f"Skipping missing task directory: {task_dir}")
        continue

    all_data[task] = aggregate_task_data(task_dir)

# ---------- MAKE 3 SEPARATE PLOTS ----------
plot_info = [
    ("rt", "Response Time (ms)", "Frequency (MHz)", "freq_vs_rt.pdf"),
    ("power", "Power (mW)", "Frequency (MHz)", "freq_vs_power.pdf"),
    ("temp", "Temperature (°C)", "Frequency (MHz)", "freq_vs_temp.pdf"),
]

save_path.mkdir(exist_ok=True)
for metric, ylabel, xlabel, filename in plot_info:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for task in task_names:
        if task not in all_data:
            continue

        metric_dict = all_data[task][metric]
        if not metric_dict:
            continue

        x, y = get_xy(metric_dict)

        ax.plot(
            x,
            y,
            marker='o',
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            label=task,
        )

    # X ticks
    all_freqs = sorted({
        freq
        for task in task_names
        if task in all_data
        for freq in all_data[task][metric].keys()
    })
    ax.set_xticks(all_freqs)
    ax.set_xticklabels([f"{int(val)}" for val in all_freqs], rotation=45)

    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight='bold')

    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # --- LEGEND ABOVE FIGURE ---
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.15),   # push above axes
        ncol=3, #len(task_names),         # horizontal layout
        frameon=False,
        prop={'size': LEGEND_FONTSIZE}
    )

    if (metric == "power"):
        ax.set_ylim(5000, 9000) 
    if (metric == "rt"):
        ax.set_ylim(15, 155) 
    if (metric == "temp"):
        ax.set_ylim(48.5, 54.5) 
    ymin, ymax = ax.get_ylim()
    ticks = np.linspace(ymin, ymax, 5) 
    ax.set_yticks(ticks)

    # leave space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    out_file = save_path / filename
    plt.savefig(out_file, bbox_inches='tight')
    plt.show()

    print(f"Saved plot to: {out_file}")
