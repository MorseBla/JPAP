import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
solution = "LQR"   # "FC", "LQR", or "DFS"

RTR_SETPOINT = 0.90
POWER_SETPOINT = 5000

if solution == "FC":
    csv_path = Path("logs/FC/controller.csv")
    save_path = Path("figures/FC_task_interrupt.pdf")
elif solution == "LQR":
    csv_path = Path("logs/LQR/controller.csv")
    save_path = Path("figures/LQR_task_interrupt.pdf")
else:
    csv_path = Path("logs/DFS/controller.csv")
    save_path = Path("figures/DFS_task_interrupt.pdf")

TASK_NAMES = {
    0: "Task 0",
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
}
TASK_NAMES = {
    0: "hist",
    1: "stereo",
    2: "mm",
    3: "particle",
}

TASK_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:red",
    3: "tab:purple",
}

FIGSIZE = (8, 6)
DPI = 300

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LINEWIDTH = 2.0

# ---------------- FONT ----------------
font_manager.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
font_manager.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf")

plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "normal",
    #"font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# ---------------- LOAD ----------------
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df.reset_index(drop=True)

# ---------------- FIX RESET CONTROL PERIOD ----------------
fixed_periods = []
offset = 0
prev_cp = None

for cp in df["control_period"].astype(int):
    if prev_cp is not None and cp < prev_cp:
        offset += prev_cp + 1
    fixed_periods.append(cp + offset)
    prev_cp = cp

df["control_period_fixed"] = fixed_periods

# Find when task 3 first appears
task4_start = df.loc[df["task"] == 3, "control_period_fixed"].min()

# ---------------- POWER DATA ----------------
power_df = (
    df.groupby("control_period_fixed", as_index=False)["measured_power"]
    .mean()
)

# ---------------- PLOT ----------------
fig, axs = plt.subplots(2, 2, figsize=FIGSIZE)
axs = axs.flatten()

metrics = [
    ("rtr", "RTR", "(a)"),
    ("deadline_miss_pct", "DMR (%)", "(b)"),
    ("period_ms", "Period (ms)", "(c)"),
]

# (1) RTR
ax = axs[0]
ax.axhline(
    y=RTR_SETPOINT,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Setpoint",
)

for task_id in sorted(df["task"].unique()):
    task_df = df[df["task"] == task_id].copy()

    if task_id == 3:
        task_df = task_df[task_df["control_period_fixed"] >= task4_start]

    ax.plot(
        task_df["control_period_fixed"],
        task_df["rtr"],
        linewidth=LINEWIDTH,
        color=TASK_COLORS.get(task_id, None),
        label=TASK_NAMES.get(task_id, f"Task {task_id}"),
    )

ax.axvline(task4_start, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel("RTR", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_xlabel("Control Period \n(a)", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_ylim(0.5, 1.2)

# (2) Deadline Miss Ratio
ax = axs[1]
for task_id in sorted(df["task"].unique()):
    task_df = df[df["task"] == task_id].copy()

    if task_id == 3:
        task_df = task_df[task_df["control_period_fixed"] >= task4_start]

    ax.plot(
        task_df["control_period_fixed"],
        task_df["deadline_miss_pct"],
        linewidth=LINEWIDTH,
        color=TASK_COLORS.get(task_id, None),
    )

ax.axvline(task4_start, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel("DMR (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_xlabel("Control Period \n(b)", fontsize=LABEL_FONTSIZE, fontweight="bold")

# (3) Task Period
ax = axs[2]
for task_id in sorted(df["task"].unique()):
    task_df = df[df["task"] == task_id].copy()

    if task_id == 3:
        task_df = task_df[task_df["control_period_fixed"] >= task4_start]

    ax.plot(
        task_df["control_period_fixed"],
        task_df["period_ms"],
        linewidth=LINEWIDTH,
        color=TASK_COLORS.get(task_id, None),
    )

ax.axvline(task4_start, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel("Period (ms)", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_xlabel("Control Period \n(c)", fontsize=LABEL_FONTSIZE, fontweight="bold")

# (4) Power
ax = axs[3]
ax.plot(
    power_df["control_period_fixed"],
    power_df["measured_power"],
    linewidth=LINEWIDTH,
    color="tab:green",
    label="Power",
)

ax.axhline(
    y=POWER_SETPOINT,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Setpoint",
)

ax.axvline(task4_start, color="gray", linestyle=":", linewidth=1.5)
ax.set_ylabel("Power (mW)", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_xlabel("Control Period \n(d)", fontsize=LABEL_FONTSIZE, fontweight="bold")
ax.set_ylim([2000, 10000])

# ---------------- GLOBAL STYLING ----------------
xmax = int(df["control_period_fixed"].max())

for ax in axs:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, xmax])

# ---------------- LEGEND ----------------
handles, labels = axs[0].get_legend_handles_labels()

power_handle = Line2D([0], [0], color="tab:green", linewidth=LINEWIDTH)
handles.append(power_handle)
labels.append("Power")

task_add_handle = Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5)
handles.append(task_add_handle)
labels.append("Task Added")

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=len(labels),
    frameon=False,
    prop={"size": 12,}, 
    #prop={"size": 12, "weight": "bold"},
)

plt.tight_layout(rect=[0, 0, 1, 0.85])

save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=DPI, bbox_inches="tight")

print(f"Saved plot to: {save_path}")
