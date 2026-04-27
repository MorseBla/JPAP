import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager

# ---------------- CONFIG ----------------
solution = "LQR"
if (solution == "FC"):
    csv_path = Path("logs/FC/controller.csv")
    save_path = Path("figures/FC_task_interrupt.pdf")
elif (solution == "LQR"):
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

FIGSIZE = (16, 3.2)
DPI = 300

# Optional Arial
font_manager.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
plt.rcParams["font.family"] = "Arial"

# ---------------- LOAD ----------------
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Sort in file order first
df = df.reset_index(drop=True)

# ---------------- FIX RESET CONTROL PERIOD ----------------
# The controller period resets to 0 when the 4th task appears.
# Make a continuous control_period_fixed.
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
# Power is repeated once per task per control period, so average it per period.
power_df = (
    df.groupby("control_period_fixed", as_index=False)["measured_power"]
    .mean()
)

# ---------------- PLOT ----------------
fig, axs = plt.subplots(1, 4, figsize=FIGSIZE, sharex=True)

metrics = [
    ("rtr", "RTR"),
    ("deadline_miss_pct", "Deadline Miss Ratio (%)"),
    ("period_ms", "Task Period (ms)"),
]

for ax, (col, ylabel) in zip(axs[:3], metrics):
    for task_id in sorted(df["task"].unique()):
        task_df = df[df["task"] == task_id].copy()

        # Task 3 should not have any line before it appears.
        if task_id == 3:
            task_df = task_df[task_df["control_period_fixed"] >= task4_start]

        ax.plot(
            task_df["control_period_fixed"],
            task_df[col],
            linewidth=1.8,
            label=TASK_NAMES.get(task_id, f"Task {task_id}")
        )

    ax.axvline(task4_start, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Control Period")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

# Power subplot
axs[3].plot(
    power_df["control_period_fixed"],
    power_df["measured_power"],
    linewidth=1.8,
    label="Power"
)
axs[3].axvline(task4_start, linestyle="--", linewidth=1.5)
axs[3].set_xlabel("Control Period")
axs[3].set_ylabel("Power Usage (mW)")
axs[3].grid(True, alpha=0.3)

# One legend above all plots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08)
)

fig.tight_layout()
fig.savefig(save_path, dpi=DPI, bbox_inches="tight")

print(f"Saved plot to: {save_path}")
