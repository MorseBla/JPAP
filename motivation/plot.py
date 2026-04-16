

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
 
# --------- CONFIG ----------
root = Path("logs/Full_LQR/hist_stereo_mm/")
root = Path("logs/LQR/hist_stereo_mm_test/") 
setpoints = [0.90, 0.80, 0.70]
setpoints = [0.90]
p_setpoint = [5000]
warmup_skip = 0         
max_periods = 100        
save_path = Path("figures")
task_names = ["stereo", "mm", "bfs"]
task_colors = ['tab:blue', 'tab:orange', 'tab:red']
 
# Styling constants
FIGSIZE = (15, 4) # Slightly taller/wider to prevent overlap
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 16
LINEWIDTH = 2.0
# ---------------------------
 
def read_data(filepath: Path):
    if not filepath.exists(): return None
    vals = []
    with filepath.open("r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try: vals.append(float(s))
            except ValueError: continue
    return vals if vals else None
 
def slice_idx(data, start, stop):
    if data is None: return None
    n = len(data)
    a = min(max(start, 0), n)
    b = min(max(stop, 0), n)
    return np.asarray(data[a:b], dtype=float)
 
 
for sp in setpoints:
    run_dir = root / f"setpoint_{sp:.2f}_{sp:.2f}_{sp:.2f}"
    print(run_dir)
    data = {name: {} for name in task_names}
    for i, name in enumerate(task_names, 1):
        t_id = f"t{i}"
        data[name]['rtr'] = slice_idx(read_data(run_dir / f"Proposed_{t_id}_rtr.txt"), warmup_skip, max_periods)
        data[name]['dl']  = slice_idx(read_data(run_dir / f"Proposed_{t_id}_deadline.txt"), warmup_skip, max_periods)
        data[name]['per'] = slice_idx(read_data(run_dir / f"Proposed_{t_id}_period.txt"), warmup_skip, max_periods)
        data[name]['low'] = slice_idx(read_data(run_dir / f"Proposed_{t_id}_lowerperiodbound.txt"), warmup_skip, max_periods)
        data[name]['high']= slice_idx(read_data(run_dir / f"Proposed_{t_id}_higherperiodbound.txt"), warmup_skip, max_periods)
 
    pwrs = slice_idx(read_data(run_dir / "Proposed_finalpowervalues.txt"), warmup_skip, max_periods)
 
    fig, axs = plt.subplots(1, 4, figsize=FIGSIZE)
 
    # (1) RTR
    ax = axs[0]
    ax.axhline(y=sp, color='black', linestyle='--', linewidth=1.5, label="Setpoint")
    for name, color in zip(task_names, task_colors):
        if data[name]['rtr'] is not None:
            ax.plot(data[name]['rtr'], linewidth=LINEWIDTH, label=name, color=color)
    ax.set_ylabel("RTR", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel("Control Period \n(a)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylim(0.5, 1.1)
 
    # (2) Deadline
    ax = axs[1]
    for name, color in zip(task_names, task_colors):
        if data[name]['dl'] is not None:
            ax.plot(data[name]['dl'], linewidth=LINEWIDTH, color=color)
    ax.set_ylabel("DMR (%)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel("Control Period \n(b)", fontsize=LABEL_FONTSIZE, fontweight='bold')
 
    ax = axs[2]
    for name, color in zip(task_names, task_colors):
        if data[name]['per'] is not None:
            ax.plot(data[name]['per'], linewidth=LINEWIDTH, color=color)
        if data[name]['low'] is not None:
            ax.plot(data[name]['low'], ':', alpha=0.6, linewidth=1.2, color=color)
        if data[name]['high'] is not None:
            ax.plot(data[name]['high'], ':', alpha=0.6, linewidth=1.2, color=color)
    ax.set_ylabel("Period (ms)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel("Control Period \n(c)", fontsize=LABEL_FONTSIZE, fontweight='bold')
 
    # (4) Power
    ax = axs[3]
    if pwrs is not None:
        ax.plot(pwrs, linewidth=LINEWIDTH, color='tab:green', label='Power')
    print(np.mean(pwrs))
    ax.set_ylabel("Power (W)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel("Control Period \n(d)", fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.axhline(y=p_setpoint, color='black', linestyle='--', linewidth=1.5, label="Setpoint")
    ax.set_ylim([3000,6000])
 
    # Global Styling
    for ax in axs:
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        for tick in ax.get_xticklabels(): tick.set_fontweight("bold")
        for tick in ax.get_yticklabels(): tick.set_fontweight("bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)
        ax.set_xlim([0,100])
 
 
    handles, labels = axs[0].get_legend_handles_labels() # Tasks + Setpoint
    power_handle = Line2D([0], [0], color='tab:green', linewidth=LINEWIDTH)
    handles.append(power_handle)
    labels.append("Power")
    bound_handle = Line2D([0], [0], color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
    handles.append(bound_handle)
    labels.append("Bounds")
 
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.92),  
        ncol=len(labels), 
        frameon=False, 
        prop={'weight': 'bold', 'size': 14} 
    )   
    plt.tight_layout(rect=[0, 0, 1, 0.85]) 
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path /'proposed.pdf', bbox_inches='tight')
