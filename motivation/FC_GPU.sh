#!/usr/bin/env bash
set -euo pipefail

APP_DIR="../application"     
SCHED_DIR="../Scheduler"     
LOG_ROOT="logs/FC_GPU"        

# Available workloads
workloads=(
  #particle
  #quasi
  hist
  stereo
  mm
  #bfs
)

# Generate all unique 3-workload combinations
combinations=()
for ((i=0; i<${#workloads[@]}; i++)); do
  for ((j=i+1; j<${#workloads[@]}; j++)); do
    for ((k=j+1; k<${#workloads[@]}; k++)); do
      combinations+=("${workloads[i]} ${workloads[j]} ${workloads[k]}")
    done
  done
done

#default_rates=(0.060 0.100 0.060)
default_rates=(0.160 0.080 0.0600)
duration=240

FC_GPU=1
N_PLUS_ONE=2
LQR=3
solution=$FC_GPU

setpoint_combos=(
  "0.80 0.80 0.80"
)
ps_set=5000

# -----------------------------
# Helpers
# -----------------------------
cleanup_shared_memory() {
  echo "Cleaning shared memory segments and related PIDs..."

  # Get shm segments created by this user
  ipcs -m | awk -v user="$USER" '
    NR>3 && $3==user {print $2}
  ' | while read -r shmid; do
      [[ -n "${shmid:-}" ]] || continue
      echo "Removing shm segment $shmid"
      ipcrm -m "$shmid" 2>/dev/null || true
  done

  echo "Killing processes attached to those shm segments..."

  ipcs -m -p | awk -v user="$USER" '
    NR>3 {print $2, $3, $4}
  ' | while read -r shmid cpid lpid; do
      # Kill only if PID belongs to current user
      for pid in "$cpid" "$lpid"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && ps -o user= -p "$pid" 2>/dev/null | grep -q "$USER"; then
          echo "Killing PID $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
  done
}

cleanup_tasks() {
  echo "Killing running tasks/server (if any) and cleaning shared memory..."
  pkill -9 -f './t[0-9]$' 2>/dev/null || true
  pkill -9 -f './server$' 2>/dev/null || true
  pkill -9 -f './scheduler$' 2>/dev/null || true
  cleanup_shared_memory

  echo "Clearing logs/figures in current experiment folder..."
  rm -rf "${LOG_ROOT:?}/"* ./figures/* 2>/dev/null || true
  mkdir -p "$LOG_ROOT" ./figures
}

compile_tasks() {
  local index=1
  echo "Compiling workloads from: $APP_DIR/"
  make -C $APP_DIR EXTRA_FLAGS="-DJetson"
}

compile_server() {
  echo "Compiling controller from: $SCHED_DIR/"
  make -C $SCHED_DIR EXTRA_FLAGS="-DJetson" #-DCLAMP_PERIODS -DBOUND=50
}

run_combination() {
  local -a workloads=("$@")
  local num_tasks="${#workloads[@]}"

  # Build rates array sized to num_tasks (fallback to 0.300 if not provided)
  local -a rates=()
  for ((i=0; i<num_tasks; i++)); do
    rates+=("${default_rates[i]:-0.300}")
  done

  # Optional: adjust rate if dxtc present
  if [[ " ${workloads[*]} " == *" dxtc "* ]]; then
    rates[0]=0.600
  fi

  local comb_name
  comb_name=$(printf "%s" "${workloads[*]}" | tr ' ' '_')
  local comb_log_dir="${LOG_ROOT}/${comb_name}"
  mkdir -p "$comb_log_dir"

  echo "Running: ${workloads[*]}"
  echo "Rates:   ${rates[*]}"
  echo "Logs at: $comb_log_dir"

  for sp_line in "${setpoint_combos[@]}"; do
   
    read -r -a setpoints <<< "$sp_line"
    
    if [[ "${#setpoints[@]}" -ne "$num_tasks" ]]; then
      echo "ERROR: setpoints '$sp_line' has ${#setpoints[@]} values but num_tasks=$num_tasks" >&2
      exit 1
    fi
    
    local sp_dir="${comb_log_dir}/setpoint_$(echo "$sp_line" | tr ' ' '_')"
    mkdir -p "$sp_dir"
    
    local scheduler_bin="../Scheduler/bin/scheduler"
    local app_bin_dir="../application/bin"
    
    local -a args=("$num_tasks" "$sp_dir" "$solution" "$duration" "$ps_set")
    
    for ((i=0; i<num_tasks; i++)); do
      #echo $workloads
      taskname="${workloads[i]}"   # should contain bfs, hist, mm, particle, quasi, stereo
      args+=("${app_bin_dir}/${taskname}" "${rates[i]}" "${setpoints[i]}")
    done
    
    echo "Executing: ${scheduler_bin} ${args[*]}"
    "${scheduler_bin}" "${args[@]}"
 
  done

  sleep 5
}

# -----------------------------
# Main
# -----------------------------
cleanup_tasks
compile_server

echo "Generated ${#combinations[@]} 3-workload combinations:"
for comb in "${combinations[@]}"; do
  echo "  $comb"
done

for comb in "${combinations[@]}"; do
  read -r -a task_array <<< "$comb"
  compile_tasks "${task_array[@]}"
  run_combination "${task_array[@]}"
done

echo "All task combinations completed."
