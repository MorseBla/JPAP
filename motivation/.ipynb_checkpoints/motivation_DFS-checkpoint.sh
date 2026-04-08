#!/usr/bin/env bash
set -euo pipefail

APP_DIR="../application"     
SCHED_DIR="../Scheduler"     
LOG_ROOT="./DFS_bounds/"        

# Available workloads
workloads=(
  #histogram_converted
  #particle_converted
  #quasirand_converted
  stereodisparity_converted
  mm_converted
  bfs_converted
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
solution=2
setpoint_combos=(
  "0.80 0.80 0.80"
)

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
  for task_name in "$@"; do
    local src="${APP_DIR}/${task_name}.cu"
    [[ -f "$src" ]] || { echo "ERROR: missing workload: $src" >&2; exit 1; }

    echo "Compiling $task_name as t$index..."
    if [[ "$task_name" == "dxtc" ]]; then
      nvcc -o "t$index" "$src" \
        -I ../Common/ -I ../Samples/5_Domain_Specific/dxtc/ \
        -L ../Common/ -lcudart -DT$index -std=c++17
    else
      nvcc -o "t$index" "$src" \
        -I ../Common/ -L ../Common/ -lcudart -DT$index -std=c++17
    fi
    ((index++))
  done
}

compile_server() {
  echo "Compiling controller from: $SCHED_DIR/"
  local src="${SCHED_DIR}/controller.cc"
  local src2="${SCHED_DIR}/scheduler.cc"
  [[ -f "$src" ]] || { echo "ERROR: missing controller: $src" >&2; exit 1; }
  [[ -f "$src2" ]] || { echo "ERROR: missing scheduler: $src2" >&2; exit 1; }
  g++ "$src" -o server -std=c++17 -lrt -pthread -DCLAMP_PERIODS -DBOUND=50
  g++ "$src2" -o scheduler -std=c++17 -lrt -pthread 
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

    local -a args=("$num_tasks" "$sp_dir" "$solution" "$duration")
    for ((i=0; i<num_tasks; i++)); do
      taskname="t$((i+1))"     # creates t1, t2, t3...
      args+=("$taskname" "${rates[i]}" "${setpoints[i]}")
    done

    echo "Executing: ./scheduler ${args[*]}"
    ./scheduler "${args[@]}"
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