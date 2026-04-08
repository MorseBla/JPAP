#!/usr/bin/env bash
set -euo pipefail

APP_DIR="../application"
SCHED_DIR="../Scheduler"
LOG_ROOT="./Proposed_bounds/"
HIPIFY_DIR="./hipified_build"

workloads=(
  histogram_converted
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

duration=400
solution=3

setpoint_values=(
  0.90
  0.85
  0.80
  0.75
  0.70
)

default_rates=(0.050 0.080 0.080)

cleanup_shared_memory() {
  echo "Cleaning shared memory segments and related PIDs..."

  ipcs -m | awk -v user="$USER" '
    NR>3 && $3==user {print $2}
  ' | while read -r shmid; do
      [[ -n "${shmid:-}" ]] || continue
      echo "Removing shm segment $shmid"
      ipcrm -m "$shmid" 2>/dev/null || true
  done

  ipcs -m -p | awk '
    NR>3 {print $2, $3, $4}
  ' | while read -r shmid cpid lpid; do
      for pid in "$cpid" "$lpid"; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && ps -o user= -p "$pid" 2>/dev/null | grep -q "^$USER$"; then
          echo "Killing PID $pid"
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
  done
}

cleanup_tasks() {
  echo "Killing running tasks/server/scheduler..."
  pkill -9 -f './t[0-9]+$' 2>/dev/null || true
  pkill -9 -f './server$' 2>/dev/null || true
  pkill -9 -f './scheduler$' 2>/dev/null || true
  cleanup_shared_memory

  #rm -rf "$HIPIFY_DIR"
  mkdir -p "$LOG_ROOT" ./figures "$HIPIFY_DIR"
}

cleanup_tasks_run() {
  echo "Killing running tasks/server/scheduler..."
  sudo pkill -9 -f '(^|/)t[0-9]+$' || true
  sudo pkill -9 -f '(^|/)server$' || true
  sudo pkill -9 -f '(^|/)scheduler$' || true
  sleep 1
  cleanup_shared_memory
}

check_tools() {
  command -v hipify-perl >/dev/null 2>&1 || {
    echo "ERROR: hipify-perl not found in PATH" >&2
    exit 1
  }

  command -v hipcc >/dev/null 2>&1 || {
    echo "ERROR: hipcc not found in PATH" >&2
    exit 1
  }

  command -v g++ >/dev/null 2>&1 || {
    echo "ERROR: g++ not found in PATH" >&2
    exit 1
  }
}

hipify_one() {
  local src="$1"
  local dst="$2"

  echo "Hipifying $src -> $dst"
  hipify-perl "$src" > "$dst"
}

compile_tasks() {
  local index=1
  echo "Hipifying and compiling workloads from: $APP_DIR/"

  for task_name in "$@"; do
    local src="${APP_DIR}/${task_name}.cu"
    local hip_src="${HIPIFY_DIR}/${task_name}.cpp"

    [[ -f "$src" ]] || { echo "ERROR: missing workload: $src" >&2; exit 1; }

    #hipify_one "$src" "$hip_src"

    echo "Compiling $task_name as t$index with hipcc..."
    hipcc -o "t$index" "$hip_src" \
      -I ../Common/ \
      -std=c++17 -lrt -pthread
    ((index++))
  done
}

compile_server() {
  echo "Compiling controller from: $SCHED_DIR/"
  local src1="${SCHED_DIR}/amdcontroller.cc"
  local src2="${SCHED_DIR}/scheduler.cc"

  [[ -f "$src1" ]] || { echo "ERROR: missing controller: $src1" >&2; exit 1; }
  [[ -f "$src2" ]] || { echo "ERROR: missing scheduler: $src2" >&2; exit 1; }

  g++ "$src1" -o server \
    -std=c++17 -pthread -lrt \
    -I /opt/rocm-5.6.0/rocm_smi/include/ \
    -L /opt/rocm-5.6.0/rocm_smi/lib/ \
    -lrocm_smi64 \
    -Wl,-rpath,/opt/rocm-5.6.0/rocm_smi/lib/ \
    -DCLAMP_PERIODS -DBOUND=50

  g++ "$src2" -o scheduler -std=c++17 -pthread -lrt
}

run_combination() {
  local -a workloads=("$@")
  local num_tasks="${#workloads[@]}"

  local -a rates=()
  for ((i=0; i<num_tasks; i++)); do
    rates+=("${default_rates[i]:-0.300}")
  done

  local comb_name
  comb_name=$(printf "%s" "${workloads[*]}" | tr ' ' '_')
  local comb_log_dir="${LOG_ROOT}/${comb_name}"
  mkdir -p "$comb_log_dir"

  echo "Running: ${workloads[*]}"
  echo "Rates:   ${rates[*]}"
  echo "Logs at: $comb_log_dir"

  for sp in "${setpoint_values[@]}"; do
    local sp_dir="${comb_log_dir}/setpoint_${sp}"
    mkdir -p "$sp_dir"

    local -a args=("$num_tasks" "$sp_dir" "$solution" "$duration")
    for ((i=0; i<num_tasks; i++)); do
      args+=("t$((i+1))" "${rates[i]}" "$sp")
    done

    echo "Executing: ./scheduler ${args[*]}"
    ./scheduler "${args[@]}"
  done

  sleep 5
}

check_tools
cleanup_tasks
compile_server

echo "Generated ${#combinations[@]} 3-workload combinations:"
for comb in "${combinations[@]}"; do
  echo "  $comb"
done

for comb in "${combinations[@]}"; do
  cleanup_tasks_run
  read -r -a task_array <<< "$comb"
  compile_tasks "${task_array[@]}"
  run_combination "${task_array[@]}"
  sleep 2
done

echo "All task combinations completed."