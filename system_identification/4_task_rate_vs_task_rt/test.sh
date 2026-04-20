#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Diagonal identification test (isolation only)
# For each task i:
#   - run only task i
#   - sweep its own period
#   - measure its own RT
# ============================================================

RUN_DURATION="5"
FREQ="918000000"

SWEEP_PERIODS_MS=(70 80 90 100 110)

TASKS=("mm" "stereo" "quasi" "hist" "particle" "bfs")

# problem sizes
MM_SIZE="1333"
STEREO_W="2304"
STEREO_H="1728"
QUASI_SIZE="16777216"
HIST_SIZE="65536000"
PARTICLE_SIZE="21948"
BFS_SIZE="2000000"

echo
echo "======================================"
echo "Setting GPU frequency to $FREQ"
echo "======================================"
sudo ../scripts/set_gpu_freq.sh "$FREQ"
sleep 1

run_task() {
    local task_name="$1"
    local period_ms="$2"
    local logfile="$3"

    local period_s
    period_s=$(awk "BEGIN { printf \"%.2f\", $period_ms / 1000.0 }")

    case "$task_name" in
        mm)
            ../tasks/mm "$MM_SIZE" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        stereo)
            ../tasks/stereo "$STEREO_W" "$STEREO_H" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        quasi)
            ../tasks/quasi "$QUASI_SIZE" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        hist)
            ../tasks/hist "$HIST_SIZE" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        particle)
            ../tasks/particle "$PARTICLE_SIZE" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        bfs)
            ../tasks/bfs "$BFS_SIZE" "$period_s" "$RUN_DURATION" 1 "$logfile"
            ;;
        *)
            echo "Unknown task: $task_name"
            exit 1
            ;;
    esac
}

for TASK in "${TASKS[@]}"; do
    OUTDIR="logs/diag/${TASK}"
    mkdir -p "$OUTDIR"

    SUMMARY_CSV="${OUTDIR}/summary.csv"
    echo "run_id,task,period_ms,freq_hz,logfile" > "$SUMMARY_CSV"

    echo
    echo "======================================"
    echo "Diagonal test for task: $TASK"
    echo "======================================"

    for period_ms in "${SWEEP_PERIODS_MS[@]}"; do
        run_id="${TASK}_sweep_${period_ms}"
        logfile="${OUTDIR}/${TASK}_${period_ms}ms__${run_id}.json"

        echo
        echo "--------------------------------------"
        echo "Running $TASK at ${period_ms} ms"
        echo "--------------------------------------"

        run_task "$TASK" "$period_ms" "$logfile"

        echo "${run_id},${TASK},${period_ms},${FREQ},${logfile}" >> "$SUMMARY_CSV"
        sleep 1
    done
done

echo
echo "======================================"
echo "Diagonal isolation sweep complete"
echo "======================================"
