set -euo pipefail

# ============================================================
# Full off-diagonal identification sweep
# ============================================================

RUN_DURATION="60"
FREQ="918000000"

STATIC_PERIOD_MS=100
SWEEP_PERIODS_MS=(60 65 70 75 80 85 90 95 100)

# static task list
TASKS1=("bfs" "hist" "mm" "particle" "quasi" "stereo")
TASKS1=("stereo" "quasi" "hist" "particle" "bfs")
TASKS1=("particle" "bfs")
TASKS1=("particle")
TASKS1=("hist" "bfs" "mm" "quasi" "stereo" "particle")
#sweep task list
TASKS2=("hist" "mm" "particle" "quasi" "stereo" "bfs")
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

# ------------------------------------------------------------
# Helper to run a task
# ------------------------------------------------------------
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

# ============================================================
# Main double loop: (i = observed, j = excited)
# ============================================================

for STATIC_TASK in "${TASKS1[@]}"; do
  for SWEEP_TASK in "${TASKS2[@]}"; do

    # skip diagonal (handled separately)
    if [[ "$STATIC_TASK" == "$SWEEP_TASK" ]]; then
        continue
    fi

    OUTDIR="logs/offdiag/${STATIC_TASK}_from_${SWEEP_TASK}"
    mkdir -p "$OUTDIR"

    SUMMARY_CSV="${OUTDIR}/summary.csv"
    echo "run_id,static_task,static_period_ms,sweep_task,sweep_period_ms,freq_hz,static_log,sweep_log" > "$SUMMARY_CSV"

    echo
    echo "======================================"
    echo "Testing: $STATIC_TASK (observe) from $SWEEP_TASK (excite)"
    echo "======================================"

    for sweep_period_ms in "${SWEEP_PERIODS_MS[@]}"; do

        run_id="static_${STATIC_PERIOD_MS}_sweep_${sweep_period_ms}"

        static_log="${OUTDIR}/${STATIC_TASK}_${STATIC_PERIOD_MS}ms__${run_id}.json"
        sweep_log="${OUTDIR}/${SWEEP_TASK}_${sweep_period_ms}ms__${run_id}.json"

        echo
        echo "--------------------------------------"
        echo "Run: $run_id"
        echo "Observed: $STATIC_TASK @ ${STATIC_PERIOD_MS} ms"
        echo "Swept:    $SWEEP_TASK @ ${sweep_period_ms} ms"
        echo "--------------------------------------"

        # launch both tasks concurrently
        run_task "$STATIC_TASK" "$STATIC_PERIOD_MS" "$static_log" &
        pid_static=$!

        run_task "$SWEEP_TASK" "$sweep_period_ms" "$sweep_log" &
        pid_sweep=$!

        wait "$pid_static"
        wait "$pid_sweep"
        echo "${run_id},${STATIC_TASK},${STATIC_PERIOD_MS},${SWEEP_TASK},${sweep_period_ms},${FREQ},${static_log},${sweep_log}" >> "$SUMMARY_CSV"

        sleep 5
    done

  done
done

echo
echo "======================================"
echo "Full off-diagonal sweep complete"
echo "======================================"
