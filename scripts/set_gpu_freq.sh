#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <frequency_in_hz>" >&2
  exit 1
fi

GPU_PATH="/sys/class/devfreq/17000000.gpu"
FREQ="$1"

if [[ ! -d "$GPU_PATH" ]]; then
  echo "Error: GPU devfreq path not found: $GPU_PATH" >&2
  exit 1
fi

# Optional: ensure it's an available OPP (some platforms expose this file)
if [[ -f "$GPU_PATH/available_frequencies" ]]; then
  if ! grep -qw "$FREQ" "$GPU_PATH/available_frequencies"; then
    echo "Error: $FREQ not in available_frequencies." >&2
    echo "Available:" >&2
    cat "$GPU_PATH/available_frequencies" >&2
    exit 1
  fi
fi

echo "userspace" > "$GPU_PATH/governor"
echo "$FREQ" > "$GPU_PATH/min_freq"
echo "$FREQ" > "$GPU_PATH/max_freq"
echo "$FREQ" > "$GPU_PATH/userspace/set_freq"

# Read back what we set
gov="$(cat "$GPU_PATH/governor")"
minf="$(cat "$GPU_PATH/min_freq")"
maxf="$(cat "$GPU_PATH/max_freq")"
setf="$(cat "$GPU_PATH/userspace/set_freq")"
curf="$(cat "$GPU_PATH/cur_freq" 2>/dev/null || echo "n/a")"

#echo "Requested: $FREQ Hz"
#echo "Readback: governor=$gov min=$minf max=$maxf set=$setf cur=$curf"

# Optional: wait up to ~2s for cur_freq to match (devfreq can be lazy)
if [[ -f "$GPU_PATH/cur_freq" ]]; then
  for _ in {1..100}; do
    cur="$(cat "$GPU_PATH/cur_freq")"
    if [[ "$cur" == "$FREQ" ]]; then
      #echo "Applied: cur_freq=$cur Hz"
      exit 0
    fi
    sleep 0.02
  done
  echo "Warning: cur_freq did not converge to $FREQ Hz (last cur_freq=$cur Hz)" >&2
fi

