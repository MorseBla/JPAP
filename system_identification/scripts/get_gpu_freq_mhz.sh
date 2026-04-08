#!/usr/bin/env bash
set -euo pipefail

GPU_PATH="/sys/class/devfreq/17000000.gpu"

if [[ ! -d "$GPU_PATH" ]]; then
  echo "Error: GPU devfreq path not found: $GPU_PATH" >&2
  exit 1
fi

if [[ ! -f "$GPU_PATH/cur_freq" ]]; then
  echo "Error: cur_freq not found at: $GPU_PATH/cur_freq" >&2
  exit 1
fi

FREQ_HZ="$(cat "$GPU_PATH/cur_freq")"

# convert to MHz 
echo $((FREQ_HZ / 1000000))

