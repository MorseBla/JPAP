#!/usr/bin/env bash

set -euo pipefail

# Configuration
RUN_DURATION="1"
FIXED_FREQ="1020000000"

PERIODS=(
  "0.60"
  "0.70"
  "0.80"
  "0.90"
  "1.00"
  "1.10"
  "1.20"
  "1.30"
  "1.40"
  "1.50"
)

# problem sizes
MM_SIZE="1333"
STEREO_W="2304"
STEREO_H="1728"
QUASI_SIZE="16777216"
HIST_SIZE="65536000"
PARTICLE_SIZE="21948"
BFS_SIZE="2000000"

mkdir -p logs
mkdir -p logs/mm
mkdir -p logs/stereo
mkdir -p logs/quasi
mkdir -p logs/hist
mkdir -p logs/particle
mkdir -p logs/bfs

echo
echo "======================================"
echo "Setting GPU frequency to $FIXED_FREQ"
echo "======================================"
sudo ../scripts/set_gpu_freq.sh "$FIXED_FREQ"
sleep 1

for PERIOD in "${PERIODS[@]}"; do
  PERIOD_TAG="${PERIOD/./p}"

  echo
  echo "======================================"
  echo "Running tasks at period $PERIOD"
  echo "======================================"

  echo "Running mm..."
  ../tasks/mm "$MM_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/mm/mm_period_${PERIOD_TAG}.json"

  echo "Running stereo..."
  ../tasks/stereo "$STEREO_W" "$STEREO_H" "$PERIOD" "$RUN_DURATION" 1 "logs/stereo/stereo_period_${PERIOD_TAG}.json"

  echo "Running quasi..."
  ../tasks/quasi "$QUASI_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/quasi/quasi_period_${PERIOD_TAG}.json"

  echo "Running hist..."
  ../tasks/hist "$HIST_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/hist/hist_period_${PERIOD_TAG}.json"

  echo "Running particle..."
  ../tasks/particle "$PARTICLE_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/particle/particle_period_${PERIOD_TAG}.json"

  echo "Running bfs..."
  ../tasks/bfs "$BFS_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/bfs/bfs_period_${PERIOD_TAG}.json"

  echo "Finished all tasks at period $PERIOD"
  sleep 1
done

echo
echo "Complete"
