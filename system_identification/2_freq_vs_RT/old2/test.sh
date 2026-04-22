#!/usr/bin/env bash

set -euo pipefail

# Configuration
PERIOD="0.70"
RUN_DURATION="60"

FREQS=(
  #306000000
  #408000000
  #510000000
  #612000000
  #714000000
  #816000000
  #918000000
  1020000000
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

for FREQ in "${FREQS[@]}"; do
  echo
  echo "======================================"
  echo "Setting GPU frequency to $FREQ"
  echo "======================================"
  sudo ../scripts/set_gpu_freq.sh "$FREQ"
  sleep 1

  echo "Running mm..."
  ../tasks/mm "$MM_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/mm/mm_$FREQ.json"

  echo "Running stereo..."
  ../tasks/stereo "$STEREO_W" "$STEREO_H" "$PERIOD" "$RUN_DURATION" 1 "logs/stereo/stereo_$FREQ.json"

  echo "Running quasi..."
  ../tasks/quasi "$QUASI_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/quasi/quasi_$FREQ.json"

  echo "Running hist..."
  ../tasks/hist "$HIST_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/hist/hist_$FREQ.json"

  echo "Running particle..."
  ../tasks/particle "$PARTICLE_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/particle/particle_$FREQ.json"

  echo "Running bfs..."
  ../tasks/bfs "$BFS_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/bfs/bfs_$FREQ.json"

  echo "Finished all tasks at frequency $FREQ"
  sleep 1
done

echo
echo "Complete"
