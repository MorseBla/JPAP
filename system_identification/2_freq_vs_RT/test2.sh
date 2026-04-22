#!/usr/bin/env bash

set -euo pipefail

# Configuration
PERIOD="0.100"
RUN_DURATION="60"

FREQS=(
  306000000
  408000000
  510000000
  612000000
  714000000
  816000000
  918000000
  1020000000
  
)

# problem sizes
MM_SIZE="1333"
STEREO_W="2304"
STEREO_H="1728"
QUASI_SIZE="16777216"
HIST_SIZE="65536000"
PARTICLE_SIZE="21948"
BFS_SIZE="20000000"


mkdir -p logs
mkdir -p logs/mm
mkdir -p logs/stereo
mkdir -p logs/quasi
mkdir -p logs/hist
mkdir -p logs/particle
mkdir -p logs/bfs

for FREQ in "${FREQS[@]}"; do
  mkdir -p logs/mm/$FREQ
  mkdir -p logs/stereo/$FREQ
  mkdir -p logs/quasi/$FREQ
  mkdir -p logs/hist/$FREQ
  mkdir -p logs/particle/$FREQ
  mkdir -p logs/bfs/$FREQ
  echo
  echo "======================================"
  echo "Setting GPU frequency to $FREQ"
  echo "======================================"
  sudo ../scripts/set_gpu_freq.sh "$FREQ"
  sleep 1

  #echo "Running mm..."
  #../tasks/mm_new "$MM_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/mm/$FREQ"

  #echo "Running stereo..."
  ../tasks/stereo_new "$STEREO_W" "$STEREO_H" "$PERIOD" "$RUN_DURATION" 1 "logs/stereo/$FREQ"

  #echo "Running quasi..."
  #../tasks/quasi_new "$QUASI_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/quasi/$FREQ"

  #echo "Running hist..."
  #../tasks/hist_new "$HIST_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/hist/$FREQ"

  #echo "Running particle..."
  #../tasks/particle_new "$PARTICLE_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/particle/$FREQ"

  #echo "Running bfs..."
  #../tasks/bfs_new "$BFS_SIZE" "$PERIOD" "$RUN_DURATION" 1 "logs/bfs/$FREQ"

  echo "Finished all tasks at frequency $FREQ"
  sleep 1
done

echo
echo "Complete"
