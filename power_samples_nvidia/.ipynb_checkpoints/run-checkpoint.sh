#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$HOME/Srini/ECRTS_power_aware/power_samples"
OUTDIR="$WORKDIR/results"
mkdir -p "$OUTDIR"

cd "$WORKDIR"

sources=(
  "histogram_converted.cu"
  "bfs_converted.cu"
  "mm_converted.cu"
  "old_hotspot_converted.cu"
  "particle_converted.cu"
  "quasirand_converted.cu"
  "stereodisparity_converted.cu"
)

echo "Compiling benchmarks..."
for src in "${sources[@]}"; do
  exe="${src%.cu}"
  echo "  -> $src"
  nvcc -O2 -std=c++17 -o "$exe" "$src"
done

echo
echo "Running benchmarks..."
for src in "${sources[@]}"; do
  exe="${src%.cu}"
  json="$OUTDIR/${exe}.json"

  echo "========================================"
  echo "Running $exe"
  echo "Output: $json"
  echo "========================================"

  if [[ "$exe" == "mm_converted" ]]; then
    ./"$exe" "$json" 1600
  else
    ./"$exe" "$json"
  fi

  echo
done

echo "Done."
echo "JSON files saved in: $OUTDIR"
