#!/usr/bin/env bash
set -euo pipefail

RAIL="VDD_CPU_GPU_CV"
#RAIL="rail"

sudo -n python3 - <<PY
from jtop import jtop

rail = "${RAIL}"

with jtop() as jetson:
    jetson.ok()  # wait until first stats are ready
    p = jetson.power.get("rail", {})
    if p is None:
        raise SystemExit(f"{rail} not found. Available: {list(jetson.power.keys())}")
    print(int(p[rail].get("power", 0)))
PY

