#!/usr/bin/env bash

LINE=$(sudo tegrastats --interval 1 | head -n 1)
echo "$LINE" | grep -oP "VDD_CPU_GPU_CV \K[0-9]+"
