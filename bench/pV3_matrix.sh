#!/bin/bash
# pV3 -- verify-marginal ablation matrix on the 8-layer subset (phase-17 method).
# Runs p48_bench.py (copy in ~/dsv41-ws2/V/bench) for rows 1/4/5 and each stub,
# ONE job at a time under the GPU lock, printing ms/step per arm.
# Usage: bash bench/pV3_matrix.sh [rows] [stubs]
set -e
ROWS=${1:-"1 4 5"}
STUBS=${2:-"none experts lin"}
L=${PV3_LAYERS:-"0,1,2,3,20,21,24,25"}
export EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1
P=~/repos/exo/.venv/bin/python
cd ~/dsv41-ws2/V
for stub in $STUBS; do
  for r in $ROWS; do
    echo "=== stub=$stub ROWS=$r layers=$L ==="
    P48_PKG=$HOME/dsv41-ws2/V P48_LAYERS=$L P48_ROWS=$r P48_STUB=$stub \
      P48_STEPS=${PV3_STEPS:-14} lockf -k ~/dsv41-gpu.lock $P bench/p48_bench.py 2>&1 \
      | grep -E "p48|Error|error" | tail -6
  done
done
echo P V3_DONE
