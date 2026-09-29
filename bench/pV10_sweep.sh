#!/bin/bash
# pV10 -- sweep EXISTING knobs over the verify band (1-2 layers, <=8 GB).
#
# pV5 measured that the MoE gate+up kernel A2 costs ~0.028 ms per (row, expert)
# slot and that this cost is independent of how many DISTINCT experts the slots
# point at. A2 launches E_sel * (2*gu_tiles/A2_TILES) threadgroups, each running
# a short in-tile loop (in_tiles/4 per simdgroup), so per-threadgroup setup is a
# plausible share of it. EXL3_MOE_A2_TILES is an existing knob (A2T adjacent
# out-tiles per threadgroup) that has only ever been exercised at A2T=1 for
# decode; gu_tiles=72 so 2/4/8 all divide. This script measures it in the
# verify band, plus the B2 prologue chunk and the dense-GEMM variant.
#
# RUN ONLY WITH PRODUCTION DOWN.  Every arm prints peak memory (the harnesses
# do); all arms are 1-2 layers. NEVER run the 8-layer p48 harness (23 GB).
#
# Usage:  bash bench/pV10_sweep.sh [arms]
#   arms: a2t | chunk | dense | all   (default: all)
set -u
P=~/repos/exo/.venv/bin/python
H=$HOME/dsv41-ws2/V
L=${PV10_LAYERS:-20,21}
ROWS=${PV10_ROWS:-1,4,6}
export EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1
ARMS=${1:-all}
cd "$H" || exit 1

run4() {   # run4 <label> <extra-env...>
  local label=$1; shift
  echo "=== pV4 $label  env: $* ==="
  env "$@" PV4_PKG="$H" PV4_LAYERS="$L" PV4_ROWS="$ROWS" PV4_REPS=12 \
    lockf -k ~/dsv41-gpu.lock "$P" bench/pV4_verify.py 2>&1 \
    | grep -E "pV4|rror" | tail -14
}

if [ "$ARMS" = "a2t" ] || [ "$ARMS" = "all" ]; then
  for T in 1 2 4; do run4 "A2_TILES=$T" EXL3_MOE_A2_TILES=$T; done
fi
if [ "$ARMS" = "chunk" ] || [ "$ARMS" = "all" ]; then
  for C in 0 128 256; do run4 "MOE_CHUNK=$C" EXL3_MOE_CHUNK=$C; done
fi
if [ "$ARMS" = "dense" ] || [ "$ARMS" = "all" ]; then
  run4 "GEMM_DEVX=1 (default)" EXL3_GEMM_DEVX=1
  run4 "GEMM_DEVX=0"           EXL3_GEMM_DEVX=0
  run4 "MOE_V2=0 (v1 kernels)" EXL3_MOE_V2=0
fi
echo "PV10_DONE"
