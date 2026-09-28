#!/bin/bash
# pB_decode_smoke.sh -- short decode smoke test on a 2-layer subset.
# Verifies (a) decode takes the UNTILED path (n=1 shape), (b) it still runs and
# produces sane logits, (c) peak stays small. Run under lockf.
set -u
HOME_DIR=$HOME
export P48_PKG="$HOME_DIR/dsv41-ws/B"
export P48_LAYERS=${P48_LAYERS:-20,24}
export P48_STEPS=${P48_STEPS:-10}
PY="$HOME_DIR/repos/exo/.venv/bin/python"

"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, os.environ["P48_PKG"])
from mlx_lm.models.deepseek_v41 import indexer as IX
for nb in (8192, 16384, 65536):
    r = IX.tiled(1, 1, 32, nb, 8)
    print(f"[pB] decode-shape routing nb={nb}: tile={r} "
          f"({'UNTILED' if r == nb else 'TILED'})", flush=True)
EOF

exec "$PY" -u "$HOME_DIR/dsv41-ws/B/p48_bench.py"
