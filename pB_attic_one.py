#!/usr/bin/env python3
"""pB_attic_one -- run one indexer shape from ONE package root, save the result.

Called twice (original tree vs worktree) by pB_attic.sh; the driver compares the
saved .npy files. Env: P48_PKG (root), PB_LAYER, PB_N, PB_NB, PB_SP, PB_TILE,
PB_OUT. Prints the peak allocation.
"""
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ["P48_PKG"])
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import indexer as IX  # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import precompute_freqs_cis  # noqa: E402

RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
ARGS = ModelArgs(compress_ratios=RATIOS, kv_source_layers=(2, 8, 14, 20),
                 index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
                 candidate_source_layer=20, candidate_topk_blocks=2048,
                 candidate_block_size=8)

LAYER = int(os.environ.get("PB_LAYER", "20"))
N = int(os.environ.get("PB_N", "32"))
NB = int(os.environ.get("PB_NB", "16384"))
SP = int(os.environ.get("PB_SP", "0"))
TILE = int(os.environ.get("PB_TILE", "-1"))
OUT = os.environ["PB_OUT"]

mx.random.seed(0)
ix = IX.Indexer(ARGS, LAYER)


def fill(t):
    if isinstance(t, dict):
        return {k: fill(v) for k, v in t.items()}
    if isinstance(t, list):
        return [fill(v) for v in t]
    return (mx.random.normal(t.shape) * 0.05).astype(mx.float32)


ix.update(fill(ix.parameters()))

COS, SIN = precompute_freqs_cis(64, 32768, 0, 160000.0, 16.0, 32, 1)
mx.random.seed(1234)
x = (mx.random.normal((1, N, 5120)) * 0.3).astype(mx.float32)
qr = (mx.random.normal((1, N, 1280)) * 0.3).astype(mx.float32)
ik = (mx.random.normal((1, NB, 128)) * 0.5).astype(mx.float32)
cand = None
if LAYER == 24:
    mx.random.seed(99)
    cand = mx.random.uniform(shape=(1, N, NB)) > 0.5
    mx.eval(cand)
if TILE >= 0 and hasattr(IX, "_TILE"):
    IX._TILE, IX._TILE_MIN_NB = TILE, 1
    if hasattr(IX, "_TILE_FORCE"):
        IX._TILE_FORCE = True     # prove the tiled path itself, at any nb


class Sh:
    def __init__(self, c=None):
        self.kv_src_cache = None
        self.index_src_cache = None
        self.topk_idxs = None
        self.candidates = c


sh = Sh(cand)
out = ix(x, qr, SP, SP + N, COS, SIN, ik, sh)
mx.eval(out)
np.save(OUT + ".idx.npy", np.array(out))
if sh.candidates is not None:
    mx.eval(sh.candidates)
    np.save(OUT + ".cand.npy", np.array(sh.candidates))
print(f"[one] root={os.environ['P48_PKG']} L{LAYER} n={N} nb={NB} sp={SP} tile={TILE} "
      f"peak={mx.get_peak_memory()/1e6:.1f}MB cand={'yes' if sh.candidates is not None else 'no'} "
      f"saved={OUT}", flush=True)
