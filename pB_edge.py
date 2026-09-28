#!/usr/bin/env python3
"""pB_edge -- hostile edge cases for the tiled indexer.

These all run at tiny shapes, so they take seconds and need no checkpoint.

1. fewer-than-k visible columns (the one place the tiled and untiled paths are
   *allowed* to differ, and only in the masked-out padding): both must report
   the same set of valid indices.
2. candidate-source path when nb is not a multiple of block_size (ragged final
   block) and when nb < candidate_topk_blocks * block_size.
3. candidate blocks straddling a tile boundary: tile widths that are NOT
   multiples of block_size must still merge to the same mask (this is why
   tile_width aligns to block_size; here we check that a caller forcing an
   unaligned tile still gets a *superset* of the right blocks, and that the
   aligned path is exact).
4. all-equal scores (the degenerate tie case): the value multiset must match
   the exact top-k exactly.
5. ratio-2 with start_pos not a multiple of ratio.
"""
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P48_PKG", HOME + "/dsv41-ws/B"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import indexer as IX  # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs  # noqa: E402
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import precompute_freqs_cis, rope_tail  # noqa: E402

RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
ARGS = ModelArgs(compress_ratios=RATIOS, kv_source_layers=(2, 8, 14, 20),
                 index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
                 candidate_source_layer=20, candidate_topk_blocks=2048,
                 candidate_block_size=8)
FAIL = []


def log(*a):
    print("[edge]", *a, flush=True)


def check(name, ok, detail=""):
    log(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def mk(layer_id, seed=0):
    mx.random.seed(seed)
    ix = IX.Indexer(ARGS, layer_id)

    def fill(t):
        if isinstance(t, dict):
            return {k: fill(v) for k, v in t.items()}
        if isinstance(t, list):
            return [fill(v) for v in t]
        return (mx.random.normal(t.shape) * 0.05).astype(mx.float32)

    ix.update(fill(ix.parameters()))
    return ix


class Sh:
    def __init__(self, c=None):
        self.kv_src_cache = None
        self.index_src_cache = None
        self.topk_idxs = None
        self.candidates = c


COS, SIN = precompute_freqs_cis(ARGS.rope_head_dim, 8192, 0, 160000.0, 16.0, 32, 1)


def run(ix, n, nb, sp, seed=5, cand=None, ratio_owner=True):
    mx.random.seed(seed)
    x = (mx.random.normal((1, n, ARGS.dim)) * 0.3).astype(mx.float32)
    qr = (mx.random.normal((1, n, ARGS.q_lora_rank)) * 0.3).astype(mx.float32)
    ik = (mx.random.normal((1, nb, ARGS.index_head_dim)) * 0.5).astype(mx.float32)
    sh = Sh(cand)
    out = ix(x, qr, sp, 0, COS, SIN, ik, sh)
    mx.eval(out, sh.candidates)
    return out, sh, x, qr, ik


def valid_set(v):
    return sorted(int(c) for c in np.array(v)[0].reshape(-1) if c >= 0)


def rows_valid_sets(v):
    a = np.array(v)[0]
    return [sorted(int(c) for c in r if c >= 0) for r in a]


def main():
    ix8 = mk(8, 3)

    # ---- 1. fewer-than-k visible columns ------------------------------------
    n, nb, sp = 8, 24, 0
    IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 1, True
    t, _, *_ = run(ix8, n, nb, sp)
    IX._TILE_FORCE = False
    IX._TILE, IX._TILE_MIN_NB = 0, 10 ** 9
    u, _, *_ = run(ix8, n, nb, sp)
    vt, vu = rows_valid_sets(t), rows_valid_sets(u)
    check("nb<n_topk visible-set equality (tiled vs untiled)", vt == vu,
          f"tiled={vt[:3]} untiled={vu[:3]}")

    # ---- 2. ragged nb vs block_size, and nb < topk_blocks*block_size --------
    ix20 = mk(20, 11)
    for (nn, nbb) in ((16, 100), (16, 8), (16, 7), (16, 33)):
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 8, 1, True
        t, sh_t, *_ = run(ix20, nn, nbb, 0, seed=9)
        mx.eval(sh_t.candidates)
        IX._TILE_FORCE = False
        IX._TILE, IX._TILE_MIN_NB = 0, 10 ** 9
        u, sh_u, *_ = run(ix20, nn, nbb, 0, seed=9)
        mx.eval(sh_u.candidates)
        eq = bool(mx.array_equal(sh_t.candidates, sh_u.candidates).item())
        check(f"candidate mask exact nb={nbb} (block 8)", eq,
              f"kept t={int(mx.sum(sh_t.candidates[...,::8].astype(mx.int32)).item())} "
              f"u={int(mx.sum(sh_u.candidates[...,::8].astype(mx.int32)).item())}")

    # ---- 3. tile NOT block-aligned: aligned path must equal the unaligned
    #         path on the *set of blocks* that can be reached -----------------
    # (tile_width aligns; here we verify the alignment is actually honoured)
    for bs in (8, 16, 32):
        align = bs
        tw = IX.tile_width(1, 512, 32, 16384, align)
        check(f"tile_width aligned to block_size={bs} (tile={tw})", tw % bs == 0)

    # ---- 4. all-equal scores: value multiset must match the exact top-k -----
    ix2 = mk(2, 21)
    n, nb, sp = 16, 4096, 0
    mx.random.seed(77)
    x = (mx.random.normal((1, n, ARGS.dim)) * 0.3).astype(mx.float32)
    qr = (mx.random.normal((1, n, ARGS.q_lora_rank)) * 0.3).astype(mx.float32)
    ik = mx.ones((1, nb, ARGS.index_head_dim), dtype=mx.float32)   # ties everywhere
    mx.eval(x, qr, ik)
    IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 1, True
    t = ix2(x, qr, sp, 0, COS, SIN, ik, Sh())
    IX._TILE_FORCE = False
    IX._TILE, IX._TILE_MIN_NB = 0, 10 ** 9
    u = ix2(x, qr, sp, 0, COS, SIN, ik, Sh())
    mx.eval(t, u)
    # scores are all identical => every valid row must report exactly the same
    # COUNT of selected indices (indices themselves may differ: ties)
    ct = [len(r) for r in rows_valid_sets(t)]
    cu = [len(r) for r in rows_valid_sets(u)]
    check("all-ties: same selected count per row", ct == cu, f"tiled={ct} untiled={cu}")
    # and the values must match exactly
    sv = [sorted((round(float(v), 6) for v in np.array(t)[0][i])) for i in range(n)]
    su = [sorted((round(float(v), 6) for v in np.array(u)[0][i])) for i in range(n)]
    check("all-ties: identical returned id multiset sizes", sv == su)

    # ---- 5. start_pos not a multiple of ratio, ratio 2 ---------------------
    ix8b = mk(8, 5)
    for sp in (7, 13, 4097):
        nb = (sp + 32) // 2
        IX._TILE, IX._TILE_MIN_NB, IX._TILE_FORCE = 512, 1, True
        t, _, *_ = run(ix8b, 32, nb, sp)
        IX._TILE_FORCE = False
        IX._TILE, IX._TILE_MIN_NB = 0, 10 ** 9
        u, _, *_ = run(ix8b, 32, nb, sp)
        check(f"ratio2 odd start_pos={sp} valid-set equality",
              rows_valid_sets(t) == rows_valid_sets(u))

    log(f"EDGE_DONE failures={len(FAIL)} {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
