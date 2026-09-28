#!/usr/bin/env python3
"""pB_memprobe -- isolate which op drives the indexer peak allocation.

Runs each stage alone in one process, resetting the peak between stages, so the
printed number is that stage's own high-water mark.
"""
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P48_PKG", HOME + "/dsv41-ws/B"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import precompute_freqs_cis, rope_tail  # noqa: E402

N = int(os.environ.get("PB_N", "512"))
NB = int(os.environ.get("PB_NB", "16384"))
H, D, RD = 32, 128, 64
TILE = int(os.environ.get("PB_TILE", "512"))


def log(*a):
    print("[mem]", *a, flush=True)


def stage(name, fn):
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out if isinstance(out, mx.array) else out[0])
    if isinstance(out, (list, tuple)):
        mx.eval(*[o for o in out if isinstance(o, mx.array)])
    dt = time.perf_counter() - t0
    log(f"{name:44s} {dt*1e3:8.1f} ms  peak={(mx.get_peak_memory()-base)/1e6:9.1f} MB "
        f"active={(mx.get_active_memory()-base)/1e6:8.1f} MB")
    return out


mx.random.seed(0)
ik32 = (mx.random.normal((1, NB, D)) * 0.5).astype(mx.float32)
q = (mx.random.normal((1, N, H, D)) * 0.5).astype(mx.float32)
w = (mx.random.normal((1, N, H)) * 0.5).astype(mx.float32)
mx.eval(ik32, q, w)
log(f"n={N} nb={NB} h={H} d={D} tile={TILE}")

# 1. einsum over the FULL nb, then head-sum (the untiled path's core)
def full():
    s = mx.einsum("bshd,btd->bsht", q, ik32)
    s = mx.maximum(s, 0.0) * w[..., None]
    return mx.sum(s, axis=2)


s_full = stage("einsum+relu+sum over full nb (untiled core)", full)

# 2. the same restricted to one tile
def tiled_slice():
    c0, c1 = 0, min(TILE, NB)
    s = mx.einsum("bshd,btd->bsht", q, ik32[:, c0:c1])
    s = mx.maximum(s, 0.0) * w[..., None]
    return mx.sum(s, axis=2)


s_tile = stage("einsum+relu+sum one tile (pre-collapse kept)", tiled_slice)
# same thing but keeping the pre-collapse tensor live (what the peak is about)
def tiled_slice_raw():
    c0, c1 = 0, min(TILE, NB)
    return mx.einsum("bshd,btd->bsht", q, ik32[:, c0:c1])


r = stage("pre-collapse (b,s,h,tile) alone", tiled_slice_raw)
mx.eval(r)
assert mx.array_equal(mx.sum(mx.maximum(r, 0.0) * w[..., None], axis=2), s_tile)
del r

# 3. the merge
k = 512
bv = mx.full((1, N, k), float("-inf"), mx.float32)
bi = mx.zeros((1, N, k), mx.int32)
cols = mx.arange(0, min(TILE, NB), dtype=mx.int32)
stage("merge_topk one tile", lambda: (
    mx.concatenate([bv, s_tile], -1), mx.concatenate(
        [bi, mx.broadcast_to(cols[None, None, :], (1, N, min(TILE, NB)))], -1)))

# 4. full loop, per-tile eval (what indexer._tiled_scores does)
def loop_eval():
    best_v = mx.full((1, N, k), float("-inf"), mx.float32)
    best_i = mx.zeros((1, N, k), mx.int32)
    for c0 in range(0, NB, TILE):
        c1 = min(c0 + TILE, NB)
        s = mx.einsum("bshd,btd->bsht", q, ik32[:, c0:c1])
        s = mx.maximum(s, 0.0) * w[..., None]
        s = mx.sum(s, axis=2)
        mx.eval(s)
        cc = mx.arange(c0, c1, dtype=mx.int32)
        v = mx.concatenate([best_v, s], -1)
        i = mx.concatenate([best_i, mx.broadcast_to(cc[None, None, :], (1, N, c1 - c0))], -1)
        p = mx.argpartition(-v, k - 1, axis=-1)[..., :k]
        best_v = mx.take_along_axis(v, p, axis=-1)
        best_i = mx.take_along_axis(i, p, axis=-1)
    return best_i


stage("FULL tiled loop, per-tile mx.eval", loop_eval)


# 4b. full loop with the merge materialized each iteration too
def loop_eval2():
    best_v = mx.full((1, N, k), float("-inf"), mx.float32)
    best_i = mx.zeros((1, N, k), mx.int32)
    for c0 in range(0, NB, TILE):
        c1 = min(c0 + TILE, NB)
        s = mx.einsum("bshd,btd->bsht", q, ik32[:, c0:c1])
        s = mx.maximum(s, 0.0) * w[..., None]
        s = mx.sum(s, axis=2)
        mx.eval(s)
        cc = mx.arange(c0, c1, dtype=mx.int32)
        v = mx.concatenate([best_v, s], -1)
        i = mx.concatenate([best_i, mx.broadcast_to(cc[None, None, :], (1, N, c1 - c0))], -1)
        p = mx.argpartition(-v, k - 1, axis=-1)[..., :k]
        best_v = mx.take_along_axis(v, p, axis=-1)
        best_i = mx.take_along_axis(i, p, axis=-1)
        mx.eval(best_v, best_i)
    return best_i


stage("FULL tiled loop, eval tile + merge", loop_eval2)

# 5. same loop WITHOUT the per-tile eval
def loop_noeval():
    best_v = mx.full((1, N, k), float("-inf"), mx.float32)
    best_i = mx.zeros((1, N, k), mx.int32)
    for c0 in range(0, NB, TILE):
        c1 = min(c0 + TILE, NB)
        s = mx.einsum("bshd,btd->bsht", q, ik32[:, c0:c1])
        s = mx.maximum(s, 0.0) * w[..., None]
        s = mx.sum(s, axis=2)
        cc = mx.arange(c0, c1, dtype=mx.int32)
        v = mx.concatenate([best_v, s], -1)
        i = mx.concatenate([best_i, mx.broadcast_to(cc[None, None, :], (1, N, c1 - c0))], -1)
        p = mx.argpartition(-v, k - 1, axis=-1)[..., :k]
        best_v = mx.take_along_axis(v, p, axis=-1)
        best_i = mx.take_along_axis(i, p, axis=-1)
    return best_i


stage("FULL tiled loop, no per-tile eval", loop_noeval)
log("MEMPROBE_DONE")
