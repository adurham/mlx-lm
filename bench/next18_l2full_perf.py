# Copyright © 2026 Adam Durham (hermes-gw)
"""Per-call cost of the L2-full small-n full-width fp32 row vs the bf16 row.

Measures the indexer's score+top-k call at the PRODUCTION small-n shapes
(decode n=1, verify n=4) at representative nb, for:
  * ROW_BF16=1 row (shipped production fallback dtype)
  * L2-full fp32 row (the design)
and reports the per-call delta (ms). nb is swept to the production depth.

The number that matters for the selection rule: is L2-full's small-n full-width
path within ~2-3 ms of the shipped (bf16-row) path? If so, L2-full is preferred
and L2-guard is unnecessary.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_l2full_perf.py
"""
from __future__ import annotations
import json
import time
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I

H_HEADS, H_DIM = 32, 128          # production index_n_heads / index_head_dim


def bench_row(n, nb, dtype, iters=40, warm=5):
    bsz = 1
    q = mx.array(np.random.default_rng(0).standard_normal((bsz, n, H_HEADS, H_DIM)).astype(np.float32)) * 0.3
    ik = mx.array(np.random.default_rng(1).standard_normal((1, nb, H_DIM)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((np.random.default_rng(2).random((bsz, n, H_HEADS)) + 0.5).astype(np.float32))
    # mimic _tiled_scores_buffer's row path with tile == nb (untiled) and the dtype
    def call():
        s = mx.einsum("bshd,btd->bsht", q, ik.astype(mx.float32))
        s = mx.maximum(s, 0.0) * w[..., None]
        s = mx.sum(s, axis=2).astype(dtype)
        v, i = I.topk_from_row(s, min(512, nb))
        mx.eval(v, i)
    for _ in range(warm):
        call()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] * 1000.0


if __name__ == "__main__":
    print("=== L2-full per-call cost (fp32 row) vs bf16 row; production shapes ===", flush=True)
    rows = []
    for n in (1, 4, 16):
        for nb in (4096, 16384, 65536):
            bf = bench_row(n, nb, mx.bfloat16)
            f32 = bench_row(n, nb, mx.float32)
            rows.append(dict(n=n, nb=nb, bf16_row_ms=round(bf, 4), fp32_row_ms=round(f32, 4),
                             delta_ms=round(f32 - bf, 4)))
            print("  " + json.dumps(rows[-1]), flush=True)
    print(json.dumps({"L2FULL_PERF": {"max_delta_ms": max(r["delta_ms"] for r in rows),
                                      "note": "delta = fp32 row minus bf16 row per indexer call"}}),
          flush=True)
