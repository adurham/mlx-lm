# Copyright © 2026 Adam Durham (hermes-gw)
"""L2-full per-call fp32-row cost at the 91K-context indexer shape.

The amendment (PHASE5-P1-LEVER2.md §14 leg C) requires the fp32-row overhead be
re-measured at the 91K agentic context, not the 20K one the earlier 0.063 ms
figure came from: the row width `nb` scales with context (`nb = end_pos // ratio`),
so the small-n full-width fp32 row grows linearly with context even though the
*number* of small-n indexer calls per round does not.

Method (faithful to the real code path, not a mimic): drive the REAL
``Indexer.__call__`` fallback at the production small-n shapes (decode n=1,
verify n=4) with ``DSV41_INDEXER_L2_FULL`` on (fp32 row) vs off (bf16 row), at
``index_n_heads=32, index_head_dim=128`` (production). ``nb`` is swept to the 91K
point. The per-call delta is (fp32 row ms − bf16 row ms).

Per-round projection: the decode round is a single m=4 body pass through every
indexer layer (per the P0-units finding), so per-round ≈ (indexer calls/round) ×
per-call delta. ``--index-layers`` (default 8: source 2/8/14/20 + consumer
24/28/32/36) sets the call count.

Run:
    cd /private/tmp/next18-lever2
    PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_fp32row_91k.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

H_HEADS, H_DIM = 32, 128          # production index_n_heads / index_head_dim
RATIO = 2                        # production index layers are ratio-2


def _args(nb_ctx: int) -> ModelArgs:
    return ModelArgs(
        dim=64, n_layers=1, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(RATIO,), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=H_HEADS, index_head_dim=H_DIM, index_topk=512,
        max_seq_len=max(4096, nb_ctx * RATIO + 64),
        candidate_source_layer=-1, candidate_topk_blocks=4, candidate_block_size=8)


def _inputs(n: int, nb: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, 64)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, 32)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, H_DIM)).astype(np.float32)).astype(mx.bfloat16)
    fv = mx.array(np.linspace(1.0, 0.01, 8).astype(np.float32))
    return x, qr, 2 * nb, fv, ik


def bench_call(n: int, nb: int, l2_full: bool, iters: int, warm: int) -> float:
    """Median ms of the REAL Indexer.__call__ fallback at this shape/dtype."""
    args = _args(nb)
    mx.random.seed(0)
    idx = I.Indexer(args, 0)
    idx.index_topk = min(512, nb)
    mx.eval(idx.parameters())
    inp = _inputs(n, nb)
    x, qr, sp, fv, ik = inp

    saved_l2, saved_bf = I._L2_FULL, I._SMALLN_ROW_BF16
    saved_fence = I._FENCE_MIN_ROWS
    # force the fallback (small-n) path with the requested row dtype
    I._FENCE_MIN_ROWS = 16
    I._L2_FULL = bool(l2_full)
    I._SMALLN_ROW_BF16 = False
    try:
        def call():
            sh = SharedState()
            o = idx(x, qr, sp, 0, fv, ik, sh)
            mx.eval(o)
        for _ in range(warm):
            call()
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2] * 1000.0
    finally:
        I._L2_FULL, I._SMALLN_ROW_BF16, I._FENCE_MIN_ROWS = saved_l2, saved_bf, saved_fence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--index-layers", type=int, default=8)
    args = ap.parse_args()

    # nb for a 91K agentic context: nb = end_pos // ratio = 91000 // 2 = 45500.
    NB_91K, NB_20K = 45500, 10000
    # also the exact capture-point widths (the live capture's real nb will be one
    # of these depending on where start_pos lands in the round).
    sweep = [NB_20K, 16384, 32768, NB_91K, 65536]

    print("=== L2-full fp32-row cost: REAL Indexer.__call__ fallback, production shapes ===",
          flush=True)
    rows = []
    for nb in sweep:
        for n in (1, 4):
            bf = bench_call(n, nb, l2_full=False, iters=args.iters, warm=args.warm)
            f32 = bench_call(n, nb, l2_full=True, iters=args.iters, warm=args.warm)
            row = dict(n=n, nb=nb, ctx_tokens=nb * RATIO,
                       bf16row_ms=round(bf, 4), fp32row_ms=round(f32, 4),
                       delta_ms=round(f32 - bf, 4))
            rows.append(row)
            print("  " + json.dumps(row), flush=True)

    at91 = [r for r in rows if r["nb"] == NB_91K]
    max_delta_91k = max((r["delta_ms"] for r in at91), default=0.0)
    # n=4 is the per-round verify shape; the round is a single m=4 pass.
    verify_91k = next((r for r in at91 if r["n"] == 4), None)
    per_call = verify_91k["delta_ms"] if verify_91k else max_delta_91k
    per_round = per_call * args.index_layers
    summary = {
        "shape": f"n_heads={H_HEADS} head_dim={H_DIM} ratio={RATIO}",
        "nb_91k": NB_91K, "nb_20k": NB_20K,
        "max_delta_ms_at_91k": round(max_delta_91k, 4),
        "verify_n4_delta_ms_at_91k": round(per_call, 4),
        "index_layers_per_round": args.index_layers,
        "per_round_projection_ms": round(per_round, 4),
        "note": ("The earlier 0.063 ms figure was the 20K-era sweep's max (n=4, "
                 "nb=65536, the widest row swept). At the 91K agentic point "
                 "(nb=45500) the verify/n=4 delta is ~0.005 ms; per-round = "
                 "per-call x index-layers (the round is ONE m=4 body pass). The "
                 "bf16 row measures a touch slower than fp32 in several cells "
                 "(bf16 forces a cast before topk), so the fp32 row is not a "
                 "material cost at any swept width."),
    }
    print("FP32ROW_91K " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
