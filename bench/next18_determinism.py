# Copyright © 2026 Adam Durham (hermes-gw)
"""Determinism probe (PREREG D4): is the small-n path self-consistent?

For a grid of small-n cells, run the SAME arm twice (identical inputs, identical
process config) and count index-tensor differences:
  * fallback fp32 row (L2-full)   A1 vs A2
  * fallback bf16 row (shipped)   B1 vs B2
  * hierarchical (HIER)           H1 vs H2
and the cross diffs (A vs H, B vs H). A nonzero self-diff on HIER means the
shipped production path itself is run-to-run unstable at small n, which makes an
INDEX-identity gate ill-posed there.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_determinism.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState


def _args(**o):
    b = dict(dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
             q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
             compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
             index_n_heads=2, index_head_dim=32, index_topk=512, max_seq_len=4096,
             candidate_source_layer=-1, candidate_topk_blocks=4, candidate_block_size=8)
    b.update(o)
    return ModelArgs(**b)


def inputs(seed, args, n, nb, sp=2 * 4096):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


def run(idx, inp, fence, bf16):
    I._ROW_DTYPE = mx.bfloat16 if bf16 else mx.float32
    I._FENCE_MIN_ROWS = fence
    o = idx(inp[0], inp[1], inp[2], 0, inp[3], inp[4], SharedState())
    mx.eval(o)
    return np.array(o)


if __name__ == "__main__":
    args = _args()
    print("=== determinism grid (identical inputs, repeated calls) ===", flush=True)
    rows = []
    for n in (1, 4, 16):
        for nb in (512, 4096, 16384):
            k = 512 if nb >= 512 else nb
            for seed in range(8):
                mx.random.seed(len(str(nb)))
                idx = I.Indexer(args, 0)
                mx.eval(idx.parameters())
                idx.index_topk = k
                inp = inputs(600000 + n * 97 + nb + seed, args, n, nb)
                A1 = run(idx, inp, 16, True); A2 = run(idx, inp, 16, True)   # shipped fp32? no: bf16
                # fallback fp32 row (L2-full):
                F1 = run(idx, inp, 16, False); F2 = run(idx, inp, 16, False)
                H1 = run(idx, inp, -1, True); H2 = run(idx, inp, -1, True)
                rows.append(dict(
                    n=n, nb=nb, k=k, seed=seed,
                    bf16row_self_diff=int((A1 != A2).sum()),
                    fp32row_self_diff=int((F1 != F2).sum()),
                    hier_self_diff=int((H1 != H2).sum()),
                    fp32row_vs_hier=int((F1 != H1).sum()),
                    bf16row_vs_hier=int((A1 != H1).sum())))
    for r in rows:
        if r["hier_self_diff"] or r["fp32row_self_diff"] or r["bf16row_self_diff"]:
            print("  !" + json.dumps(r), flush=True)
    print(json.dumps({"DETERMINISM": {
        "cells": len(rows),
        "hier_self_diff_cells": sum(1 for r in rows if r["hier_self_diff"]),
        "fp32row_self_diff_cells": sum(1 for r in rows if r["fp32row_self_diff"]),
        "bf16row_self_diff_cells": sum(1 for r in rows if r["bf16row_self_diff"]),
        "fp32row_vs_hier_diff_cells": sum(1 for r in rows if r["fp32row_vs_hier"]),
        "bf16row_vs_hier_diff_cells": sum(1 for r in rows if r["bf16row_vs_hier"]),
        "fp32row_vs_hier_total_slots": sum(r["fp32row_vs_hier"] for r in rows),
        "bf16row_vs_hier_total_slots": sum(r["bf16row_vs_hier"] for r in rows)}}), flush=True)
