# Copyright © 2026 Adam Durham (hermes-gw)
"""Diagnostic: classify the residual ROW_BF16=0 divergences.

Q1: are the fallback's shared-strip fp32 scores BITWISE equal to the
    hierarchical exact pass's per-row gathered fp32 scores on the same columns?
Q2: for a divergent cell, is the difference an EXACT fp32 tie, a ~1-ulp
    near-tie (different rounding path), or a genuine value loss?

Run:  PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_diffdiag.py
"""
from __future__ import annotations

import json

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0


def _args(**o):
    b = dict(dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
             q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
             compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
             index_n_heads=2, index_head_dim=32, index_topk=512, max_seq_len=4096,
             candidate_source_layer=-1, candidate_topk_blocks=4, candidate_block_size=8)
    b.update(o)
    return ModelArgs(**b)


def build(seed, n, nb, sp, args):
    mx.random.seed(seed)
    idx = I.Indexer(args, 0)
    mx.eval(idx.parameters())
    rng = np.random.default_rng(seed + 999)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    mx.eval(q, w, lens)
    return idx, q.astype(mx.float32), ik, w.astype(mx.float32), lens.astype(mx.int32)


def main():
    args = _args()
    out = {}

    print("=== Q1: shared-strip vs gathered fp32 score parity (bitwise) ===", flush=True)
    q1 = []
    for seed in range(3):
        n, nb = 8, 2048
        idx, q, ik, w, lens = build(1000 + seed, n, nb, 2 * nb, args)
        # same columns, both scorers, fp32
        s_shared = H._score_shared_columns(q, ik.astype(mx.float32), w)      # [b,n,nb]
        safe = mx.arange(nb, dtype=mx.int32)[None, :]
        keys = H._gather_rows(ik.astype(mx.float32),
                              mx.broadcast_to(mx.arange(nb, dtype=mx.int32)[None, None, :],
                                              (1, n, nb)))
        s_gath = H._score_gathered_columns(q, keys, w)                        # [b,n,nb]
        mx.eval(s_shared, s_gath)
        bitwise = bool(np.array_equal(np.array(s_shared), np.array(s_gath)))
        maxabs = float(np.abs(np.array(s_shared) - np.array(s_gath)).max())
        q1.append(dict(seed=seed, bitwise_equal=bitwise, maxabs_diff=maxabs))
        print("  " + json.dumps(q1[-1]), flush=True)
    out["q1_shared_vs_gathered"] = q1

    print("=== Q2: classify residual divergences at k-boundary (fp32-row fallback vs HIER) ===", flush=True)
    rows = []
    for (n, nb, k) in ((4, 4096, 512), (16, 16384, 512), (16, 4096, 512), (2, 16384, 512)):
        for seed in range(6):
            s = 500000 + n * 31 + nb + seed
            idx, q, ik, w, lens = build(s, n, nb, 2 * nb, args)
            # fallback fp32 row
            sc = mx.einsum("bshd,btd->bsht", q, ik.astype(mx.float32))
            sc = mx.maximum(sc, 0.0) * w[..., None]
            row = mx.sum(sc, axis=2).astype(mx.float32)
            vis = mx.arange(nb)[None, :] < lens
            row = mx.where(vis[None], row, float("-inf"))
            mx.eval(row)
            vf, ifb = I.topk_from_row(row, k)
            mx.eval(vf, ifb)
            # HIER
            hv, hi, _ = H.hierarchical_topk_prod(q, ik, w, lens, k, I._HIER_BLOCK,
                                                 I._HIER_STRIP, 4096, 16)
            mx.eval(hv, hi)
            vr = np.array(vf).reshape(-1); ir = np.array(ifb).reshape(-1)
            hr = np.array(hv.astype(mx.float32)).reshape(-1) if False else np.array(hv).reshape(-1)
            hi_r = np.array(hi).reshape(-1)
            nd = int((ir != hi_r).sum())
            # boundary values
            kth_fb = float(np.sort(np.array(row).reshape(-1))[::-1][k - 1])
            kth_hi = float(np.sort(np.array(hv).reshape(-1))[::-1][k - 1])
            # for differing slots, gap between fallback value and HIER value
            if nd:
                # value gap at the k-th boundary (fallback minus hier) relative to scale
                scale = abs(kth_fb) if kth_fb else 1.0
                rel = (kth_fb - kth_hi) / scale if scale else 0.0
            else:
                rel = 0.0
            rows.append(dict(n=n, nb=nb, k=k, seed=s, idiff=nd,
                             kth_fallback=kth_fb, kth_hier=kth_hi,
                             kth_rel_gap=rel))
            print("  " + json.dumps(rows[-1]), flush=True)
    out["q2_boundary_class"] = rows
    tot = sum(r["idiff"] for r in rows)
    print(json.dumps({"DIFFDIAG": {"cells": len(rows), "total_idiff": tot,
                                   "max_rel_kth_gap": max(abs(r["kth_rel_gap"]) for r in rows)}}),
          flush=True)


if __name__ == "__main__":
    main()
