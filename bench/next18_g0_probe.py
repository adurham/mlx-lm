# Copyright © 2026 Adam Durham (hermes-gw)
"""G0 probe: is the shipped HIER path exact vs the full-width fp32 truth?

Decides premise (2). Compares, on identical inputs:
  TRUTH  = full-width fp32 score row -> topk_from_row   (exact, no coarse pass)
  HIER   = hierarchical_topk_prod(..., overfetch=16)      (shipped)
and classifies each differing slot as a VALUE LOSS (HIER missing a higher-scoring
column) or an INDEX TIE (same value, different pick).

Also reproduces the brief's fp32-row vs bf16-row fallback comparison.

Run:  PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_g0_probe.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
from mlx_lm.models.deepseek_v41.config import ModelArgs


def _args(**over):
    base = dict(
        dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
        compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=512, max_seq_len=4096,
        candidate_source_layer=-1, candidate_topk_blocks=4, candidate_block_size=8)
    base.update(over)
    return ModelArgs(**base)


def make_indexer(args, layer_id, seed):
    mx.random.seed(seed)
    idx = I.Indexer(args, layer_id)
    mx.eval(idx.parameters())
    return idx


def derive(idx, seed, n, nb, sp):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, idx_args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, idx_args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, idx_args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, idx_args.rope_head_dim // 2).astype(np.float32))
    # score inputs exactly as __call__ / capture harness
    from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
    from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    mx.eval(q, w, lens)
    return q.astype(mx.float32), ik, w.astype(mx.float32), lens.astype(mx.int32)


def full_width_row(q32, ik, w32, lens, dtype):
    """Full-width [b,n,nb] score row in `dtype` (mirrors the untiled fallback)."""
    s = mx.einsum("bshd,btd->bsht", q32, ik.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w32[..., None]
    s = mx.sum(s, axis=2).astype(dtype)
    vis = mx.arange(ik.shape[1])[None, :] < lens
    s = mx.where(vis[None], s, float("-inf"))
    mx.eval(s)
    return s


def _npf(a):
    """mlx -> numpy, upcasting bf16 (no PEP3118 format) to fp32 first."""
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.array(a)


def classify(v_ref, i_ref, hv, hi, k):
    """Compare two (value,index) top-k sets. Return loss/tie counts."""
    vr = v_ref.reshape(-1)
    hr = hv.reshape(-1)
    ir = i_ref.reshape(-1)
    hi_r = hi.reshape(-1)
    # index-level
    ndiff = int((ir != hi_r).sum())
    # value multiset diff: sort each descending, compare rank-wise
    vr_s = np.sort(vr)[::-1]
    hr_s = np.sort(hr)[::-1]
    # count rank positions where the k-th-largest value differs materially
    d = np.abs(vr_s - hr_s)
    # a "value loss" is a slot where HIER's value is strictly below truth beyond fp32 eps
    scale = max(1e-6, float(np.abs(vr_s).max()))
    lost = int((vr_s - hr_s > 1e-4 * scale).sum())
    return ndiff, lost, float(d.max()) if d.size else 0.0


if __name__ == "__main__":
    idx_args = _args()  # module-global used by derive()
    print("=== G0 PROBE: HIER vs full-width fp32 TRUTH (plain role, no candidates) ===", flush=True)
    rows = []
    for n in (1, 2, 4, 16):
        for nb in (512, 4096, 16384):
            k = 512 if nb >= 512 else nb
            for seed in range(4):
                s = 70000 + n * 131 + nb + seed
                idx = make_indexer(idx_args, 0, seed=1)
                q, ik, w, lens = derive(idx, s, n, nb, sp=2 * nb)
                block, cstrip, estrip = I._HIER_BLOCK, I._HIER_STRIP, 4096

                # TRUTH: fp32 full-width row
                row32 = full_width_row(q, ik, w, lens, mx.float32)
                v_ref, i_ref = I.topk_from_row(row32, k)
                mx.eval(v_ref, i_ref)

                # HIER shipped (overfetch=16), coarse always bf16
                hv, hi, _ = H.hierarchical_topk_prod(
                    q, ik, w, lens, k, block, cstrip, estrip, 16)
                mx.eval(hv, hi)
                ndiff, lost, vmax = classify(
                    np.array(v_ref), np.array(i_ref), np.array(hv), np.array(hi), k)

                # bf16-row fallback (production) for the brief's comparison
                rowbf = full_width_row(q, ik, w, lens, mx.bfloat16)
                vbf, ibf = I.topk_from_row(rowbf, k)
                mx.eval(vbf, ibf)
                ndiff_bf, lost_bf, _ = classify(
                    np.array(v_ref), np.array(i_ref), _npf(vbf), np.array(ibf), k)

                row = dict(n=n, nb=nb, k=k, seed=s,
                           hier_vs_truth_idiff=ndiff, hier_vs_truth_valueloss=lost,
                           hier_vs_truth_maxv=vmax,
                           bf16row_vs_fp32truth_idiff=ndiff_bf,
                           bf16row_vs_fp32truth_valueloss=lost_bf,
                           nhier_slots=int(hi.size))
                rows.append(row)
                print("  " + json.dumps(row), flush=True)

    tot = sum(r["nhier_slots"] for r in rows)
    tot_idiff = sum(r["hier_vs_truth_idiff"] for r in rows)
    tot_lost = sum(r["hier_vs_truth_valueloss"] for r in rows)
    print(json.dumps({"G0_HIER_VS_TRUTH": {
        "cells": len(rows), "slots": tot,
        "hier_idiff_vs_fp32truth": tot_idiff,
        "hier_VALUE_LOSS_vs_fp32truth": tot_lost}}), flush=True)
