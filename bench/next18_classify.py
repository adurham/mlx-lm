# Copyright © 2026 Adam Durham (hermes-gw)
"""Definitive classifier for the residual (fp32-everywhere) divergences.

For a divergent cell we compute the fallback's EXACT fp32 full-width row R
(the "value truth") and answer, per differing output slot, whether the two
paths picked columns of EQUAL fp32 score (a tie) or whether one path lost a
higher-scoring column (a value loss). We also size the tie class at the k-th
boundary.

Cells covered: plain (residual at ROW_BF16=0 / L2-full) and consumer
(consumer_skip ON vs the full-width masked fallback), the two strata where the
residual concentrates.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_classify.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState
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


def inputs(seed, args, n, nb, sp):
    rng = np.random.default_rng(seed)
    if sp is None:
        sp = 2 * nb
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


def build(role, n, nb, k, seed):
    args = _args(candidate_source_layer=(0 if role != "plain" else -1))
    mx.random.seed(len(str(nb)))
    idx = I.Indexer(args, 0)
    mx.eval(idx.parameters())
    idx.index_topk = k
    inp = inputs(seed, args, n, nb, None)
    return idx, inp


def full_row(idx, inp, mask=None):
    x, qr, sp, fv, ik = inp
    nb = ik.shape[1]; n = x.shape[1]
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2).astype(mx.float32)          # [1,n,nb] fp32
    vis = mx.arange(nb)[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    if mask is not None:
        row = mx.where(mask, row, float("-inf"))
    mx.eval(row)
    return np.array(row), float(lens.max())


def classify(role, n, nb, k, seed, cmask=None):
    idx, inp = build(role, n, nb, k, seed)
    # two paths through the real __call__
    sh_d = SharedState()
    if cmask is not None:
        sh_d.candidates = cmask
    I._FENCE_MIN_ROWS = 16
    out_d = idx(*inp[:2], inp[2], 0, inp[3], inp[4], sh_d); mx.eval(out_d)
    sh_h = SharedState()
    if cmask is not None:
        sh_h.candidates = cmask
    I._FENCE_MIN_ROWS = -1
    out_h = idx(*inp[:2], inp[2], 0, inp[3], inp[4], sh_h); mx.eval(out_h)
    od, oh = np.array(out_d), np.array(out_h)
    nd = int((od != oh).sum())
    if nd == 0:
        return dict(role=role, n=n, nb=nb, k=k, seed=seed, ndiff=0, verdict="EQUAL")
    R, _ = full_row(idx, inp, mask=cmask)
    # fallback (out_d) is the full-width path; its indices ARE in R's index space
    # (offset=0). Compare per-slot values.
    ties = 0; losses = 0; worst = 0.0
    for b in range(od.shape[0]):
        for r in range(od.shape[1]):
            for s in range(od.shape[2]):
                di, hi = int(od[b, r, s]), int(oh[b, r, s])
                vd = float(R[b, r, di]) if 0 <= di < nb else float("-inf")
                vh = float(R[b, r, hi]) if 0 <= hi < nb else float("-inf")
                if di != hi:
                    if vd == vh:
                        ties += 1
                    elif vh < vd:
                        losses += 1
                        worst = max(worst, vd - vh)
                    else:
                        # hier picked a HIGHER value: fallback lost one (should
                        # not happen for the full-width arm)
                        losses += 1
                        worst = max(worst, vh - vd)
    # tie-class size at the boundary: how many columns share the fallback kth value
    Rr = R.reshape(-1)
    finite = np.sort(Rr[np.isfinite(Rr)])[::-1]
    kth = float(finite[k - 1])
    tieclass = int((Rr == kth).sum())
    verdict = "TIE" if losses == 0 else "VALUE_LOSS"
    return dict(role=role, n=n, nb=nb, k=k, seed=seed, ndiff=nd,
                tie_slots=ties, loss_slots=losses, worst_loss=float(worst),
                kth_value=kth, tie_class_size=tieclass, verdict=verdict)


def block_mask(rng, b, n, nb, block, n_keep):
    NB = -(-nb // block)
    keep = np.zeros((b, n, NB), bool)
    for bi in range(b):
        for r in range(n):
            keep[bi, r, rng.choice(NB, size=min(n_keep, NB), replace=False)] = True
    return mx.array(np.repeat(keep, block, axis=-1)[..., :nb])


if __name__ == "__main__":
    print("=== CLASSIFY residual divergences (L2-full / ROW_BF16=0 regime) ===", flush=True)
    res = []
    # plain residual cells (from the suite: n>=2 at nb=16384, plus n=4/16 grid)
    plain = [(2, 16384, 511, 372016), (3, 16384, 511, 373012), (3, 16384, 512, 373014),
             (4, 16384, 511, 372000), (4, 4096, 511, 90000), (16, 16384, 512, 372002),
             (16, 4096, 512, 88000), (4, 512, 512, 71036)]
    for (n, nb, k, s) in plain:
        r = classify("plain", n, nb, k, s)
        res.append(r); print("  " + json.dumps(r), flush=True)
    # consumer residual cells
    for n in (4, 16):
        for nb in (512, 4096):
            for k in (64, 128):
                cm = block_mask(np.random.default_rng(4321), 1, n, nb, 8, max(1, nb // 16))
                r = classify("consumer", n, nb, k, 77000 + n * 211 + nb + k, cmask=cm)
                res.append(r); print("  " + json.dumps(r), flush=True)
    tot_tie = sum(r.get("tie_slots", 0) for r in res)
    tot_loss = sum(r.get("loss_slots", 0) for r in res)
    print(json.dumps({"CLASSIFY": {"cells": len(res),
                                   "cells_with_diff": sum(1 for r in res if r["ndiff"]),
                                   "total_tie_slots": tot_tie,
                                   "total_loss_slots": tot_loss,
                                   "verdict": "ALL TIES" if tot_loss == 0 else "VALUE LOSS PRESENT"}}),
          flush=True)
