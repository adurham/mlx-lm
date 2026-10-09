# Copyright © 2026 Adam Durham (hermes-gw)
"""Reproduce specific divergent suite cells (ROW_BF16=0 residual) and classify.

For each divergent cell, re-run BOTH paths through the real Indexer.__call__
(fallback = _FENCE_MIN_ROWS=16; HIER = _FENCE_MIN_ROWS=-1) and, on every differing
output slot, print the underlying fp32 score value each path selected, so we can
tell a VALUE LOSS from an exact/near TIE.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_repro.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
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


def inputs(seed, args, n, nb, sp):
    rng = np.random.default_rng(seed)
    if sp is None:
        sp = 2 * nb
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


def repro(role, n, nb, k, s, bf16):
    args = _args(candidate_source_layer=(0 if role != "plain" else -1))
    mx.random.seed(len(str(nb)))
    idx = I.Indexer(args, 0)
    mx.eval(idx.parameters())
    idx.index_topk = k
    x, qr, sp, fv, ik = inputs(s, args, n, nb, None)
    I._ROW_DTYPE = mx.bfloat16 if bf16 else mx.float32

    I._FENCE_MIN_ROWS = 16
    out_d = idx(x, qr, sp, 0, fv, ik, SharedState()); mx.eval(out_d)
    I._FENCE_MIN_ROWS = -1
    out_h = idx(x, qr, sp, 0, fv, ik, SharedState()); mx.eval(out_h)
    od, oh = np.array(out_d), np.array(out_h)
    nd = int((od != oh).sum())
    if nd == 0:
        return dict(role=role, n=n, nb=nb, k=k, s=s, ndiff=0)

    # reference full fp32 row in index space, and per-slot score values
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
    from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2).astype(mx.float32)
    vis = mx.arange(nb)[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    mx.eval(row)
    R = np.array(row)                    # [1,n,nb] fp32
    off = 0                              # offset passed above is 0
    # differing positions
    pos = np.argwhere(od != oh)
    detail = []
    for (b, r, slot) in pos[:6]:
        di, hi = int(od[b, r, slot]), int(oh[b, r, slot])
        vd = float(R[b, r, di - off]) if 0 <= di - off < nb else None
        vh = float(R[b, r, hi - off]) if 0 <= hi - off < nb else None
        detail.append(dict(slot=int(slot), fb_idx=di, hier_idx=hi,
                           fb_val=vd, hier_val=vh,
                           equal=bool(vd is not None and vh is not None and vd == vh)))
    # also: does HIER's returned set (as values) equal the top-k of R on the multiset?
    hiv = np.sort([float(R[0, r, int(oh[0, r, sl]) - off]) if 0 <= int(oh[0, r, sl]) - off < nb
                   else float("-inf") for r in range(n) for sl in range(oh.shape[-1])])[::-1]
    refv = np.sort(R.reshape(-1))[::-1][:len(hiv)]
    vloss = int((refv - hiv > 1e-5 * max(1.0, float(np.abs(refv).max()))).sum())
    return dict(role=role, n=n, nb=nb, k=k, s=s, ndiff=nd, value_loss=vloss,
                detail=detail)


if __name__ == "__main__":
    cells = [(2, 16384, 511, 372016), (3, 16384, 511, 373012), (3, 16384, 512, 373014),
             (4, 16384, 511, 372000), (16, 16384, 512, 372002)]
    print("=== ROW_BF16=0 residual reproduction (plain) ===", flush=True)
    for (n, nb, k, s) in cells:
        r = repro("plain", n, nb, k, s, bf16=False)
        print("  " + json.dumps(r), flush=True)
