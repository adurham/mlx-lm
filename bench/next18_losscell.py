# Copyright © 2026 Adam Durham (hermes-gw)
"""Focused repro of the one VALUE_LOSS cell (plain n=16 nb=16384 k=513 seed=385721).

Determines (a) is it deterministic, (b) which arm loses values, (c) does it also
occur in the SHIPPED bf16 config (i.e. is it a pre-existing HIER property, not an
L2-full artefact), (d) how does it behave vs the pure fp32 TRUTH
(hierarchical_topk_prod overfetch=16 vs full-width fp32 row).

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_losscell.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
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


def row_fp32(idx, inp):
    x, qr, sp, fv, ik = inp
    nb = ik.shape[1]; n = x.shape[1]
    if sp is None:
        sp = 2 * nb
    q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
    q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
    lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
    sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
    row = mx.sum(sc, axis=2).astype(mx.float32)
    vis = mx.arange(nb)[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    mx.eval(row, q, ik, w, lens)
    return np.array(row), q, w, lens


def run(idx, inp, fence, bf16):
    I._ROW_DTYPE = mx.bfloat16 if bf16 else mx.float32
    I._FENCE_MIN_ROWS = fence
    out = idx(inp[0], inp[1], inp[2], 0, inp[3], inp[4], SharedState())
    mx.eval(out)
    return np.array(out)


def arm_loss(R, idx_out, nb):
    """Return (#slots where this arm's value < R's value at that slot's index,
    worst gap). Uses R for the score at that index (bitwise the fallback row)."""
    loss = 0; worst = 0.0
    for b in range(idx_out.shape[0]):
        for r in range(idx_out.shape[1]):
            for s in range(idx_out.shape[2]):
                j = int(idx_out[b, r, s])
                if 0 <= j < nb and j != -1:
                    v = float(R[b, r, j])
                else:
                    v = float("-inf")
                # truth value at this rank = k-th largest of R (rank s, but top-k
                # sorted ascending by index, so rank alignment is per-slot)
                pass
    return loss, worst


if __name__ == "__main__":
    n, nb, k, seed = 16, 16384, 513, 385721
    args = _args()
    mx.random.seed(len(str(nb)))
    idx = I.Indexer(args, 0)
    mx.eval(idx.parameters())
    idx.index_topk = k
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    inp = (x, qr, 2 * nb, fv, ik)

    R, q, w, lens = row_fp32(idx, inp)
    # truth per query row (rows are independent)
    def truth_row(r):
        return np.sort(R[0, r])[::-1][:k]

    out_fb_f32 = run(idx, inp, 16, bf16=False)
    out_fb_bf16 = run(idx, inp, 16, bf16=True)
    out_hier = run(idx, inp, -1, bf16=True)   # HIER (row dtype irrelevant there)

    # per-row value multiset of each arm's returned slots (via R)
    def vals(o):
        vs = np.empty((o.shape[1], k), np.float64)
        for r in range(o.shape[1]):
            row = []
            for s in range(o.shape[2]):
                j = int(o[0, r, s])
                row.append(float(R[0, r, j]) if 0 <= j < nb else float("-inf"))
            vs[r] = np.sort(np.array(row))[::-1]
        return vs

    TR = np.stack([truth_row(r) for r in range(n)])
    scale = max(1.0, float(np.abs(TR).max()))
    for name, o in (("fallback_fp32row(L2-full)", out_fb_f32),
                    ("fallback_bf16row(shipped)", out_fb_bf16),
                    ("hier", out_hier)):
        v = vals(o)
        loss = int((TR - v > 1e-6 * scale).sum())
        worst = float((TR - v).max())
        print(json.dumps({"arm": name, "value_slots_below_truth": loss,
                          "worst_gap_vs_truth": round(worst, 6)}), flush=True)

    # determinism
    a = run(idx, inp, 16, bf16=False)
    b = run(idx, inp, 16, bf16=False)
    c = run(idx, inp, -1, bf16=True)
    d = run(idx, inp, -1, bf16=True)
    print(json.dumps({"determinism": {
        "fallback_fp32_repeat_equal": bool(np.array_equal(a, b)),
        "hier_repeat_equal": bool(np.array_equal(c, d)),
        "fb_vs_hier_ndiff": int((a != c).sum())}}), flush=True)

    # direct: does HIER pick a strictly lower value than the fallback (per row)?
    fbv = vals(out_fb_f32); hv = vals(out_hier)
    print(json.dumps({"hier_vs_fallback_fp32row": {
        "slots_hier_below": int((fbv - hv > 1e-6 * scale).sum()),
        "worst": float((fbv - hv).max())}}), flush=True)

    # The exact HIER-vs-TRUTH at the function level, overfetch=16, many n=16 seeds
    print("=== HIER (of=16) vs fp32 TRUTH value-loss sweep over seeds, n=16 ===", flush=True)
    tot = 0; worst = 0.0; bad = 0
    for sd in range(40):
        r2 = np.random.default_rng(900000 + sd)
        ik2 = mx.array(r2.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
        inp2 = (x, qr, 2 * nb, fv, ik2)
        R2, qq, ww, ll = row_fp32(idx, inp2)
        TR2 = np.stack([np.sort(R2[0, r])[::-1][:k] for r in range(n)])
        hv2, hi2, _ = H.hierarchical_topk_prod(qq, ik2, ww, ll, k, I._HIER_BLOCK,
                                               I._HIER_STRIP, 4096, 16)
        mx.eval(hv2, hi2)
        v2 = np.empty((n, k), np.float64)
        hh = np.array(hi2)
        for r in range(n):
            row = []
            for s in range(k):
                j = int(hh[0, r, s])
                row.append(float(R2[0, r, j]) if 0 <= j < nb else float("-inf"))
            v2[r] = np.sort(np.array(row))[::-1]
        s2 = max(1.0, float(np.abs(TR2).max()))
        loss = int((TR2 - v2 > 1e-6 * s2).sum())
        if loss:
            bad += 1; tot += loss; worst = max(worst, float((TR2 - v2).max()))
    print(json.dumps({"hier_vs_truth_sweep_n16": {"seeds": 40, "seeds_with_loss": bad,
                                                   "total_loss_slots": tot,
                                                   "worst_gap": round(worst, 6)}}), flush=True)
