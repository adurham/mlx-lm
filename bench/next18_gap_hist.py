# Copyright © 2026 Adam Durham (hermes-gw)
"""Trigger-rate / kth-boundary-gap histogram for the (unnecessary) L2-guard.

The pre-registered selection rule prefers L2-full if it is within ~2-3 ms of
L2-guard. Measured per-call delta (fp32 row minus bf16 row) is <=0.063 ms, so
L2-full is preferred and L2-guard is NOT built. This probe still reports the
numbers the guard WOULD have run on, as supporting evidence:

  * the kth-vs-(k+1)th gap of the fp32 full-width row at small n, vs the derived
    per-call bound delta = 2^-8 * max|row| (the bf16 storage rounding of the fp32
    score row);
  * the resulting escape rate (fraction of calls where gap <= 2*delta, i.e. the
    guard would fall back to full-width), i.e. the guard's trigger rate;
  * for contrast, the same gap on the bf16 STORED row (the shipped fallback).

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_gap_hist.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
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


def main():
    args = _args()
    print("=== kth-boundary gap histogram, small n (L2-guard trigger rate) ===", flush=True)
    rows = []
    for n in (1, 4, 16):
        for nb in (512, 4096, 16384):
            k = 512 if nb >= 512 else nb
            for seed in range(6):
                mx.random.seed(len(str(nb)))
                idx = I.Indexer(args, 0)
                mx.eval(idx.parameters())
                rng = np.random.default_rng(600000 + n * 97 + nb + seed)
                x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
                qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
                ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
                fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
                sp = 2 * nb
                q = idx.wq_b(qr).reshape(1, n, idx.n_heads, idx.head_dim)
                q = rope_tail(q, idx.rope_head_dim, *cos_sin_at(fv, mx.arange(sp, sp + n)))
                q = fake_quant_fp4_ue8m0(q, 32)
                w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
                lens = ((sp + mx.arange(n) + 1) // idx.ratio)[:, None]
                sc = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
                sc = mx.maximum(sc, 0.0) * w[..., None].astype(mx.float32)
                row = mx.sum(sc, axis=2).astype(mx.float32)        # [1,n,nb]
                vis = mx.arange(nb)[None, :] < lens
                row = mx.where(vis[None], row, float("-inf"))
                mx.eval(row)
                R = np.array(row)
                # per row kth / (k+1)th finite
                esc = 0; tot = 0; deltas = []
                for r in range(n):
                    fin = np.sort(R[0, r][np.isfinite(R[0, r])])[::-1]
                    if fin.size < k + 1:
                        continue
                    gap = float(fin[k - 1] - fin[k])
                    delta = (2.0 ** -8) * float(np.abs(R[0, r][np.isfinite(R[0, r])]).max())
                    deltas.append(delta)
                    tot += 1
                    if gap <= 2.0 * delta:
                        esc += 1
                rows.append(dict(n=n, nb=nb, k=k, seed=seed, rows=tot,
                                 escape=esc, escape_rate=(esc / tot if tot else 0.0),
                                 delta_lo=min(deltas) if deltas else 0.0,
                                 delta_hi=max(deltas) if deltas else 0.0))
    for r in rows:
        print("  " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in r.items()}), flush=True)
    tot_rows = sum(r["rows"] for r in rows)
    tot_esc = sum(r["escape"] for r in rows)
    print(json.dumps({"GAP_HIST": {"rows": tot_rows, "escape": tot_esc,
                                   "trigger_rate": (tot_esc / tot_rows if tot_rows else 0.0),
                                   "bound": "delta = 2^-8 * max|row|"}}), flush=True)


if __name__ == "__main__":
    main()
