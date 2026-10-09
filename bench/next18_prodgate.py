# Copyright © 2026 Adam Durham (hermes-gw)
"""Production-H gate: zero-column census + L2-full-vs-HIER stability at H>=8.

Confirms the mechanism (a column scores exactly 0.0 iff every head's pre-ReLU
dot is <= 0, P ~ 2^-H) by censusing the exact-zero fraction of the fp32 full-width
row against H, and broadens the L2-full-vs-HIER index-identity proof to the
PRODUCTION head count (DeepSeek-V4.1-Flash config: index_n_heads=64) over a wider
seed/nb grid than the fixture sweep.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_prodgate.py
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


def _args(h, **o):
    b = dict(dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
             q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
             compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
             index_n_heads=h, index_head_dim=32, index_topk=512, max_seq_len=4096,
             candidate_source_layer=-1, candidate_topk_blocks=4, candidate_block_size=8)
    b.update(o)
    return ModelArgs(**b)


def inputs(seed, args, n, nb, sp):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, n, args.dim)).astype(np.float32))
    qr = mx.array(rng.standard_normal((1, n, args.q_lora_rank)).astype(np.float32))
    ik = mx.array(rng.standard_normal((1, nb, args.index_head_dim)).astype(np.float32))
    fv = mx.array(np.linspace(1.0, 0.01, args.rope_head_dim // 2).astype(np.float32))
    return x, qr, sp, fv, ik


def row_fp32(idx, inp):
    x, qr, sp, fv, ik = inp
    nb = ik.shape[1]; n = x.shape[1]
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
    mx.eval(row)
    return np.array(row)


def run(idx, inp, fence, bf16):
    I._ROW_DTYPE = mx.bfloat16 if bf16 else mx.float32
    I._FENCE_MIN_ROWS = fence
    o = idx(inp[0], inp[1], inp[2], 0, inp[3], inp[4], SharedState())
    mx.eval(o)
    return np.array(o)


if __name__ == "__main__":
    print("=== zero-column census vs H (fp32 row; visible columns only) ===", flush=True)
    census = []
    for h in (2, 4, 8, 16, 32, 64):
        args = _args(h)
        mx.random.seed(7)
        idx = I.Indexer(args, 0); mx.eval(idx.parameters())
        idx.index_topk = 512
        inp = inputs(424242, args, 8, 4096, sp=2 * 4096)
        R = row_fp32(idx, inp)
        fin = R[np.isfinite(R)]
        zero_frac = float((fin == 0.0).mean())
        census.append(dict(H=h, visible=fin.size, zero_cols=int((fin == 0.0).sum()),
                           zero_frac=zero_frac, expected_2pow=2.0 ** -h))
        print("  " + json.dumps(census[-1]), flush=True)

    print("=== production-H stability: L2-full vs HIER, H in {8,32,64} ===", flush=True)
    stab = []
    for h in (8, 32, 64):
        tot_fb = tot_bf = tot_slots = 0
        for n in (1, 4, 16):
            for nb in (512, 4096, 16384):
                k = 512 if nb >= 512 else nb
                for seed in range(8):
                    args = _args(h)
                    mx.random.seed(len(str(nb)))
                    idx = I.Indexer(args, 0); mx.eval(idx.parameters())
                    idx.index_topk = k
                    inp = inputs(700000 + n * 131 + nb + seed, args, n, nb, sp=2 * nb)
                    F = run(idx, inp, 16, False)
                    B = run(idx, inp, 16, True)
                    Hh = run(idx, inp, -1, True)
                    tot_fb += int((F != Hh).sum())
                    tot_bf += int((B != Hh).sum())
                    tot_slots += F.size
        stab.append(dict(index_n_heads=h, slots=tot_slots,
                         L2full_vs_hier_diff=tot_fb, bf16row_vs_hier_diff=tot_bf))
        print("  " + json.dumps(stab[-1]), flush=True)

    # determinism replicate at H=64 (PREREG D4)
    print("=== determinism replicate at production H=64 ===", flush=True)
    det = []
    for n in (1, 4, 16):
        for nb in (4096, 16384):
            args = _args(64)
            mx.random.seed(len(str(nb)))
            idx = I.Indexer(args, 0); mx.eval(idx.parameters())
            idx.index_topk = 512
            inp = inputs(800000 + n + nb, args, n, nb, sp=2 * nb)
            F1 = run(idx, inp, 16, False); F2 = run(idx, inp, 16, False)
            H1 = run(idx, inp, -1, True); H2 = run(idx, inp, -1, True)
            det.append(dict(n=n, nb=nb, L2full_self=int((F1 != F2).sum()),
                            hier_self=int((H1 != H2).sum()),
                            L2full_vs_hier=int((F1 != H1).sum())))
            print("  " + json.dumps(det[-1]), flush=True)

    print(json.dumps({"PROD_GATE": {"census": census, "stability": stab,
                                    "determinism": det}}), flush=True)
