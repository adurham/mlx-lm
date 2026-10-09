# Copyright © 2026 Adam Durham (hermes-gw)
"""Head-count sensitivity of the residual divergence (production relevance).

The suite's synthetic fixtures use index_n_heads=2, which makes a large class of
columns score EXACTLY 0.0 (a column is 0 iff every head's pre-ReLU dot is <= 0,
P ~ 2^-H) -- a giant tie class at the k-boundary that no two implementations
break identically. Production uses index_n_heads=32 (P(exact 0) ~ 2^-32 ~ 0).
This probe re-runs the residual cell family across H in {2,4,8,16,32} and reports
fallback-fp32row (L2-full) vs HIER index diffs, to separate a CONFIG artifact
from a real design gap.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_headcount.py
"""
from __future__ import annotations
import json
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState


def _args(h, **o):
    b = dict(dim=64, n_layers=3, n_heads=4, head_dim=128, rope_head_dim=16,
             q_lora_rank=32, o_lora_rank=32, o_groups=4, window_size=8,
             compress_ratios=(2, 2, 2), kv_source_layers=(0,), index_source_layers=(0,),
             index_n_heads=h, index_head_dim=32, index_topk=512, max_seq_len=4096,
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
    print("=== head-count sensitivity: L2-full (fp32row) vs HIER index diffs ===", flush=True)
    rows = []
    for h in (2, 4, 8, 16, 32):
        tot_fb = tot_bf = tot_slots = cells = 0
        for n in (1, 4, 16):
            for nb in (512, 4096, 16384):
                k = 512 if nb >= 512 else nb
                for seed in range(6):
                    args = _args(h)
                    mx.random.seed(len(str(nb)))
                    idx = I.Indexer(args, 0)
                    mx.eval(idx.parameters())
                    idx.index_topk = k
                    inp = inputs(600000 + n * 97 + nb + seed, args, n, nb, sp=2 * nb)
                    F = run(idx, inp, 16, False)
                    B = run(idx, inp, 16, True)
                    Hh = run(idx, inp, -1, True)
                    tot_fb += int((F != Hh).sum())
                    tot_bf += int((B != Hh).sum())
                    tot_slots += F.size
                    cells += 1
        rows.append(dict(index_n_heads=h, cells=cells, slots=tot_slots,
                         L2full_vs_hier_diff=tot_fb, bf16row_vs_hier_diff=tot_bf))
        print("  " + json.dumps(rows[-1]), flush=True)
    print(json.dumps({"HEADCOUNT": rows}), flush=True)
