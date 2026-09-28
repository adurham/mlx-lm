#!/usr/bin/env python3
"""pD3 -- traceback probe for the _prefill overflow at DSv4.1 MoE geometry."""
import os, sys, traceback
HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
ROWS = int(os.environ.get("PD_ROWS", "512"))
KK = int(os.environ.get("PD_KK", "6"))
NE = os.environ.get("PD_EXPERTS")
NE = int(NE) if NE else None

ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, n_experts=NE, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
print(f"[pD3] E={E} D={D} H={H} gu_tiles={sg._gu_tiles} dn_tiles={sg._dn_tiles} "
      f"gu={tuple(sg._gu_trellis.shape)} dn={tuple(sg._dn_trellis.shape)}", flush=True)

rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)

N = ROWS * KK
print(f"[pD3] N={N} pairs, nb_max={N//64 + E + 1}", flush=True)

# step through _prefill manually to find the overflow
D_, H_ = sg.input_dims, sg.hidden_dims
flat = idx.reshape(-1)
order = mx.argsort(flat)
inv = mx.argsort(order)
sidx = flat[order]
mx.eval(sidx)
idxr = sidx.reshape(N, 1).astype(mx.uint32)
tok = (mx.arange(N, dtype=mx.uint32) // KK)[order]
mx.eval(idxr, tok)
tok_x = mx.concatenate([tok, mx.zeros((64,), dtype=tok.dtype)])
sidx_x = mx.concatenate([sidx, mx.zeros((64,), dtype=sidx.dtype)])
x_pairs = x.reshape(ROWS, D_)[tok_x]
xh_g = MOE._rows_prep()(x_pairs, sg._gu_suh[sidx_x, 0])
mx.eval(xh_g)
print(f"[pD3] xh_g {tuple(xh_g.shape)} ok", flush=True)

# the table
nb_max = N // 64 + E + 1
tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
mx.eval(tab, nbr)
print(f"[pD3] tab {tuple(tab.shape)} nbr={np.array(nbr)} ok", flush=True)

# the kernel
from mlx_lm.models.exl3.gemv_metal import inner_mm_seg_mlx
gt = sg._gu_tiles
for label, args in (
    ("gu", dict(tn_base=0, tiles_per_e=gt, out_e=H_, trellis=sg._gu_trellis, x=xh_g)),
    ("dn", dict(tn_base=0, tiles_per_e=sg._dn_tiles, out_e=D_, trellis=sg._dn_trellis,
                x=MOE._rows_prep()(mx.concatenate([mx.zeros((N, H_)) , mx.zeros((64, H_))]), sg._dn_suh[sidx_x]))),
):
    try:
        y = inner_mm_seg_mlx(args["x"], args["trellis"], sg._k, sg._cb, tab, nbr,
                             n_rows=N, tn_base=args["tn_base"],
                             tiles_per_e=args["tiles_per_e"], out_e=args["out_e"])
        mx.eval(y)
        print(f"[pD3] {label} kernel OK {tuple(y.shape)}", flush=True)
    except Exception as e:
        print(f"[pD3] {label} FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        break
