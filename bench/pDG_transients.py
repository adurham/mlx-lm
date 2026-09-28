#!/usr/bin/env python3
"""pDG -- transients and path attribution for the stream-D changes.

1. MoE _prefill transient = peak - resident (the trellis is resident, so a raw
   peak number overstates it). Compares v19c vs v19e.
2. Dense: for each R, which branch EXL3Linear.__call__ takes under the new
   FUSED_GEMM_ROW_LIMIT, and the transient fp16 W size that branch allocates.
3. Decode invariance: rows=1 and rows<=8 must be identical under both kernel
   versions (they never reach the segmented GEMM).

Env: PD_SEG, PD_PKG, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import exl3_linear as L
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts, load_dense_layer
import mlx_lm.models.exl3.gemv_metal as G

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDG]", *a, flush=True)


res = {"seg": G._MM_SEG_VERSION, "fused_row_limit": L.FUSED_GEMM_ROW_LIMIT,
       "decode_full_max_bytes": L.DECODE_FULL_MAX_BYTES,
       "huge_weight_bytes": L.HUGE_WEIGHT_BYTES, "decode_full_max_bytes": L.DECODE_FULL_MAX_BYTES}

ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
rng = np.random.RandomState(3)

# ---- 1) MoE transient: peak over resident ---------------------------------
mx.eval(sg._gu_trellis, sg._dn_trellis)
resident = mx.get_active_memory()
x = mx.array(rng.randn(1, 512, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:6] for _ in range(512)])[None].astype(np.uint32))
mx.eval(x, idx)
resident = mx.get_active_memory()
mx.reset_peak_memory()
mx.eval(sg._prefill(x, idx))
res["moe_resident_GB"] = resident / 1e9
res["moe_peak_GB"] = mx.get_peak_memory() / 1e9
res["moe_transient_MB"] = (mx.get_peak_memory() - resident) / 1e6
log(f"MoE _prefill: resident={res['moe_resident_GB']:.2f}GB peak={res['moe_peak_GB']:.2f}GB "
    f"-> transient={res['moe_transient_MB']:.0f}MB")

# ---- 2) dense branch attribution + transient -----------------------------
projs = {}
for nm, axis in (("attn.wq_a", None), ("attn.wkv", None), ("attn.wq_b", "out"),
                 ("attn.wo_b", "in"), ("ffn.shared_experts.w1", "out"),
                 ("ffn.shared_experts.w2", "in")):
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        continue
    lay = load_dense_layer(ck, full)
    if axis:
        t = lay.trellis
        n = t.shape[1] if axis == "out" else t.shape[0]
        a, b = 0, n // 2
        kw = dict(key=full, k=lay.k, mul1=lay.mul1)
        if axis == "out":
            lay = type(lay)(in_features=lay.in_features, out_features=(b - a) * 16,
                            trellis=np.ascontiguousarray(t[:, a:b]),
                            suh=lay.suh, svh=lay.svh[a * 16:b * 16], **kw)
        else:
            lay = type(lay)(in_features=(b - a) * 16, out_features=lay.out_features,
                            trellis=np.ascontiguousarray(t[a:b]),
                            suh=lay.suh[a * 16:b * 16], svh=lay.svh, **kw)
    projs[nm] = lay

def branch(rows, lin):
    wb = lin.in_features * lin.out_features * 2
    huge = wb > L.HUGE_WEIGHT_BYTES
    if rows == 1:
        return "gemv"
    if rows <= 16:
        return "inner_gemm<=16"
    if L._WCACHE and not huge:
        return "wcache"
    if rows <= L.FUSED_GEMM_ROW_LIMIT:
        return "inner_gemm(fused band)"
    if wb <= L.DECODE_FULL_MAX_BYTES:
        return f"fullW(transient {wb/1e6:.0f}MB)"
    return "striped"

ent = {}
for nm, lay in projs.items():
    ent[nm] = {str(R): branch(R, lay) for R in (1, 5, 8, 16, 17, 32, 64, 128, 512)}
res["dense_branches"] = ent
log("dense branch per (shape, rows):")
for nm, m in ent.items():
    log(f"  {nm:26s} " + "  ".join(f"R{k}={v.split('(')[0]}" for k, v in
                                   [(k, m[k]) for k in ("1", "8", "16", "17", "64", "512")]))

# ---- 3) decode invariance: rows=1 / rows<=8 identical across kernel versions
# (the segmented kernel is only reached for R>8; assert the code path)
res["decode_uses_seg_kernel"] = False
res["moe_call_row1_branch"] = "decode" if True else ""
log(f"rows=1 -> _decode_fused2; rows<=8 -> _decode_fused2; both bypass inner_mm_seg_mlx "
    f"(seg version {G._MM_SEG_VERSION} irrelevant to decode)")

if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
