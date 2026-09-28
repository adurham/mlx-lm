#!/usr/bin/env python3
"""pDI -- decode-path invariance under the stream-D changes.

The brief requires rows<=8 (decode + spec verify) to stay bit-identical. Three
invariance assertions, all at real DSv4.1 geometry:

 1. EXL3SwitchGLU rows=1 and rows<=8 ride _decode_fused2, which never touches
    the segmented GEMM -> outputs must be identical across EXL3_MM_SEG.
 2. EXL3Linear rows<=16 ride the devx inner_gemm, never the fused band or
    fullW -> outputs identical across EXL3_FUSED_ROW_LIMIT.
 3. rows=17..64 DO change (documented): both are valid EXL3 evaluations.

Saves npy per (case, setting) so a second process compares bit-exactly.

Env: PD_KIND=seg|limit, PD_VAL, PD_JSON.
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

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bwp".replace(
    "2.9bwp", "2.9bpw")
LAYER = int(os.environ.get("PD_LAYER", "20"))
KIND = os.environ.get("PD_KIND", "seg")
VAL = os.environ.get("PD_VAL", G._MM_SEG_VERSION if KIND == "seg"
                     else str(L.FUSED_GEMM_ROW_LIMIT))
TAG = os.environ.get("PD_TAG", f"{KIND}_{VAL}")
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDI]", *a, flush=True)


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
rng = np.random.RandomState(23)
res = {"kind": KIND, "val": VAL, "seg": G._MM_SEG_VERSION,
       "fused_row_limit": L.FUSED_GEMM_ROW_LIMIT, "cases": {}}

# --- 1) MoE decode rows 1..8 -------------------------------------------------
for R in (1, 2, 5, 8):
    x = mx.array(rng.randn(1, R, D).astype(np.float16))
    idx = mx.array(np.stack([rng.permutation(E)[:6] for _ in range(R)])[None].astype(np.uint32))
    mx.eval(x, idx)
    y = sg(x, idx)
    mx.eval(y)
    key = f"moe_R{R}"
    np.save(os.path.join(HOME, f"dsv41-ws/D/pDI_{key}_{TAG}.npy"),
            np.array(y.astype(mx.float32)))
    res["cases"][key] = float(np.abs(np.array(y.astype(mx.float32))).sum())
    log(f"MoE R={R} sum={res['cases'][key]:.8e}")

# --- 2) dense rows <= 16 -----------------------------------------------------
for nm, axis in (("attn.wq_a", None), ("attn.wq_b", "out"), ("attn.wo_b", "in"),
                 ("ffn.shared_experts.w1", "out")):
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
    for R in (1, 5, 8, 16, 17, 32, 64):
        x = mx.array(np.random.RandomState(R).randn(R, lay.in_features).astype(np.float16))
        mx.eval(x)
        y = L.EXL3Linear(lay)(x)
        mx.eval(y)
        key = f"dense_{nm.replace('.', '_')}_R{R}"
        np.save(os.path.join(HOME, f"dsv41-ws/D/pDI_{key}_{TAG}.npy"),
                np.array(y.astype(mx.float32)))
        res["cases"][key] = float(np.abs(np.array(y.astype(mx.float32))).sum())
    log(f"dense {nm:24s} rows 1..64 recorded")

if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
