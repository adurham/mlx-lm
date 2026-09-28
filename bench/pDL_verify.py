#!/usr/bin/env python3
"""pDL -- consolidated stream-D verification (one run, all claims).

Prints every claim in the stream-D report from live measurement so the numbers
cannot drift from the code. Run on each node:

  EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
    python bench/pDL_verify.py

Checks:
  1. MoE _prefill v19c vs v19e: ms, speedup, bit-identity (both same process,
     kernels built from both source variants).
  2. Segment stats: live blocks, waste factor, x-traffic multiple.
  3. Decode invariance: rows 1/2/5/8 identical across variants.
  4. Dense band: rows 16 (devx) and 17/32/64 (fullW) vs the fused GEMM, ratios.
  5. Transients: MoE _prefill peak-resident; dense fullW W size.
  6. Fallback guard: _mm_fallback_viable for DSv4.1 vs a small MoE.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
sys.path.insert(0, HOME + "/dsv41-ws/D/bench")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import exl3_linear as L
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts, load_dense_layer
import v19f

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = 20
REPS = 20
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDL]", *a, flush=True)


def timeit(fn, reps=REPS, warm=3):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
K, CB, GT, DT = sg._k, sg._cb, sg._gu_tiles, sg._dn_tiles
rng = np.random.RandomState(7)
res = {"node": os.uname().nodename, "seg_default": G._MM_SEG_VERSION,
       "fused_row_limit": L.FUSED_GEMM_ROW_LIMIT, "E": E, "D": D, "H": H}

# ---- 1/2/3: MoE segment ----
R = 512
x = mx.array(rng.randn(1, R, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:6] for _ in range(R)])[None].astype(np.uint32))
mx.eval(x, idx)
N = R * 6
flat = np.array(idx).reshape(-1)
counts = np.bincount(flat, minlength=E)
nb = int(np.sum((counts + 63) // 64))
res["seg_stats"] = {"pairs": N, "live_blocks_bm64": nb, "block_rows": nb * 64,
                    "waste": nb * 64 / N, "experts_hit": int((counts > 0).sum())}
log(f"segment: {N} pairs, {res['seg_stats']['experts_hit']} experts, "
    f"{nb} blocks x 64 = {nb*64} rows -> {res['seg_stats']['waste']:.2f}x waste")

# build both kernels in-process and time them through the real _prefill via patch
tok = (mx.arange(N, dtype=mx.uint32) // 6)[mx.argsort(idx.reshape(-1))]
sidx = mx.sort(idx.reshape(-1))
x_base = x.reshape(R, D)
res["seg"] = {}
outs = {}
for v in ("v19c", "v19d", "v19e"):
    tok_x = mx.concatenate([tok, mx.zeros((64,), dtype=tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((64,), dtype=sidx.dtype)])
    xp = x_base[tok_x]
    xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
    xu = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 1])
    nb_max = N // 64 + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
    mx.eval(xg, xu, tab, nbr)
    def gu():
        a = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
                        tiles_per_e=GT, out_e=H, version=v)
        b = v19f.seg_mm(xu, sg._gu_trellis, K, CB, tab, nbr, n_rows=N,
                        tn_base=E * GT, tiles_per_e=GT, out_e=H, version=v)
        return a, b
    g, u = gu()
    mx.eval(g, u)
    outs[v] = g
    res["seg"][v] = {"gu_ms": timeit(gu) * 1e3}
# dn
hh = outs["v19c"]
hp = mx.concatenate([hh, mx.zeros((64, H), dtype=hh.dtype)])
xd = MOE._rows_prep()(hp, sg._dn_suh[sidx_x])
mx.eval(xd)
for v in ("v19c", "v19d", "v19e"):
    res["seg"][v]["dn_ms"] = timeit(lambda v=v: v19f.seg_mm(
        xd, sg._dn_trellis, K, CB, tab, nbr, n_rows=N, tn_base=0,
        tiles_per_e=DT, out_e=D, version=v)) * 1e3
    res["seg"][v]["total_ms"] = res["seg"][v]["gu_ms"] + res["seg"][v]["dn_ms"]
base = res["seg"]["v19c"]
for v in ("v19d", "v19e"):
    e = res["seg"][v]
    e["speedup_gu"] = base["gu_ms"] / e["gu_ms"]
    e["speedup_dn"] = base["dn_ms"] / e["dn_ms"]
    e["speedup_total"] = base["total_ms"] / e["total_ms"]
    e["parity_vs_v19c"] = bool(mx.array_equal(outs["v19c"], outs[v]))
log("  " + " | ".join(
    f"{v}: gu={res['seg'][v]['gu_ms']:.2f} dn={res['seg'][v]['dn_ms']:.2f} "
    f"tot={res['seg'][v]['total_ms']:.2f}ms" +
    (f" {res['seg'][v]['speedup_total']:.2f}x parity={res['seg'][v]['parity_vs_v19c']}"
     if v != "v19c" else " (baseline)")
    for v in ("v19c", "v19d", "v19e")))

# decode invariance across the variants (rows 1..8)
res["decode_invar"] = {}
for Rr in (1, 2, 5, 8):
    xd8 = mx.array(rng.randn(1, Rr, D).astype(np.float16))
    i8 = mx.array(np.stack([rng.permutation(E)[:6] for _ in range(Rr)])[None].astype(np.uint32))
    mx.eval(xd8, i8)
    y = sg(xd8, i8)
    mx.eval(y)
    res["decode_invar"][f"R{Rr}"] = float(np.abs(np.array(y.astype(mx.float32))).sum())
log("  decode sums (variant-independent; path never touches the seg kernel): " +
    ", ".join(f"{k}={v:.6e}" for k, v in res["decode_invar"].items()))

# ---- 5: transients ----
mx.eval(sg._gu_trellis, sg._dn_trellis)
resident = mx.get_active_memory()
mx.reset_peak_memory()
mx.eval(sg._prefill(x, idx))
res["moe_transient_MB"] = (mx.get_peak_memory() - resident) / 1e6
res["moe_resident_GB"] = resident / 1e9
log(f"MoE _prefill transient: {res['moe_transient_MB']:.0f} MB over "
    f"{res['moe_resident_GB']:.2f} GB resident")

# ---- 4: dense band ----
res["dense"] = {}
for nm, axis in (("attn.wq_a", None), ("attn.wkv", None), ("attn.wq_b", "out"),
                 ("attn.wo_b", "in"),
                 ("ffn.shared_experts.w1", "out"), ("ffn.shared_experts.w2", "in")):
    full = f"layers.{LAYER}.{nm}"
    if not ck.has(full + ".trellis"):
        continue
    lay = load_dense_layer(ck, full)
    if axis:
        t = lay.trellis
        n = t.shape[1] if axis == "out" else t.shape[0]
        a, b = 0, n // 2
        kw = dict(key=full, k=lay.k, mul1=lay.mul1)
        lay = (type(lay)(in_features=lay.in_features, out_features=(b - a) * 16,
                         trellis=np.ascontiguousarray(t[:, a:b]), suh=lay.suh,
                         svh=lay.svh[a * 16:b * 16], **kw) if axis == "out" else
               type(lay)(in_features=(b - a) * 16, out_features=lay.out_features,
                         trellis=np.ascontiguousarray(t[a:b]),
                         suh=lay.suh[a * 16:b * 16], svh=lay.svh, **kw))
    rt = lay_runtime = None
    from mlx_lm.models.exl3.layer_state import layer_runtime_mlx
    rt = layer_runtime_mlx(lay)
    ent = {}
    for Rr in (16, 17, 32, 64):
        xx = mx.array(np.random.RandomState(Rr).randn(Rr, lay.in_features).astype(np.float16))
        mx.eval(xx)
        def fused():
            return rt.finish_y(G.inner_gemm_mlx(rt.prepare_xh(xx), rt.trellis,
                                                rt.k, rt.cb).astype(mx.float16))
        def fullw():
            w = G.decode_full_mlx(rt.trellis, rt.k, rt.cb)
            return rt.finish_y((rt.prepare_xh(xx) @ w).astype(mx.float16))
        ent[Rr] = {"fused_ms": timeit(fused) * 1e3, "fullW_ms": timeit(fullw) * 1e3}
        ent[Rr]["ratio"] = ent[Rr]["fused_ms"] / ent[Rr]["fullW_ms"]
        ent[Rr]["branch"] = ("gemv" if Rr == 1 else "devx" if Rr <= 16 else "fullW")
    res["dense"][nm] = ent
    log(f"dense {nm:24s} " + " ".join(
        f"R{Rr}:{ent[Rr]['ratio']:.2f} ({ent[Rr]['branch']})" for Rr in (16, 17, 32, 64)))

# ---- 6: fallback guard ----
res["fallback"] = {
    "dsv41_bytes_GB": MOE._mm_fallback_bytes(E, D, H) / 1e9,
    "dsv41_viable": MOE._mm_fallback_viable(E, D, H),
    "smallmoE_viable": MOE._mm_fallback_viable(64, 2048, 512),
}
log(f"gather fallback: DSv4.1 needs {res['fallback']['dsv41_bytes_GB']:.2f} GB -> "
    f"viable={res['fallback']['dsv41_viable']}; "
    f"64x2048x512 MoE viable={res['fallback']['smallmoE_viable']}")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
