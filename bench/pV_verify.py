#!/usr/bin/env python3
"""pV -- stream-V verify-marginal probe (single node, layer subset, <=8 GB).

Measures, on ONE MoE layer and the layer's dense EXL3 projections:
  1. MoE expert cost vs R (rows in the verify window), realistic routing,
     plus the "perfect dedup" ceiling (every row routed to the same experts)
     and a "no dedup possible" floor (all experts distinct across rows).
  2. Dense EXL3 small-batch GEMM: ms vs rows 1..8 for every real projection
     of the layer, with the marginal ms/row.
Prints mx.get_peak_memory() at the end.

Run (on the Mac, under the GPU lock):
  EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
    ~/repos/exo/.venv/bin/python bench/pV_verify.py
Env: PV_PKG (sys.path prefix, default ~/dsv41-ws2/V), PV_LAYER (default 20),
     PV_REPS, PV_JSON (dump json), PV_MODE=moe|dense|all
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV_PKG", HOME + "/dsv41-ws2/V"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts, load_dense_layer
from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV_LAYER", "20"))
REPS = int(os.environ.get("PV_REPS", "15"))
MODE = os.environ.get("PV_MODE", "all")
OUT = os.environ.get("PV_JSON")


def log(*a):
    print("[pV]", *a, flush=True)


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


res = {"node": os.uname().nodename, "layer": LAYER}
ck = Exl3Checkpoint(CK)
rng = np.random.RandomState(11)

# ---------------------------------------------------------------- MoE
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
log(f"MoE layer {LAYER}: E={E} D={D} H={H} k={sg._k} gu_tiles={sg._gu_tiles} "
    f"dn_tiles={sg._dn_tiles} active={mx.get_active_memory()/1e9:.2f}GB")
res["moe_dim"] = {"E": E, "D": D, "H": H, "k": sg._k}


def moe_rows(R, uniq_frac):
    """(1,R,kk) indices with controlled expert overlap; uniq_frac in [0,1]:
    1.0 -> rows draw disjoint experts as much as possible, 0.0 -> all rows
    share the first row's experts."""
    kk = 7  # 6 routed + shared slot (exl3_build stacks the shared expert last)
    base = rng.permutation(E - 1)[:6]
    rows = []
    for r in range(R):
        if r == 0 or uniq_frac >= 1.0:
            e = rng.permutation(E - 1)[:6] if r else base
        else:
            n_new = int(round(uniq_frac * 6))
            keep = base[: 6 - n_new]
            fresh = rng.permutation(E - 1 - len(keep))
            fresh = fresh[~np.isin(fresh, keep)][:n_new]
            e = np.concatenate([keep, fresh])[:6]
        rows.append(np.concatenate([e, [E - 1]]))
    return mx.array(np.stack(rows)[None].astype(np.uint32))


xr = {R: mx.array(rng.randn(1, R, D).astype(np.float16)) for R in range(1, 9)}
res["moe"] = {}
for R in (1, 2, 3, 4, 5, 6, 8):
    idx = moe_rows(R, 1.0)
    mx.eval(idx, xr[R])
    n_uniq = len(np.unique(np.array(idx)))
    t = timeit(lambda R=R, idx=idx: sg(xr[R], idx)) * 1e3
    res["moe"][f"R{R}"] = {"ms": t, "uniq": n_uniq, "slots": R * 7}
    log(f"  MoE R={R}: {t:7.3f} ms  slots={R*7:3d} uniq={n_uniq:3d} "
        f"unique_ratio={n_uniq/(R*7):.2f}")
# ceiling: every row -> the same 6 experts (7 unique total)
res["moe_ceiling"] = {}
for R in (2, 4, 6):
    idx = moe_rows(R, 0.0)
    mx.eval(idx)
    n_uniq = len(np.unique(np.array(idx)))
    t = timeit(lambda R=R, idx=idx: sg(xr[R], idx)) * 1e3
    res["moe_ceiling"][f"R{R}"] = {"ms": t, "uniq": n_uniq}
    log(f"  MoE ceiling (1 unique set for all rows) R={R}: {t:7.3f} ms "
        f"uniq={n_uniq}  [R1={res['moe']['R1']['ms']:.3f}]")

# ------------------------------------------------------------- dense
if MODE in ("all", "dense"):
    PROJS = [
        ("attn.wq_a", None), ("attn.wkv", None), ("attn.compressor.wkv", None),
        ("attn.indexer.wq_b", None), ("attn.wq_b", "out"), ("attn.wo_b", "in"),
        ("ffn.shared_experts.w1", "out"), ("ffn.shared_experts.w2", "in"),
    ]
    res["dense"] = {}
    for nm, axis in PROJS:
        full = f"layers.{LAYER}.{nm}"
        if not ck.has(full + ".trellis"):
            continue
        lay = load_dense_layer(ck, full)
        if axis:  # rank-0 rank slice, same as the real build
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
        from mlx_lm.models.exl3.layer_state import layer_runtime_mlx
        rt = layer_runtime_mlx(lay)
        mx.eval(rt.trellis, rt.suh, rt.svh)
        ent = {}
        for R in (1, 2, 3, 4, 5, 6, 8):
            xx = mx.array(np.random.RandomState(R).randn(R, lay.in_features).astype(np.float16))
            mx.eval(xx)

            def f(xx=xx, rt=rt):
                return rt.finish_y(G.inner_gemm_mlx(rt.prepare_xh(xx), rt.trellis,
                                                    rt.k, rt.cb).astype(mx.float16))
            ent[f"R{R}"] = timeit(f) * 1e3
        ent["slope_1_4"] = (ent["R4"] - ent["R1"]) / 3
        ent["slope_4_8"] = (ent["R8"] - ent["R4"]) / 4
        res["dense"][nm] = ent
        log(f"  dense {nm:24s} in={lay.in_features:5d} out={lay.out_features:5d} " +
            " ".join(f"R{R}:{ent[f'R{R}']:6.3f}" for R in (1, 2, 3, 4, 5, 6, 8)) +
            f"  slope1-4={ent['slope_1_4']:.3f} slope4-8={ent['slope_4_8']:.3f} ms/row")

log(f"peak={mx.get_peak_memory()/1e9:.2f}GB active={mx.get_active_memory()/1e9:.2f}GB")
res["peak_GB"] = mx.get_peak_memory() / 1e9
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
    log("wrote", OUT)
log("PV_DONE")
