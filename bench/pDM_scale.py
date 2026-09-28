#!/usr/bin/env python3
"""pDM -- guard against a large-N regression: v19c vs v19d vs v19e across N.

v19d/v19e only remove work when blk_len < 64 (tiny segments). At large N the
segments fill their blocks (rlive=4 everywhere) so the guard is a no-op and
only the A-load hoist (v19e) differs. Verify no regression from N=512 to
N=24576 (R=4096), i.e. the regime a long-prompt prefill actually runs in.

Env: PD_NS, PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
sys.path.insert(0, HOME + "/dsv41-ws/D/bench")
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
import v19f

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
KK = 6
REPS = int(os.environ.get("PD_REPS", "8"))
NS = [int(x) for x in os.environ.get("PD_NS", "512,1536,3072,9216,18432,24576").split(",")]
OUT = os.environ.get("PD_JSON")


def rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 if sys.platform == "darwin" \
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def log(*a):
    print("[pDM]", *a, flush=True)


def timeit(fn, reps=REPS, warm=2):
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
K, CB, GT = sg._k, sg._cb, sg._gu_tiles
rng = np.random.RandomState(9)
res = {"E": E, "D": D, "H": H, "ns": {}}

for N in NS:
    R = max(1, N // KK)
    x = mx.array(rng.randn(1, R, D).astype(np.float16))
    idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(R)])[None].astype(np.uint32))
    mx.eval(x, idx)
    mx.reset_peak_memory()
    flat = idx.reshape(-1)
    sidx = mx.sort(flat)
    tok = (mx.arange(R * KK, dtype=mx.uint32) // KK)[mx.argsort(flat)]
    tok_x = mx.concatenate([tok, mx.zeros((64,), dtype=tok.dtype)])
    sidx_x = mx.concatenate([sidx, mx.zeros((64,), dtype=sidx.dtype)])
    xp = x.reshape(R, D)[tok_x]
    xg = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 0])
    xu = MOE._rows_prep()(xp, sg._gu_suh[sidx_x, 1])
    nb_max = R * KK // 64 + E + 1
    tab, nbr = MOE._seg_table_fn(E, nb_max, 64)(sidx)
    mx.eval(xg, xu, tab, nbr)
    live = int(np.array(nbr)[0])
    counts = np.bincount(np.array(flat), minlength=E)
    ent = {"R": R, "pairs": R * KK, "live_blocks": live,
           "avg_rows_per_expert": float(R * KK / (counts > 0).sum()),
           "waste": live * 64 / (R * KK)}
    for v in ("v19c", "v19d", "v19e"):
        def gu(v=v):
            a = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=R * KK,
                            tn_base=0, tiles_per_e=GT, out_e=H, version=v)
            b = v19f.seg_mm(xu, sg._gu_trellis, K, CB, tab, nbr, n_rows=R * KK,
                            tn_base=E * GT, tiles_per_e=GT, out_e=H, version=v)
            return a, b
        g, u = gu()
        mx.eval(g, u)
        ent[f"{v}_gu_ms"] = timeit(gu) * 1e3
    # peak + bit-parity (the guard/hoist must not change any live value)
    ent["peak_MB"] = mx.get_peak_memory() / 1e6
    ya = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=R * KK,
                     tn_base=0, tiles_per_e=GT, out_e=H, version="v19c")
    mx.eval(ya)
    for v in ("v19d", "v19e"):
        yv = v19f.seg_mm(xg, sg._gu_trellis, K, CB, tab, nbr, n_rows=R * KK,
                         tn_base=0, tiles_per_e=GT, out_e=H, version=v)
        mx.eval(yv)
        ent[f"parity_{v}"] = bool(mx.array_equal(ya, yv))
    ent["d_speedup"] = ent["v19c_gu_ms"] / ent["v19d_gu_ms"]
    ent["e_speedup"] = ent["v19c_gu_ms"] / ent["v19e_gu_ms"]
    res["ns"][str(N)] = ent
    log(f"N={R*KK:6d} (R={R:4d}) avgrows/expert={ent['avg_rows_per_expert']:6.1f} "
        f"waste={ent['waste']:5.2f}x | v19c={ent['v19c_gu_ms']:8.2f} "
        f"v19d={ent['v19d_gu_ms']:8.2f} ({ent['d_speedup']:5.2f}x) "
        f"v19e={ent['v19e_gu_ms']:8.2f} ({ent['e_speedup']:5.2f}x) ms "
        f"mlx_peak={ent['peak_MB']:5.0f}MB rss_peak={rss_mb():5.0f}MB "
        f"parity d/e={ent['parity_v19d']}/{ent['parity_v19e']}")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
