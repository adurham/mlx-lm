#!/usr/bin/env python3
"""pDE -- _prefill segment timing at the EXACT model shape (531 x 6 = 3186).

Also times _prefill with a SKEWED expert distribution (what a trained router
produces: fewer distinct experts -> longer segments) to show the guard's gain
is largest exactly where DSv4.1 sits (tiny, uniform segments).

Env: PD_SEG, PD_PKG, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
import mlx_lm.models.exl3.gemv_metal as G

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
REPS = int(os.environ.get("PD_REPS", "25"))
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDE]", *a, flush=True)


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
rng = np.random.RandomState(11)

CASES = {}
# A) uniform routing, the model shape 531x6
n531 = 531
idx_u = np.stack([rng.permutation(E)[:6] for _ in range(n531)])[None].astype(np.uint32)
CASES["uniform_531x6"] = idx_u
# B) skewed routing (Zipf-ish, 64 hot experts) -- trained-router-like
w = 1.0 / (np.arange(1, E + 1) ** 0.9)
w = w / w.sum()
hot_ = rng.choice(E, size=n531 * 6, replace=True, p=w)
idx_s = hot_.reshape(1, n531, 6).astype(np.uint32)
CASES["skewed_531x6"] = idx_s

res = {"seg": G._MM_SEG_VERSION, "E": E, "D": D, "H": H,
       "trellis_MB": (sg._gu_trellis.size + sg._dn_trellis.size) * 2 / 1e6, "cases": {}}
for name, idx in CASES.items():
    B, S, kk = idx.shape
    m = mx.array(idx)
    x = mx.array(rng.randn(B, S, D).astype(np.float16))
    mx.eval(m, x)
    flat = np.array(m).reshape(-1)
    counts = np.bincount(flat, minlength=E)
    nb = int(np.sum((counts + 63) // 64))
    mx.reset_peak_memory()
    y = sg._prefill(x, m)
    mx.eval(y)
    ent = {
        "rows": S, "pairs": S * kk,
        "experts_hit": int((counts > 0).sum()),
        "avg_rows_per_expert": float(flat.size / (counts > 0).sum()),
        "live_blocks_bm64": nb, "block_rows": nb * 64,
        "waste": nb * 64 / flat.size,
        "prefill_ms": timeit(lambda: sg._prefill(x, m)) * 1e3,
        "peak_MB": mx.get_peak_memory() / 1e6,
        "checksum": float(np.abs(np.array(y.astype(mx.float32))).sum()),
    }
    res["cases"][name] = ent
    log(f"{name:16s} pairs={ent['pairs']:5d} experts={ent['experts_hit']:3d} "
        f"avgrows/expert={ent['avg_rows_per_expert']:5.1f} blocks={nb:4d} "
        f"waste={ent['waste']:5.2f}x -> {ent['prefill_ms']:7.2f} ms "
        f"peak={ent['peak_MB']:.0f}MB sum={ent['checksum']:.6e}")
    np.save(os.path.join(HOME, f"dsv41-ws/D/pDE_{G._MM_SEG_VERSION}_{name}.npy"),
            np.array(y.astype(mx.float32)))
    # what the pure trellis read would cost
    log(f"{'':16s} trellis {res['trellis_MB']:.0f}MB -> "
        f"{res['trellis_MB']/1e3/5e11*1e6:.1f} ms at 500GB/s, "
        f"effective {res['trellis_MB']/1e3/(ent['prefill_ms']/1e3)/1e3:.0f} GB/s")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
