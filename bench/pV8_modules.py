#!/usr/bin/env python3
"""pV8 -- per-module ms vs rows for the REAL layer-20 modules (no stubs).

Builds one block (layer 20), then times every EXL3 module's __call__ with the
real input shapes it sees at verify time, for rows 1..8. Prints ms and the
ms/row slope so a module that re-streams its trellis per row shows up.

Run:
  PV8_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV8_modules.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV8_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.exl3.loader import Exl3Checkpoint

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYER = int(os.environ.get("PV8_LAYER", "20"))
REPS = int(os.environ.get("PV8_REPS", "15"))
ROWS = [int(x) for x in os.environ.get("PV8_ROWS", "1,2,3,4,5,6,8").split(",")]


def log(*a):
    print(f"[pV8]", *a, flush=True)


def timeit(fn, reps=REPS, warm=4):
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
args = ModelArgs.from_dict(ck.config)
n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0
blk, rep = eb.build_block(ck, args, LAYER, native=Exl3Checkpoint(NATIVE))
mx.eval(blk.parameters())
log(f"built block {LAYER}; report keys {sorted(rep)[:6]}...")


def walk(mod, path=""):
    for name, child in mod.children().items():
        p = f"{path}.{name}" if path else name
        yield p, child
        yield from walk(child, p)


MODS = []
for p, child in walk(blk):
    if isinstance(child, (eb.Exl3Proj, eb.Exl3Member, eb.Exl3GroupedStack)):
        MODS.append((p, child))
log(f"{len(MODS)} EXL3 modules")
rng = np.random.RandomState(5)


def in_dim(m):
    for src in (m, getattr(m, "_lin", None), getattr(m, "_g", None)):
        if src is None:
            continue
        d = getattr(src, "in_features", None)
        if d:
            return int(d)
    lins = getattr(getattr(m, "_g", None), "lins", None)
    if lins:
        return int(lins[0].in_features)
    return None


res = {}
for p, m in MODS:
    d = in_dim(m)
    if not d:
        log(f"  skip {p} (unknown in_features)")
        continue
    ent = {}
    for R in ROWS:
        xx = mx.array(rng.randn(R, d).astype(np.float16))
        mx.eval(xx)
        try:
            ent[R] = timeit(lambda xx=xx, m=m: m(xx)) * 1e3
        except Exception as ex:
            ent[R] = float("nan")
            log(f"  {p} R={R} FAILED {type(ex).__name__}: {ex}")
    res[p] = ent
    if all(np.isnan(list(ent.values()))):
        continue
    log(f"  {p:34s} in={d:5d} " +
        " ".join(f"R{R}:{ent[R]:6.3f}" for R in ROWS) +
        f"   slope1-4={(ent[4]-ent[1])/3:.4f} slope4-8={(ent[8]-ent[4])/4:.4f} ms/row")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
out = os.environ.get("PV8_JSON")
if out:
    json.dump(res, open(out, "w"), indent=1)
log("PV8_DONE")
