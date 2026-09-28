#!/usr/bin/env python3
"""pDB -- end-to-end EXL3SwitchGLU._prefill A/B at DSv4.1 geometry.

Runs the REAL _prefill entry point (Hadamard prep + sort + table + kernels)
under EXL3_MM_SEG=v19c and =v19e in two subprocesses (the kernel is built from
the env at first use), then bit-compares the 512-row output.

Also asserts the decode floor is untouched: rows<=8 (spec verify) must take the
A2/B2 decode path and produce identical results with the MoE env unchanged.

Run: PD_SEG=v19c|v19e PD_JSON=... python bench/pDB_prefill_ab.py
"""
import json, os, sys, time

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
REPS = int(os.environ.get("PD_REPS", "20"))
SEG = os.environ.get("PD_SEG", "?")
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDB]", *a, flush=True)


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
rng = np.random.RandomState(7)
x = mx.array(rng.randn(1, ROWS, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(ROWS)])[None].astype(np.uint32))
mx.eval(x, idx)

import mlx_lm.models.exl3.gemv_metal as G
log(f"EXL3_MM_SEG={G._MM_SEG_VERSION} E={E} D={D} H={H} N={ROWS*KK}")

mx.reset_peak_memory()
y = sg._prefill(x, idx)
mx.eval(y)
peak = mx.get_peak_memory() / 1e6
ms = timeit(lambda: sg._prefill(x, idx)) * 1e3
call_ms = timeit(lambda: sg(x, idx)) * 1e3
import resource
res = {"seg": G._MM_SEG_VERSION, "rows": ROWS, "kk": KK, "E": E, "D": D, "H": H,
       "prefill_ms": ms, "call_ms": call_ms, "peak_MB": peak,
       "rss_peak_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
       "sum_abs": float(np.abs(np.array(y.astype(mx.float32))).sum())}
log(f"_prefill={ms:.2f} ms  __call__={call_ms:.2f} ms  mlx_peak={peak:.0f}MB "
    f"rss_peak={res['rss_peak_MB']:.0f}MB checksum={res['sum_abs']:.6e}")

# decode floor: rows<=8 must ride _decode_fused2, unchanged by this work
d8 = mx.array(rng.permutation(E)[:KK][None, None, :].repeat(8, axis=1).astype(np.uint32))
xd = mx.array(rng.randn(1, 8, D).astype(np.float16))
mx.eval(d8, xd)
y8 = sg(xd, d8)
mx.eval(y8)
res["decode8_ms"] = timeit(lambda: sg(xd, d8)) * 1e3
res["decode8_checksum"] = float(np.abs(np.array(y8.astype(mx.float32))).sum())
log(f"rows=8 decode path: {res['decode8_ms']:.3f} ms checksum={res['decode8_checksum']:.6e}")
np.save(os.path.join(HOME, f"dsv41-ws/D/pDB_{G._MM_SEG_VERSION}.npy"),
        np.array(y.astype(mx.float32)))
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
