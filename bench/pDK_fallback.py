#!/usr/bin/env python3
"""pDK -- the gather_mm fallback is unusable at DSv4.1 dims; seg path scale test.

_MOE_MM gates the v19c/e segmented GEMM on N <= _MM_MAX_ROWS (default 9216).
Above that, _prefill falls back to decode_full_eg_mlx + mx.gather_mm. At DSv4.1
rank-0 MoE dims that fallback's rhs is
    dn: (E=384, H=1152, D=5120) fp16 = 4.53 GB
    gu: (2E,  D=5120, H=1152) fp16 = 9.06 GB
both over MLX's int32 buffer limit (~2 GB) -> OverflowError, i.e. an N above the
cap does not degrade, it CRASHES. Verify, then measure the segmented path at
large N to show raising the cap is safe.

Env: PD_NS (comma list of N), PD_REPS, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3 import exl3_moe as MOE
from mlx_lm.models.exl3 import gemv_metal as G
from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PD_LAYER", "20"))
KK = 6
REPS = int(os.environ.get("PD_REPS", "10"))
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDK]", *a, flush=True)


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
res = {"E": E, "D": D, "H": H, "seg": G._MM_SEG_VERSION,
       "mm_max_rows": MOE._MM_MAX_ROWS,
       "gather_dn_bytes": E * H * D * 2, "gather_gu_bytes": 2 * E * D * H * 2,
       "int32_limit": 2**31 - 1}

for nm, nb in (("dn", E * H * D * 2), ("gu", 2 * E * D * H * 2)):
    res[f"{nm}_fits_int32"] = nb <= 2**31 - 1
    log(f"gather rhs {nm}: {nb/1e9:.2f} GB -> fits int32: {nb <= 2**31-1}")

# prove the guard routes away from the impossible fallback
rng = np.random.RandomState(5)
x = mx.array(rng.randn(1, 64, D).astype(np.float16))
idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(64)])[None].astype(np.uint32))
mx.eval(x, idx)
res["fb_bytes_GB"] = MOE._mm_fallback_bytes(E, D, H) / 1e9
res["fb_viable"] = MOE._mm_use_fallback(E, D, H, 384)
log(f"fallback geometry: {res['fb_bytes_GB']:.2f} GB rhs, viable={res['fb_viable']}")
# force the old behaviour to show the crash (subprocess would die: do it in a fork)
import subprocess, sys as _s
code = ("import os,sys,numpy as np,mlx.core as mx;"
        "sys.path.insert(0, os.environ['PD_PKG']);"
        "from mlx_lm.models.exl3 import exl3_moe as M;"
        "from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts;"
        "import os;"
        "ck=Exl3Checkpoint(os.path.expanduser('~')+'/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw');"
        "sg=load_experts(ck,20,rank=0,world=2);"
        "M._MM_MAX_ROWS=0; M._mm_use_fallback=lambda *a: True;"
        "r=np.random.RandomState(5);D=sg.input_dims;E=sg.num_experts;"
        "x=mx.array(r.randn(1,64,D).astype(np.float16));"
        "idx=mx.array(np.stack([r.permutation(E)[:6] for _ in range(64)])[None].astype(np.uint32));"
        "mx.eval(x,idx);y=sg._prefill(x,idx);mx.eval(y);print('UNEXPECTED_OK')")
env = dict(os.environ, PD_PKG=os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
r = subprocess.run([os.sys.executable, "-c", code], capture_output=True, text=True, env=env)
tail = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else ""
res["forced_fallback_rc"] = r.returncode
res["forced_fallback_tail"] = tail
log(f"forced fallback rc={r.returncode}: {tail[:120]}")
MOE._MOE_MM = True

# seg path at large N (above the current cap)
res["ns"] = {}
for N_target in [int(x) for x in os.environ.get("PD_NS", "3072,9216,18432").split(",")]:
    S = N_target // KK
    x = mx.array(rng.randn(1, S, D).astype(np.float16))
    idx = mx.array(np.stack([rng.permutation(E)[:KK] for _ in range(S)])[None].astype(np.uint32))
    mx.eval(x, idx)
    mx.reset_peak_memory()
    y = sg._prefill(x, idx)
    mx.eval(y)
    ent = {"rows": S, "pairs": S * KK, "peak_MB": mx.get_peak_memory() / 1e6,
           "ms": timeit(lambda: sg._prefill(x, idx)) * 1e3}
    ent["trellis_GB_per_s"] = (sg._gu_trellis.size + sg._dn_trellis.size) * 2 / 1e9 / (ent["ms"] / 1e3)
    res["ns"][str(S * KK)] = ent
    log(f"N={ent['pairs']:6d} (R={S:4d}) _prefill={ent['ms']:8.2f} ms peak={ent['peak_MB']:6.0f}MB "
        f"effective {ent['trellis_GB_per_s']:.0f} GB/s of trellis")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
