#!/usr/bin/env python3
"""pV5 -- what the MoE verify cost is actually made of (layer 20, single node).

Isolates kernel A2 (gate+up) and B2 (down) of EXL3SwitchGLU._decode_fused2 and
sweeps: number of slots, number of UNIQUE experts, number of unique x rows.
Also compares _decode_fused2 against the _prefill (segmented) path at R=2..6.

Run:
  PV5_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV5_moe.py
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV5_PKG", HOME + "/dsv41-ws2/V"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV5_LAYER", "20"))
REPS = int(os.environ.get("PV5_REPS", "15"))


def log(*a):
    print("[pV5]", *a, flush=True)


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
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
kA, kB = sg._kernels2()
log(f"layer {LAYER} E={E} D={D} H={H} k={sg._k} gu_tiles={sg._gu_tiles} dn_tiles={sg._dn_tiles}")
rng = np.random.RandomState(3)


def run_A2(x2d, sel):
    """Kernel A2 only (gate+up mapped GEMV over slots)."""
    kk = 1
    R = int(x2d.shape[0])
    E_sel = R
    sel_u = sel.astype(mx.uint32)
    suh_sel = sg._gu_suh[sel].reshape(E_sel * 2, D)
    x_rep = mx.broadcast_to(x2d[:, None, :], (R, 2, D)).reshape(E_sel * 2, D)
    xh = MOE._rows_prep()(x_rep, suh_sel)
    dims = mx.array([D // 16, sg._gu_tiles, int(sg._gu_trellis.shape[1]), E], dtype=mx.uint32)
    return kA(
        inputs=[xh.reshape(-1), sg._gu_trellis.reshape(-1).view(mx.uint32),
                MOE._fwd_perm_u32(), sel_u, dims],
        template=[("T", mx.float16)],
        grid=(E_sel * (2 * sg._gu_tiles // MOE._A2_TILES) * MOE._GEM_THREADS, 1, 1),
        threadgroup=(MOE._GEM_THREADS, 1, 1),
        output_shapes=[(E_sel * 2 * H,)],
        output_dtypes=[mx.float16],
    )[0]


def run_B2(h, sel):
    E_sel = int(sel.shape[0])
    sel_u = sel.astype(mx.uint32)
    dims_b = mx.array([H // 16, sg._dn_tiles, int(sg._dn_trellis.shape[1]), E], dtype=mx.uint32)
    return kB(
        inputs=[h, sg._dn_trellis.reshape(-1).view(mx.uint32), MOE._fwd_perm_u32(),
                sel_u, sg._gu_svh.reshape(-1), sg._dn_suh.reshape(-1),
                sg._dn_svh.reshape(-1), dims_b],
        template=[("T", mx.float16)],
        grid=(E_sel * (sg._dn_tiles // 8) * MOE._GEM_THREADS, 1, 1),
        threadgroup=(MOE._GEM_THREADS, 1, 1),
        output_shapes=[(E_sel * D,)],
        output_dtypes=[mx.float16],
    )[0]


log("--- A2 (gate+up): slots vs unique experts (8 distinct x rows always) ---")
x8 = mx.array(rng.randn(8, D).astype(np.float16))
mx.eval(x8)
for slots in (7, 14, 21, 28, 56):
    R = slots  # one row per slot
    uniq_sets = {
        "distinct": mx.array(rng.permutation(E)[:slots].astype(np.uint32)),
        "one_expert": mx.array(np.full(slots, 7, np.uint32)),
    }
    for nm, sel in uniq_sets.items():
        mx.eval(sel)
        nr = min(R, 8)
        xa = mx.concatenate([x8[:nr]] * ((R + nr - 1) // nr), axis=0)[:R]
        t = timeit(lambda sel=sel, xa=xa: run_A2(xa, sel)) * 1e3
        log(f"  A2 slots={slots:3d} {nm:11s}: {t:7.3f} ms")

log("--- A2 with a single shared x row (28 slots) ---")
for nm, xa in (("28 distinct rows", mx.concatenate([x8] * 4, axis=0)[:28]),
               ("1 row broadcast", mx.concatenate([x8[:1]] * 28, axis=0))):
    sel = mx.array(rng.permutation(E)[:28].astype(np.uint32))
    mx.eval(sel, xa)
    t = timeit(lambda: run_A2(xa, sel)) * 1e3
    log(f"  A2 {nm:18s}: {t:7.3f} ms")

log("--- B2 (down): slots vs unique experts ---")
h28 = mx.array(rng.randn(28, 2 * H).astype(np.float16))
mx.eval(h28)
for slots in (7, 14, 28, 56):
    for nm, sel in (("distinct", mx.array(rng.permutation(E)[:slots].astype(np.uint32))),
                    ("one_expert", mx.array(np.full(slots, 7, np.uint32)))):
        mx.eval(sel)
        hb = mx.concatenate([h28] * ((slots + 27) // 28), axis=0)[:slots]
        t = timeit(lambda sel=sel, hb=hb: run_B2(hb, sel)) * 1e3
        log(f"  B2 slots={slots:3d} {nm:11s}: {t:7.3f} ms")

log("--- full _decode_fused2 vs _prefill at R=1..6 (kk=7) ---")
for R in (1, 2, 3, 4, 5, 6, 8):
    x = mx.array(rng.randn(1, R, D).astype(np.float16))
    idx = mx.array(np.stack([np.concatenate([rng.permutation(E - 1)[:6], [E - 1]])
                             for _ in range(R)])[None].astype(np.uint32))
    mx.eval(x, idx)
    uniq = len(np.unique(np.array(idx)))
    t2 = timeit(lambda: sg._decode_fused2(x.reshape(R, D), idx.reshape(R, 7))) * 1e3

    def pf():
        return sg._prefill(x, idx)
    t3 = None
    try:
        t3 = timeit(pf) * 1e3
    except Exception as ex:
        t3 = f"FAIL({type(ex).__name__})"
    log(f"  R={R} uniq={uniq:3d}: decode_fused2 {t2:7.3f} ms   prefill {t3 if isinstance(t3, str) else f'{t3:7.3f} ms'}")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV5_DONE")
