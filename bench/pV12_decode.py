#!/usr/bin/env python3
"""pV12 -- how much of the MoE verify cost is DECODE-repeatable?

pV11 showed _decode_fused2 is flat in unique-expert count (1.157 -> 1.163 ms
from u=6 to u=24, a 4x traffic swing) -- so expert BYTES are not the cost. The
cost is per-slot decode/FMA work. That work is only removable if several rows
share one decode pass, i.e. MT rows per threadgroup (the dense small-batch
trick). This probe prices exactly that, without needing the new kernel:

  A2  at S slots, all pointing at DIFFERENT experts, 1 row each   (today)
  A2  at S slots, S/MT groups of MT rows on ONE expert each        (proxy for
      a grouped kernel: same decode count as the grouped kernel would do,
      just spread over S/MT launches instead of one)

If the second is far cheaper per decode, the grouped kernel is worth building;
if not, the addressable win does not exist.

Run (production down, 1 layer, ~2 GB):
  PV12_PKG=~/dsv41-ws2/V EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV12_decode.py
"""
import os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV12_PKG", HOME + "/dsv41-ws2/V"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.exl3.loader import Exl3Checkpoint, load_experts
from mlx_lm.models.exl3 import exl3_moe as MOE

CK = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
LAYER = int(os.environ.get("PV12_LAYER", "20"))
REPS = int(os.environ.get("PV12_REPS", "21"))
SWEEPS = int(os.environ.get("PV12_SWEEPS", "3"))


def log(*a):
    print("[pV12]", *a, flush=True)


ck = Exl3Checkpoint(CK)
sg = load_experts(ck, LAYER, rank=0, world=2)
E, D, H = sg.num_experts, sg.input_dims, sg.hidden_dims
GT, DT, K, CB = sg._gu_tiles, sg._dn_tiles, sg._k, sg._cb
mx.eval(sg._gu_trellis, sg._dn_trellis, sg._gu_suh, sg._gu_svh, sg._dn_suh, sg._dn_svh)
kA, kB = sg._kernels2()
log(f"layer {LAYER} E={E} D={D} H={H} k={K} gu_tiles={GT} dn_tiles={DT}")
rng = np.random.RandomState(31)


def a2_slots(x2d, sel):
    """Kernel A2 over `sel` slots; x2d has one row per slot."""
    R = int(x2d.shape[0])
    sel_u = sel.astype(mx.uint32)
    suh_sel = sg._gu_suh[sel].reshape(R * 2, D)
    x_rep = mx.broadcast_to(x2d[:, None, :], (R, 2, D)).reshape(R * 2, D)
    xh = MOE._rows_prep()(x_rep, suh_sel)
    dims = mx.array([D // 16, GT, int(sg._gu_trellis.shape[1]), E], dtype=mx.uint32)
    return kA(inputs=[xh.reshape(-1), sg._gu_trellis.reshape(-1).view(mx.uint32),
                      MOE._fwd_perm_u32(), sel_u, dims],
              template=[("T", mx.float16)],
              grid=(R * (2 * GT // MOE._A2_TILES) * MOE._GEM_THREADS, 1, 1),
              threadgroup=(MOE._GEM_THREADS, 1, 1),
              output_shapes=[(R * 2 * H,)], output_dtypes=[mx.float16])[0]


def timed(fn, reps=REPS, warm=4):
    for _ in range(warm):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


log("--- A2: S slots, distinct experts, 1 row each (today's MoE shape) ---")
res = {}
for S in (12, 24, 36, 48):
    x = mx.array(rng.randn(S, D).astype(np.float16))
    sel = mx.array(rng.permutation(E)[:S].astype(np.uint32))
    mx.eval(x, sel)
    med, mn = timed(lambda: a2_slots(x, sel))
    res[S] = med
    log(f"  S={S:3d}: {med:7.3f} ms (min {mn:7.3f})  {med/S:.4f} ms/slot  "
        f"{S*2.95:.0f} MB of expert bytes touched")
log("--- A2: same slot count, but MT rows share ONE expert (grouped proxy) ---")
log("    (each launch = MT rows on one expert; S/MT launches)")
XG, SELG = {}, {}
for S in (24, 48):
    for MT in (2, 4):
        nl = S // MT
        XG[(S, MT)] = [mx.array(rng.randn(MT, D).astype(np.float16)) for _ in range(nl)]
        SELG[(S, MT)] = [
            mx.array(np.full(MT, rng.permutation(E)[0], dtype=np.uint32)) for _ in range(nl)]
        mx.eval(*XG[(S, MT)], *SELG[(S, MT)])
gg = {}
for S in (24, 48):
    for MT in (2, 4):
        nl = S // MT
        per = []
        for sw in range(SWEEPS):
            t0 = time.perf_counter()
            outs = [a2_slots(XG[(S, MT)][g], SELG[(S, MT)][g]) for g in range(nl)]
            mx.eval(outs)          # must depend on the outputs: mx.zeros(1) does NOT
            per.append((time.perf_counter() - t0) * 1e3)
        gg[(S, MT)] = float(np.median(per))
        log(f"  S={S:3d} MT={MT}: {gg[(S,MT)]:7.3f} ms in {nl} launches "
            f"({nl} unique experts = {nl*2.95:.0f} MB) "
            f"vs one-launch S={S} {res[S]:.3f} ms")

log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV12_DONE")
