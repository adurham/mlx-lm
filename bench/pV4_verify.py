#!/usr/bin/env python3
"""pV4 -- the REAL verify marginal on a layer subset (p63 vcost shape).

p48's ROWS>1 arm computes full fp32 logits for every row (129k vocab x 5120),
which is not what spec verify does. This harness measures rows 1..6 the way
spec.py does it: `model(vin, cache, return_taps=True, argmax=True)`, one sync
per step, snap/rollback between steps. Optional stubs isolate components.

Run:
  PV4_PKG=~/dsv41-ws2/V PV4_LAYERS=0,1,2,3,20,21,24,25 PV4_ROWS=1,2,3,4,5,6 \
   PV4_STUB=none EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
   lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV4_verify.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV4_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A
from mlx_lm.models.deepseek_v41 import spec as SP

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV4_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
ROWS = [int(x) for x in os.environ.get("PV4_ROWS", "1,2,3,4,5,6").split(",")]
REPS = int(os.environ.get("PV4_REPS", "12"))
STUB = os.environ.get("PV4_STUB", "none")


def log(*a):
    print("[pV4]", *a, flush=True)


if STUB == "experts":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(
        x[:, None, :] * 0, idx.shape + (x.shape[-1],))
elif STUB == "dense":
    def _z(x, n):
        return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (n,)).astype(x.dtype)
    eb.Exl3Proj.__call__ = lambda self, x: _z(x, self._lin.out_features)
    eb.Exl3Member.__call__ = lambda self, x: _z(x, self.out_features)
    eb.Exl3GroupedStack.__call__ = lambda self, x: _z(x, self._g.outs[0])
elif STUB == "moekern":
    # keep everything except the two MoE kernels: route every slot's output to
    # a cheap broadcast (same shapes) so the suh gather / prep / sum stay live
    import mlx_lm.models.exl3.exl3_moe as EM

    def _fake(self, x2d, indices):
        R, kk = int(indices.shape[0]), int(indices.shape[1])
        return mx.broadcast_to(x2d[:, None, :] * 0,
                               (R, kk, self.input_dims))
    EM.EXL3SwitchGLU._decode_fused2 = _fake
elif STUB == "attn":
    A.Attention.__call__ = lambda self, x, sp, c, s: x

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
log(f"built layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB stub={STUB}")

allids = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = allids[:64]
feed = allids[64:64 + 40]
cache = model.make_cache(1, max_seq_len=len(ids) + 256)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
logits = model(mx.array([ids]), cache, last_logit_only=True, argmax=True)
mx.eval(logits)
pos0 = cache.offset
log(f"prefill OK offset={pos0}")

res = {}
for R in ROWS:
    cache.offset = pos0
    ts = []
    for i in range(REPS):
        pos = cache.offset
        sn = SP.snap(cache, pos)
        vin = mx.array([[feed[(i + j) % len(feed)] for j in range(R)]], dtype=mx.int32)
        s0 = time.perf_counter()
        out = model(vin, cache, return_taps=True, argmax=True)
        mx.eval(out[0])
        ts.append((time.perf_counter() - s0) * 1e3)
        SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
    res[R] = float(np.median(ts[3:]))
    log(f"R={R}: {res[R]:7.3f} ms  (raw {[f'{x:.1f}' for x in ts]})")
base = res[ROWS[0]]
for R in ROWS[1:]:
    log(f"   marginal vs R{ROWS[0]}: R{R}-R{ROWS[0]} = {res[R]-base:6.3f} ms "
        f"({(res[R]-base)/(R-ROWS[0]):6.3f} ms/row)")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB active={mx.get_active_memory()/1e9:.2f}GB")
log("PV4_DONE")
