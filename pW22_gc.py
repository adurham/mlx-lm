#!/usr/bin/env python3
"""pW22 -- is the +570 ms first decode step after a long prefill a Python GC
pause?

Evidence it is not compile/allocation (pW18-pW21, 30 layers/16K, ~80 GB):
  * first step build 580-650 ms vs steady 50 ms; eval 3 ms flat; GPU 0;
  * every mx.compile key already warm (pW11 logged cold keys: all in warmup);
  * 195 fresh device allocations price out at 0.032 ms each = ~6 ms;
  * pooling/priming the exact miss sizes does not help (it hurts);
  * repeating the SAME step after a cache rollback costs 49.9 ms (pW21 E),
    i.e. the premium is a one-time host event after the prefill, not per-shape
    work and not per-input.

Signature: host-only, once per prefill, superlinear in live-object count
(15 ms at 8 layers/23 GB, ~45 at 20/55 GB, 570 at 30/80 GB). That is a
generation-2 cyclic-GC pass.

This probe attributes time to GC directly with gc.callbacks (a callback that
records timestamps gives exact collection durations) and tests gc.disable() and
gc.freeze() as candidate structural fixes.

Arm A: default, gc.callbacks timing on  -> when do collections happen, how long
Arm B: gc.disable() around prefill+decode
Arm C: gc.freeze() after load (the structural fix: model arrays live forever)
Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (4)
"""
import gc
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

PH = ["boot"]
GC_EV = []          # (phase, gen, ms)
_gc_t0 = [0.0]


def _cb(phase, info):
    if phase == "start":
        _gc_t0[0] = time.perf_counter()
    else:
        GC_EV.append((PH[0], info.get("generation"), (time.perf_counter() - _gc_t0[0]) * 1e3))


gc.callbacks.append(_cb)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "4"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW22] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PH[0] = "warmup"
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
print(f"[pW22] gc thresholds={gc.get_threshold()} count={gc.get_count()}", flush=True)


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def arm(name, *, mode):
    if mode == "freeze":
        gc.collect()
        gc.freeze()                       # everything live now is immortal
        print(f"[pW22] frozen {gc.get_freeze_count()} objects", flush=True)
    if mode == "disable":
        gc.disable()
    cache = fresh_cache()
    GC_EV.clear()
    PH[0] = f"{name}:prefill"
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pre_gc = list(GC_EV)
    PH[0] = f"{name}:decode"
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    out = []
    for i in range(STEPS):
        GC_EV.clear()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        out.append((tb * 1e3, (dt - tb) * 1e3, sum(e[2] for e in GC_EV), len(GC_EV)))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in out[2:]]))
    print(f"[pW22] {name}: prefill {tp:.1f}s  prefill gc: {len(pre_gc)} collections "
          f"{sum(e[2] for e in pre_gc):.0f}ms total", flush=True)
    for i, (b_, e, gcms, n) in enumerate(out):
        print(f"[pW22]   step{i} build {b_:7.2f} eval {e:5.2f} gc {gcms:7.2f}ms "
              f"({n} collections)", flush=True)
    print(f"[pW22] {name}: first {out[0][0]+out[0][1]:.1f}ms steady {steady:.1f}ms "
          f"ratio {(out[0][0]+out[0][1])/steady:.2f}x", flush=True)
    if mode == "disable":
        gc.enable()
    if mode == "freeze":
        gc.unfreeze()
    del cache
    mx.clear_cache()


arm("A default", mode="none")
arm("B gc-disabled", mode="disable")
gc.collect()
arm("C gc-frozen", mode="freeze")
print(f"[pW22] peak {mx.get_peak_memory()/1e9:.2f}GB PW22_DONE", flush=True)
