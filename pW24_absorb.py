#!/usr/bin/env python3
"""pW24 -- the first-decode-step premium is the first GPU flush after a long
prefill (a host sync), not compilation. Where can it be absorbed?

pW23 cProfile (30 layers / 16K): sparse_attention.py:205(sparse_attn) holds
530 of 549 ms of step-0 host build, same 31/30 call count as steady state
(38 ms). That function ends each layer's attention with mx.eval(out) -- a
blocking sync. So the premium is a one-time host wait inside the first sync
after the prefill; everything else (compile keys, allocation, GC, page faults)
has been excluded by measurement.

Arms:
  A  default                                     (baseline)
  B  mx.synchronize() right after prefill        (absorb at the boundary)
  C  mx.eval of a 1-element array after prefill  (cheap fence)
  D  after prefill, one full decode step then rollback to the offset
     (a "boundary warm": check the next real step is steady)
Reports, per arm: prefill s (incl. the boundary call), then steps 0..3 ms.

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (4)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "4"))


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def step(cache, nxt):
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    return lg, tb * 1e3, (dt - tb) * 1e3


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW24] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def arm(name, mode):
    cache = fresh_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    tbound = 0.0
    if mode in ("sync", "fence", "warmstep"):
        tb0 = time.perf_counter()
        if mode == "sync":
            mx.synchronize()
        elif mode == "fence":
            mx.eval(mx.zeros((1,), dtype=mx.uint8))
        else:                        # warmstep: full throwaway decode + rollback
            pos = int(cache.offset)
            sn = SP.snap(cache, pos)
            nxt0 = am.reshape(-1)[-1:].astype(mx.int32)
            _, _, _ = step(cache, nxt0)
            SP.rollback(cache, sn, pos, SP.stashes(cache))
            mx.eval([lc.comp_state.kv_state for lc in cache.layers
                     if lc.comp_state is not None])
        tbound = time.perf_counter() - tb0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    out = []
    for i in range(STEPS):
        lg, b_, e = step(cache, nxt)
        out.append((b_, e))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in out[2:]]))
    print(f"[pW24] {name}: prefill {tp:.1f}s boundary {tbound*1e3:7.1f}ms | " + " ".join(
        f"s{i} {b_:7.2f}+{e:5.2f}" for i, (b_, e) in enumerate(out))
        + f" | first {out[0][0]+out[0][1]:.1f}ms steady {steady:.1f}ms "
          f"ratio {(out[0][0]+out[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


arm("A default", "none")
arm("B synchronize", "sync")
arm("C tiny-fence", "fence")
arm("D warmstep+rollback", "warmstep")
arm("A again", "none")
print(f"[pW24] peak {mx.get_peak_memory()/1e9:.2f}GB PW24_DONE", flush=True)
