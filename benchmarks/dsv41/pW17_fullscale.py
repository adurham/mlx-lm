#!/usr/bin/env python3
"""pW17 -- does the first-decode-step premium blow up at FULL memory scale?

pW10/pW16 showed the premium is bounded (~2.2-2.8x, +45 ms) at 20 layers /
55 GB. The production observation is 1.3-1.9 s = 17-25x. Two-node-only factors
cannot be tested here; what CAN be tested single-node is memory scale: a
30-layer subset sits at ~84 GB active / 115 GB wired limit, close to the
production 105 GB/rank.

Arms (ctx=16384, 5 decode steps each, fresh cache):
  A  default boundary
  B  idle 10 s before step 0  (separates "post-prefill state" from "GPU idle")
  C  boundary as A, but with per-step memory/latency detail
Reports per step: host build ms, eval ms, GPU ms, device newBuffer count,
active/peak/pool GB.

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (5)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW17_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def misses(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
dev = mx.device_info()
print(f"[pW17] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB "
      f"wired_limit_default={mx.get_wired_limit()/1e9:.2f}GB working_set={dev['max_recommended_working_set_size']/1e9:.1f}GB",
      flush=True)
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW17] warmup {time.perf_counter()-t0:.1f}s active {mx.get_active_memory()/1e9:.2f}GB", flush=True)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def arm(name, idle_s=0.0, prime=False):
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pf_peak = mx.get_peak_memory()
    tp2 = PF.decode_prime(model, cache) if prime else 0.0
    if idle_s:
        time.sleep(idle_s)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    print(f"[pW17] {name}: prefill {tp:.1f}s peak {pf_peak/1e9:.2f}GB "
          f"prime {tp2*1e3:.1f}ms pool {mx.get_cache_memory()/1e9:.2f}GB", flush=True)
    ts = []
    for i in range(STEPS):
        o0 = off()
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        sz = misses(o0)
        ts.append((tb * 1e3, (dt - tb) * 1e3))
        print(f"[pW17]   step{i} build {tb*1e3:8.2f} eval {(dt-tb)*1e3:6.2f} "
              f"gpu {mx.metal.gpu_time_ns()/1e6:7.2f} newBuf {len(sz):4d}/{sum(sz)/1e6:6.2f}MB "
              f"active {mx.get_active_memory()/1e9:.2f} pool {mx.get_cache_memory()/1e9:.2f}", flush=True)
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in ts[3:]]))
    print(f"[pW17] {name}: first {ts[0][0]+ts[0][1]:.2f}ms steady {steady:.2f}ms "
          f"ratio {(ts[0][0]+ts[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


arm("A default")
arm("B idle-10s", idle_s=10.0)
arm("C default-2nd", prime=False)
print(f"[pW17] peak {mx.get_peak_memory()/1e9:.2f}GB PW17_DONE", flush=True)
