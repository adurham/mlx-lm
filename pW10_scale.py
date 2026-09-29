#!/usr/bin/env python3
"""pW10 -- post-prefill decode boundary at ~60-70 GB active (scale test).

Same protocol as pW8 but with a 20-layer subset, so the model sits at the
memory scale where the full two-node run showed the 1.3-1.9 s first steps.
Arms vary context length and the prefill->decode boundary hygiene:

  A ctx=16384, boundary default   (pool left full of prefill-shaped buffers)
  B ctx=16384, boundary clear     (final_clear=True)
  C ctx=16384, boundary clear + memory pressure (limit = active + 0.5 GB)
  D ctx=8192,  boundary default
  E ctx=8192,  boundary clear
  F ctx=512,   boundary clear     (big model, NO long prefill: control)

Per step: host build ms, eval ms, device newBuffer calls/bytes, pool size.
This distinguishes "long prefill leaves a bad allocator state" from
"any first decode step at this model size".

Env: PW_LAYERS (default 0..19), PW_STEPS (6)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW10_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", ",".join(str(i) for i in range(20))).split(",")]
STEPS = int(os.environ.get("PW_STEPS", "6"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW10] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW10] warmup {time.perf_counter()-t0:.1f}s", flush=True)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
MEMLIMIT = mx.get_memory_limit()
dev = mx.device_info()
print(f"[pW10] memlimit={MEMLIMIT/1e9:.1f}GB working_set={dev['max_recommended_working_set_size']/1e9:.1f}GB "
      f"ram={dev['memory_size']/1e9:.1f}GB", flush=True)


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def delta(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        vals = [int(x) for x in f.read().split(b"\n") if x.strip()]
    return len(vals), sum(vals)


def arm(name, ctx, *, final_clear=False, pressure=False):
    cache = model.make_cache(1, max_seq_len=max(ctx, 256) + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    if ctx > 0:
        ids = (base * (ctx // len(base) + 1))[:ctx]
        t0 = time.perf_counter()
        am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                        final_clear=final_clear)
        mx.eval(am)
        tp = time.perf_counter() - t0
        nxt = am.reshape(-1)[-1:].astype(mx.int32)
    else:
        tp = 0.0
        nxt = mx.array([base[0]], dtype=mx.int32)
    pf_peak = mx.get_peak_memory()
    pool_pf = mx.get_cache_memory()
    act = mx.get_active_memory()
    if pressure:
        mx.set_memory_limit(act + int(0.5e9))
    rows = []
    for i in range(STEPS):
        o0 = off()
        p0 = mx.get_cache_memory()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        c, b = delta(o0)
        rows.append((tb * 1e3, (dt - tb) * 1e3, c, b / 1e6, p0 / 1e9,
                     mx.get_cache_memory() / 1e9))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    mx.set_memory_limit(MEMLIMIT)
    print(f"[pW10] {name}: ctx={ctx} prefill {tp:.2f}s peak {pf_peak/1e9:.2f}GB "
          f"pool_after_prefill {pool_pf/1e9:.2f}GB active {act/1e9:.2f}GB", flush=True)
    print("[pW10]   " + " | ".join(
        f"b{b_:.1f}+e{e:.1f} nbuf{c} {mb:.1f}MB pool{p0:.2f}" for b_, e, c, mb, p0, p1 in rows),
        flush=True)
    steady = np.median([r[0] + r[1] for r in rows[3:]]) if len(rows) > 3 else float("nan")
    print(f"[pW10]   first {rows[0][0]+rows[0][1]:.2f}ms  steady {steady:.2f}ms  "
          f"ratio {(rows[0][0]+rows[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


arm("A default", 16384)
arm("B boundary-clear", 16384, final_clear=True)
arm("C clear+pressure", 16384, final_clear=True, pressure=True)
arm("D default", 8192)
arm("E boundary-clear", 8192, final_clear=True)
arm("F no-long-prefill(control)", 0, final_clear=True)
print(f"[pW10] peak {mx.get_peak_memory()/1e9:.2f}GB PW10_DONE", flush=True)
