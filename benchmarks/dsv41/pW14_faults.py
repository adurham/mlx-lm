#!/usr/bin/env python3
"""pW14 -- are the first post-prefill decode steps paying OS page faults?

Samples this process's own mach counters (pageins / faults / compressions /
swapouts) via libproc-free `ps` on the pid, around each decode step, plus MLX
active/peak/pool. 20-layer subset, 16K prefill.

If boundary steps show large pagein/compression deltas that vanish by steady
state, the full-scale 1.3-1.9 s is memory-pressure page-in, not compilation.
Env: PW_LAYERS (0..19), PW_CTX (16384), PW_STEPS (6)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(20))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "6"))
PID = os.getpid()


def os_counters():
    """(pageins, faults, cow_faults, compression, swapouts) from ps."""
    try:
        out = subprocess.run(
            ["ps", "-o", "pageins=,faults=,cow_faults=,compressions=,swapouts=",
             "-p", str(PID)],
            capture_output=True, text=True, timeout=5).stdout.split()
        return tuple(int(x) for x in out)
    except Exception:  # noqa: BLE001
        return (0, 0, 0, 0, 0)


def vmstat_lines():
    vals = {}
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    for ln in out.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                vals[k.strip()] = int(v)
    return vals


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW14] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
mx.eval(am)
print(f"[pW14] prefill {CTX} in {time.perf_counter()-t0:.2f}s peak={mx.get_peak_memory()/1e9:.2f}GB "
      f"active={mx.get_active_memory()/1e9:.2f}GB pool={mx.get_cache_memory()/1e9:.2f}GB", flush=True)
c = os_counters()
print(f"[pW14] counters after prefill: pageins={c[0]} faults={c[1]} cow={c[2]} "
      f"compress={c[3]} swapouts={c[4]}", flush=True)

nxt = am.reshape(-1)[-1:].astype(mx.int32)
for i in range(STEPS):
    c0 = os_counters()
    v0 = vmstat_lines()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    c1 = os_counters()
    v1 = vmstat_lines()
    d = [c1[k] - c0[k] for k in range(5)]
    fr = v1.get("Pages free", 0) - v0.get("Pages free", 0)
    co = v1.get("Pages occupied by compressor", 0) - v0.get("Pages occupied by compressor", 0)
    sw = v1.get("Pages swapped out", 0) - v0.get("Pages swapped out", 0)
    print(f"[pW14] step{i} build {tb*1e3:7.2f} eval {(dt-tb)*1e3:6.2f} | "
          f"pageins+{d[0]} faults+{d[1]} cow+{d[2]} compress+{d[3]} swapout+{d[4]} | "
          f"free {fr*16/1024:+.0f}MB comp {co*16/1024:+.1f}MB swapout {sw*16/1024:+.1f}MB | "
          f"mlx active {mx.get_active_memory()/1e9:.2f} pool {mx.get_cache_memory()/1e9:.2f}",
          flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)
print(f"[pW14] peak {mx.get_peak_memory()/1e9:.2f}GB PW14_DONE", flush=True)
