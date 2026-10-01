#!/usr/bin/env python3
"""pW9 -- exact size histogram of the decode-boundary buffer-pool misses.

After a 16K prefill, dump the sizes of every device newBuffer (pool miss) for
decode step 0, 1, 2 and a steady step, plus the pool size before/after each.
That says precisely which shape classes are missing from the pool.

Env: PW_LAYERS (0,1,2,3,20,21,24,25), PW_CTX (16384), PW_STEPS (5)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW9_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))
CLEAR = int(os.environ.get("PW_CLEAR", "4"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW9] layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None

mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, clear_cache_every=CLEAR)
mx.eval(am)
print(f"[pW9] prefill {time.perf_counter()-t0:.2f}s peak {mx.get_peak_memory()/1e9:.2f}GB "
      f"pool {mx.get_cache_memory()/1e9:.2f}GB", flush=True)


def read_sizes(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


def hist(sizes, label):
    import collections
    c = collections.Counter(sizes)
    print(f"[pW9] {label}: {len(sizes)} misses {sum(sizes)/1e6:.2f} MB; "
          f"distinct={len(c)}", flush=True)
    for k, n in sorted(c.items(), reverse=True)[:18]:
        print(f"[pW9]     {k:10d} B x{n}", flush=True)


nxt = am.reshape(-1)[-1:].astype(mx.int32)
for i in range(STEPS):
    pool0 = mx.get_cache_memory()
    o0 = os.path.getsize(BUFL)
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    sz = read_sizes(o0)
    hist(sz, f"step{i} build {tb*1e3:.2f}ms eval {(dt-tb)*1e3:.2f}ms "
             f"pool {pool0/1e9:.2f}->{mx.get_cache_memory()/1e9:.2f}GB")
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)
print(f"[pW9] peak {mx.get_peak_memory()/1e9:.2f}GB PW9_DONE", flush=True)
