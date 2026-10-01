#!/usr/bin/env python3
"""pW3 -- localize the first-decode-step cost after long prefill (subset).

Runs the p64-shaped sequence: warmup (scratch cache) -> chunked prefill of C
tokens -> plain 1-row decode steps, timing each step's host build / eval /
GPU / dispatch count. PW_LOC=1 additionally fences (mx.eval) after every Block
so the slow layer of the first step is identified by wall time.

Env: PW_CTX (8192), PW_LAYERS (2,20), PW_STEPS (10), PW_PLAIN (0=driver),
     PW_LOC (0), PW_CHUNK (0=driver default), PW_STUB (none|idx|sattn)
"""
import collections
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

EVENTS = []          # (ms, kind, name)
_t0 = time.perf_counter()
SLOW_MS = float(os.environ.get("PW_SLOW_MS", "15"))


def _note(kind, name, ms, extra=""):
    if ms > SLOW_MS:
        EVENTS.append(f"{kind}:{name} {ms:.0f}ms {extra}")


_orig_compile = mx.compile


def _counting_compile(fun=None, **kw):
    c = _orig_compile(fun, **kw) if fun is not None else _orig_compile(**kw)
    name = getattr(fun, "__qualname__", str(fun))[:60]

    def w(*a, **k):
        t0 = time.perf_counter()
        r = c(*a, **k)
        _note("compile", name, (time.perf_counter() - t0) * 1e3)
        return r

    return w


mx.compile = _counting_compile
_orig_mk = mx.fast.metal_kernel


def _counting_mk(*a, **kw):
    k = _orig_mk(*a, **kw)
    name = kw.get("name", "?")

    def w(*a2, **k2):
        t0 = time.perf_counter()
        r = k(*a2, **k2)
        _note("metal", name, (time.perf_counter() - t0) * 1e3,
              str(k2.get("template", "")))
        return r

    return w


mx.fast.metal_kernel = _counting_mk

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as A, indexer as IX, prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import model as M  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
STEPS = int(os.environ.get("PW_STEPS", "10"))
PLAIN = os.environ.get("PW_PLAIN", "0") == "1"
LOC = os.environ.get("PW_LOC", "0") == "1"
CHUNK = int(os.environ.get("PW_CHUNK", "0"))
STUB = os.environ.get("PW_STUB", "none")

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
if STUB == "idx":
    def _cheap(self, x, qr, sp, off, c, s, ik, sh):
        nb = ik.shape[1]
        k = min(self.index_topk, nb)
        n = x.shape[1]
        return mx.broadcast_to(mx.arange(k, dtype=mx.int32)[None, None] + off,
                               (x.shape[0], n, k)) + (x[..., :1] * 0).astype(mx.int32)
    IX.Indexer.__call__ = _cheap
elif STUB == "sattn":
    A.sparse_attn = lambda q, kv, sink, idx, sc, *a, **k: q
print(f"[pW3] built layers={LAYERS} stub={STUB} plain={PLAIN} loc={LOC} "
      f"active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW3] warmup {time.perf_counter()-t0:.1f}s", flush=True)
EVENTS.clear()

base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None

mx.reset_peak_memory()
t0 = time.perf_counter()
if PLAIN:
    ch = CHUNK or 512
    pos = 0
    while pos < CTX:
        piece = ids[pos:pos + ch]
        r = model(mx.array([piece]), cache, last_logit_only=True, argmax=True)
        mx.eval(r)
        pos += len(piece)
    am = r
else:
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
    mx.eval(am)
tp = time.perf_counter() - t0
print(f"[pW3] prefill {CTX} in {tp:.2f}s ({CTX/tp:.0f} tok/s) peak={mx.get_peak_memory()/1e9:.2f}GB",
      flush=True)
print(f"[pW3] prefill-events {EVENTS}", flush=True)
EVENTS.clear()

blocks = [b for b in model.layers]
if LOC:
    orig_block = M.Block.__call__

    def _timed_block(self, x, pre_mix, start_pos, cache, shared):
        t0 = time.perf_counter()
        h, pm = orig_block(self, x, pre_mix, start_pos, cache, shared)
        mx.eval(h, pm)
        dt = (time.perf_counter() - t0) * 1e3
        EVENTS.append(f"layer{self.layer_id} {dt:.0f}ms")
        return h, pm
    M.Block.__call__ = _timed_block

nxt = am.reshape(-1)[-1:].astype(mx.int32)
print("[pW3] step build eval gpu disp  events", flush=True)
for i in range(STEPS):
    EVENTS.clear()
    mx.metal.reset_gpu_time()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    d0 = mx.metal.dispatch_count()
    mx.eval(lg)
    te = time.perf_counter() - t0
    print(f"[pW3] {i:3d} {tb*1e3:8.1f} {(te-tb)*1e3:7.1f} "
          f"{mx.metal.gpu_time_ns()/1e6:7.1f} {mx.metal.dispatch_count()-d0:5d}  "
          f"{EVENTS}", flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)

print(f"[pW3] peak {mx.get_peak_memory()/1e9:.2f}GB PW3_DONE", flush=True)
