#!/usr/bin/env python3
"""pW1 -- locate the first-decode-step cost after a LONG prefill (W stream).

Reproduces the phase-21 observation on a single-node layer subset (<=8 GB):
after an 8K/16K prefill the first decode steps cost 1.3-1.9 s instead of
steady state.  Splits every step into host (python graph build) vs GPU
(mx.eval) time, counts dispatches, and -- crucially -- records every
``mx.compile``d function call whose first invocation is slow (i.e. a
re-trace/recompile for a new shape/dtype key) with its name and shapes.
Same for ``mx.fast.metal_kernel`` specializations.

Env: PW_CTX (8192), PW_LAYERS (2,20), PW_CTX_FIRST (skip prefill if 0),
     PW_STEPS (12), PW_STUB (none|idx|sattn|experts), PW_SLOW_MS (20).
Writes: nothing (stdout only).
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

SLOW_MS = float(os.environ.get("PW_SLOW_MS", "20"))
EVENTS = []          # (ms, kind, name, shapes)
_t0 = time.perf_counter()


def _rel():
    return time.perf_counter() - _t0


_orig_compile = mx.compile


def _counting_compile(fun=None, **kw):
    c = _orig_compile(fun, **kw) if fun is not None else _orig_compile(**kw)
    name = getattr(fun, "__qualname__", str(fun))

    def w(*a, **k):
        t0 = time.perf_counter()
        r = c(*a, **k)
        dt = (time.perf_counter() - t0) * 1e3
        if dt > SLOW_MS:
            EVENTS.append((dt, "compile", name,
                           tuple(getattr(x, "shape", None) for x in a
                                 if isinstance(x, mx.array))))
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
        dt = (time.perf_counter() - t0) * 1e3
        if dt > SLOW_MS:
            EVENTS.append((dt, "metal", name, tuple(k2.get("template", ()))))
        return r

    return w


mx.fast.metal_kernel = _counting_mk

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as A, indexer as IX, prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
DO_PREFILL = int(os.environ.get("PW_CTX_FIRST", "1")) == 1
STEPS = int(os.environ.get("PW_STEPS", "12"))
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
elif STUB == "experts":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(
        x[:, None, :] * 0, idx.shape + (x.shape[-1],))
print(f"[pW1] built layers={LAYERS} stub={STUB} active={mx.get_active_memory()/1e9:.1f}GB", flush=True)

t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW1] warmup {time.perf_counter()-t0:.1f}s", flush=True)
for e in EVENTS:
    print(f"[pW1]   warmup-slow {e[1]}:{e[2]} {e[0]:.0f}ms {e[3]}", flush=True)
EVENTS.clear()

base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None

if DO_PREFILL:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
    mx.eval(am)
    tp = time.perf_counter() - t0
    print(f"[pW1] prefill {CTX} tok in {tp:.2f}s ({CTX/tp:.0f} tok/s) peak={mx.get_peak_memory()/1e9:.2f}GB",
          flush=True)
    for e in EVENTS:
        print(f"[pW1]   prefill-slow {e[1]}:{e[2]} {e[0]:.0f}ms {e[3]}", flush=True)
    EVENTS.clear()
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
else:
    nxt = mx.array([base[0]], dtype=mx.int32)

print("[pW1] step  build_ms  eval_ms  disp  | slow host events", flush=True)
for i in range(STEPS):
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    d0 = mx.metal.dispatch_count()
    mx.eval(lg)
    te = time.perf_counter() - t0
    disp = mx.metal.dispatch_count() - d0
    ev = "; ".join(f"{e[2]}({e[0]:.0f}ms)" for e in EVENTS)
    EVENTS.clear()
    print(f"[pW1] {i:4d}  {tb*1e3:8.1f} {te*1e3-tb*1e3:8.1f} {disp:6d}  | {ev}", flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)

print(f"[pW1] peak {mx.get_peak_memory()/1e9:.2f}GB PW_DONE", flush=True)
