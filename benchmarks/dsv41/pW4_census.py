#!/usr/bin/env python3
"""pW4 -- census of the prefill->decode boundary (which op is COLD at decode?).

For each context in PW_CTXS (0 = no prefill, then 2048/8192/16384):
  warmup -> [prefill] -> 6 decode steps.
Records, per phase:
  * every mx.compile'd callable key whose FIRST call happens in that phase
    (name, input shapes, host ms) -- "cold compile" evidence;
  * every mx.fast.metal_kernel specialization first called in that phase;
  * per decode step: host build ms, eval ms, GPU ms, dispatches;
  * the shapes feeding sparse_attn / Indexer / Attention per call, so the
    first decode step's shapes can be diffed against steady state and against
    the warmup decode step.
Prints `PW4_COLD <phase> ...` lines for cold items and a shape diff summary.

Env: PW_CTXS (0,2048,8192,16384), PW_LAYERS (2,20), PW_STEPS (6),
     PW_LONG (0 = 512-row chunks throughout), PW_STUB, PW_SLOW_MS (2)
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

SLOW_MS = float(os.environ.get("PW_SLOW_MS", "2"))
PHASE = ["startup"]
COLD = []                     # (phase, kind, name, shapes)
PHASE_TIMES = collections.defaultdict(list)


def _phase():
    return PHASE[0]


def _cold(kind, name, shapes, ms):
    COLD.append((_phase(), kind, name, shapes, ms))


_orig_compile = mx.compile


def _counting_compile(fun=None, **kw):
    c = _orig_compile(fun, **kw) if fun is not None else _orig_compile(**kw)
    base = getattr(fun, "__qualname__", None) or getattr(fun, "__name__", None) or "lambda"
    tname = getattr(getattr(fun, "__self__", None), "__class__", None)
    name = f"{tname.__name__}.{base}" if tname is not None else base
    seen = set()

    def w(*a, **k):
        shapes = tuple((getattr(x, "shape", None), getattr(x, "dtype", None)) for x in a)
        t0 = time.perf_counter()
        r = c(*a, **k)
        ms = (time.perf_counter() - t0) * 1e3
        key = (shapes, tuple((str(getattr(v, "shape", None)), str(getattr(v, "dtype", None)))
                             for v in k.values()))
        if key not in seen:
            seen.add(key)
            if ms > SLOW_MS or PHASE[0].startswith("decode"):
                _cold("mx.compile", name, shapes, ms)
        return r

    return w


mx.compile = _counting_compile
_orig_mk = mx.fast.metal_kernel


def _counting_mk(*a, **kw):
    k = _orig_mk(*a, **kw)
    name = kw.get("name", "?")
    seen = set()

    def w(*a2, **k2):
        tmpl = tuple(k2.get("template", ()))
        t0 = time.perf_counter()
        r = k(*a2, **k2)
        ms = (time.perf_counter() - t0) * 1e3
        key = (tmpl, tuple(str(s) for s in k2.get("output_shapes", ())))
        if key not in seen:
            seen.add(key)
            if ms > SLOW_MS or PHASE[0].startswith("decode"):
                _cold("metal_kernel", name, (tmpl, tuple(k2.get("output_shapes", ()))), ms)
        return r

    return w


mx.fast.metal_kernel = _counting_mk

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as A, indexer as IX, prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTXS = [int(x) for x in os.environ.get("PW_CTXS", "0,2048,8192,16384").split(",")]
STEPS = int(os.environ.get("PW_STEPS", "6"))
LONG = int(os.environ.get("PW_LONG", "0"))

# --- shape taps: record the shapes each key op sees, per call -------------
SHAPES = collections.defaultdict(list)   # tag -> list of shape tuples (per phase)
_orig_sattn = A.sparse_attn
_orig_idx = IX.Indexer.__call__
_orig_attn = A.Attention.__call__


def _tap_sattn(q, kv, sink, idxs, scale, *a, **k):
    SHAPES[f"{_phase()}|sparse_attn"].append(
        ("q", tuple(q.shape), "kv", tuple(kv.shape), "idxs", tuple(idxs.shape)))
    return _orig_sattn(q, kv, sink, idxs, scale, *a, **k)


def _tap_idx(self, x, qr, sp, off, cos, sin, index_k, shared):
    SHAPES[f"{_phase()}|indexer"].append(
        (f"L{self.layer_id}", "x", tuple(x.shape), "index_k", tuple(index_k.shape),
         "sp", sp, "off", off))
    return _orig_idx(self, x, qr, sp, off, cos, sin, index_k, shared)


def _tap_attn(self, x, sp, cache, shared):
    lc = cache.layers[self.layer_id]
    SHAPES[f"{_phase()}|attn"].append((f"L{self.layer_id}", "x", tuple(x.shape),
                                       "sp", sp, "ratio", self.ratio))
    return _orig_attn(self, x, sp, cache, shared)


A.sparse_attn = _tap_sattn
IX.Indexer.__call__ = _tap_idx
A.Attention.__call__ = _tap_attn

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
PHASE[0] = "build"
print(f"[pW4] built layers={LAYERS} active={mx.get_active_memory()/1e9:.1f}GB", flush=True)
_di = mx.metal.device_info()
print(f"[pW4] device recommended_working_set={_di.get('max_recommended_working_set_size', 0)/1e9:.1f}GB "
      f"memory_size={_di.get('memory_size', 0)/1e9:.1f}GB", flush=True)

PHASE[0] = "warmup"
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW4] warmup {time.perf_counter()-t0:.1f}s", flush=True)

base = json.load(open(HOME + "/p30_prompt_ids.json"))
for C in CTXS:
    PHASE[0] = f"prefill{C}"
    cache = model.make_cache(1, max_seq_len=max(C, 256) + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    if C > 0:
        ids = (base * (C // len(base) + 1))[:C]
        t0 = time.perf_counter()
        am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                        long_threshold=8192 if LONG else 10 ** 9)
        mx.eval(am)
        tp = time.perf_counter() - t0
        print(f"[pW4] prefill {C} in {tp:.2f}s ({C/tp:.0f} tok/s) peak={mx.get_peak_memory()/1e9:.2f}GB",
              flush=True)
        nxt = am.reshape(-1)[-1:].astype(mx.int32)
    else:
        nxt = mx.array([base[0]], dtype=mx.int32)
    PHASE[0] = f"d{C}"
    steps = []
    for i in range(STEPS):
        PHASE[0] = f"d{C}s{i}"
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        d0 = mx.metal.dispatch_count()
        mx.eval(lg)
        dt = time.perf_counter() - t0
        steps.append((tb * 1e3, (dt - tb) * 1e3, mx.metal.gpu_time_ns() / 1e6,
                      mx.metal.dispatch_count() - d0))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    PHASE[0] = f"after{C}"
    print(f"[pW4] ctx={C:6d} steps " + " | ".join(
        f"{b:.1f}+{e:.1f}ms gpu {g:.1f} d{d}" for b, e, g, d in steps), flush=True)

PHASE[0] = "report"
print("=== COLD-DECODE CALIBRATION (clear_compile_cache, then one decode) ===", flush=True)
PHASE[0] = "coldcal"
for C in CTXS:
    cache2 = model.make_cache(1, max_seq_len=max(C, 256) + 512)
    for li, lc in enumerate(cache2.layers):
        if li not in LAYERS:
            lc.comp_state = None
    if C > 0:
        ids = (base * (C // len(base) + 1))[:C]
        am2 = PF.prefill(model, ids, cache2, last_logit_only=True, argmax=True)
        mx.eval(am2)
        nxt2 = am2.reshape(-1)[-1:].astype(mx.int32)
    else:
        nxt2 = mx.array([base[0]], dtype=mx.int32)
    mx.clear_compile_cache()
    PHASE[0] = f"cold{C}"
    ts = []
    for i in range(3):
        t0 = time.perf_counter()
        lg = model(nxt2[None], cache2, last_logit_only=True, argmax=True)
        mx.eval(lg)
        ts.append((time.perf_counter() - t0) * 1e3)
        nxt2 = lg.reshape(-1)[-1:].astype(mx.int32)
    print(f"[pW4] ctx={C} COLD decode steps: " + " ".join(f"{t:.1f}ms" for t in ts), flush=True)
    del cache2
    mx.clear_cache()

print("=== COLD CALLS (first use of a compile key / kernel specialization) ===", flush=True)
by_phase = collections.defaultdict(list)
for ph, kind, name, shapes, ms in COLD:
    by_phase[ph].append((ms, kind, name, shapes))
for ph in sorted(by_phase):
    items = sorted(by_phase[ph], reverse=True)
    hot = [(ms, k, n) for ms, k, n, s in items if ms > 5.0]
    print(f"[pW4] phase {ph}: {len(items)} cold keys, {len(hot)} slower than 5ms", flush=True)
    for ms, k, n, s in items[:14]:
        print(f"[pW4]    {ms:8.1f}ms {k:14s} {n}  {str(s)[:150]}", flush=True)

print("=== decode-step shape diff (sparse_attn / indexer per layer) ===", flush=True)
for C in CTXS:
    first = SHAPES.get(f"d{C}s0|sparse_attn", [])
    steady = SHAPES.get(f"d{C}s{STEPS-1}|sparse_attn", [])
    i_first = SHAPES.get(f"d{C}s0|indexer", [])
    i_steady = SHAPES.get(f"d{C}s{STEPS-1}|indexer", [])
    w_first = SHAPES.get(f"d{C}s0|attn", [])
    print(f"[pW4] ctx={C}: sparse_attn first={first}", flush=True)
    print(f"[pW4] ctx={C}: sparse_attn steady={steady} identical={first == steady}", flush=True)
    print(f"[pW4] ctx={C}: indexer first={i_first}", flush=True)
    print(f"[pW4] ctx={C}: indexer steady={i_steady} identical={i_first == i_steady}", flush=True)
    print(f"[pW4] ctx={C}: attn first={w_first}", flush=True)

print(f"[pW4] peak {mx.get_peak_memory()/1e9:.2f}GB PW4_DONE", flush=True)
