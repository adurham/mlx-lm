#!/usr/bin/env python3
"""pW11 -- where does the first post-prefill decode step's host time go?

8-layer subset, driver prefill 8K, then:
  * cProfile of decode step 0 and of a steady step (top by tottime);
  * per-op census (python graph ms / eval ms) for both steps, with an eval
    fence inside each wrapper for attribution (structure changes, attribution
    is what we want);
  * mx.compile key-cold counter: how many *new* keys each compiled fn sees in
    step 0 vs steady (first-call-of-a-key = a trace).
Env: PW_LAYERS, PW_CTX (8192), PW_STEPS (5)
"""
import collections
import cProfile
import io
import json
import os
import pstats
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

# --- mx.compile key-cold counter ------------------------------------------
COLD = collections.Counter()
CALLS = collections.Counter()
_phase = ["boot"]
_oc = mx.compile


def _cmp(fun=None, **kw):
    c = _oc(fun, **kw) if fun is not None else _oc(**kw)
    nm = getattr(fun, "__qualname__", "?")[:48]
    seen = set()

    def w(*a, **k):
        CALLS[nm] += 1
        key = tuple((getattr(x, "shape", None), getattr(x, "dtype", None)) for x in a)
        if key not in seen:
            seen.add(key)
            COLD[nm] += 1
            if COLD[nm] < 6:
                print(f"[pW11] COLDKEY[{_phase[0]}] {nm} {key}", flush=True)
        t0 = time.perf_counter()
        r = c(*a, **k)
        dt = (time.perf_counter() - t0) * 1e3
        if dt > 8:
            print(f"[pW11] SLOW[{_phase[0]}] {nm} {dt:.1f}ms {key}", flush=True)
        return r
    return w


mx.compile = _cmp

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as A, indexer as IX, prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, hc_fused as HF  # noqa: E402
from mlx_lm.models.deepseek_v41 import compressor as CP  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import RMSNorm  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
STEPS = int(os.environ.get("PW_STEPS", "5"))

CEN = {}


def wrap(tag, obj, attr):
    orig = getattr(obj, attr)

    def w(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        tg = time.perf_counter() - t0
        outs = r if isinstance(r, (list, tuple)) else [r]
        t1 = time.perf_counter()
        mx.eval([t for t in outs if isinstance(t, mx.array)])
        te = time.perf_counter() - t1
        d = CEN.setdefault(tag, [0, 0.0, 0.0])
        d[0] += 1
        d[1] += tg * 1e3
        d[2] += te * 1e3
        return r
    setattr(obj, attr, w)


wrap("block", M.Block, "__call__")
wrap("attn", A.Attention, "__call__")
wrap("indexer", IX.Indexer, "__call__")
wrap("sparse_attn", A, "sparse_attn")
wrap("compressor", CP.Compressor, "__call__")
wrap("moe", MO.MoE, "__call__")
wrap("gate", MO.Gate, "__call__")
wrap("shared", MO.SharedExpert, "__call__")
wrap("experts", eb.Exl3Experts, "__call__")
wrap("proj", eb.Exl3Proj, "__call__")
wrap("member", eb.Exl3Member, "__call__")
wrap("hc_mix", HF, "mixes_and_collapse")
wrap("hc_exp", HF, "hc_expand")
wrap("rms", RMSNorm, "__call__")
wrap("rope", A, "rope_tail")

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW11] layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
_phase[0] = "warmup"
PF.warmup(model)
_phase[0] = "prefill"
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
print(f"[pW11] prefill {CTX} in {time.perf_counter()-t0:.2f}s peak={mx.get_peak_memory()/1e9:.2f}GB",
      flush=True)

nxt = am.reshape(-1)[-1:].astype(mx.int32)


def one_step(label, profile=False):
    for k in CEN:
        CEN[k] = [0, 0.0, 0.0]
    _phase[0] = label
    if profile:
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        pr.disable()
        sio = io.StringIO()
        pstats.Stats(pr, stream=sio).sort_stats("tottime").print_stats(22)
        print(f"[pW11] === {label}: build-before-eval {tb*1e3:.1f}ms", flush=True)
        print(sio.getvalue()[:5200], flush=True)
    else:
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
    t1 = time.perf_counter()
    mx.eval(lg)
    te = time.perf_counter() - t1
    print(f"[pW11] {label}: build {tb*1e3:7.2f}ms eval {te*1e3:6.2f}ms", flush=True)
    print(f"[pW11]   per-op (calls graph_ms eval_ms): " + "; ".join(
        f"{k} {v[0]} {v[1]:.1f} {v[2]:.1f}" for k, v in
        sorted(CEN.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:12]), flush=True)
    return lg


lg = one_step("step0", profile=True)
nxt = lg.reshape(-1)[-1:].astype(mx.int32)
for i in range(1, STEPS):
    lg = one_step(f"step{i}", profile=(i == STEPS - 1))
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)

print("[pW11] compile-key cold counts (fn: cold_keys/calls):", flush=True)
for k, v in COLD.most_common(20):
    print(f"[pW11]   {k}: {v}/{CALLS[k]}", flush=True)
print(f"[pW11] peak {mx.get_peak_memory()/1e9:.2f}GB PW11_DONE", flush=True)
