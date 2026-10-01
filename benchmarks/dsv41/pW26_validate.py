#!/usr/bin/env python3
"""pW26 -- validate PF.decode_prime: timing win + bit-exact parity.

A (baseline): prefill 16K -> decode steps 0..3 timed.
B (fixed):    prefill 16K -> PF.decode_prime() -> steps 0..3 timed.
PARITY: two fresh runs (with/without the prime) teacher-forced over the same
tokens; compare the first real decode step's fp32 logits bit-exactly, and the
compressor carry (kv_state/score_state) bit-exactly, and the window ring.

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

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "4"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW26] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def prefill_to(cache):
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=True)
    mx.eval(am)
    return am


def arm(name, prime):
    cache = fresh_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = prefill_to(cache)
    tp = time.perf_counter() - t0
    tprime = PF.decode_prime(model, cache) if prime else 0.0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    out = []
    for i in range(STEPS):
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        out.append((tb * 1e3, (dt - tb) * 1e3))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in out[2:]]))
    print(f"[pW26] {name}: prefill {tp:.1f}s prime {tprime*1e3:7.1f}ms | " + " ".join(
        f"s{i} {b_:7.2f}+{e:5.2f}" for i, (b_, e) in enumerate(out))
        + f" | first {out[0][0]+out[0][1]:7.2f}ms steady {steady:6.2f}ms "
          f"ratio {(out[0][0]+out[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()
    return tp, tprime


arm("A baseline", False)
arm("B primed", True)
arm("A baseline-2", False)
arm("B primed-2", True)

# ---------------- parity: identical feed, bit compare -----------------------
print("[pW26] === parity (prime must be a semantic no-op) ===", flush=True)
res = {}
for name, prime in (("A", False), ("B", True)):
    cache = fresh_cache()
    am = prefill_to(cache)
    if prime:
        PF.decode_prime(model, cache)
    # snapshot carry + ring before the step
    carry = []
    for lc in cache.layers:
        if lc.comp_state is not None:
            carry.append((mx.array(lc.comp_state.kv_state),
                          mx.array(lc.comp_state.score_state)))
    ring = [mx.array(lc.win_kv) for lc in cache.layers]
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    lg = model(nxt[None], cache, last_logit_only=True)
    row = lg[0, 0].astype(mx.float32)
    mx.eval(row)
    res[name] = (row, carry, ring)
    del cache
    mx.clear_cache()

same_logits = bool(mx.array_equal(res["A"][0], res["B"][0]).item())
d = float(mx.abs(res["A"][0] - res["B"][0]).max().item())
max_carry = max((mx.abs(a[0] - b[0]).max().item() for a, b in
                 zip(res["A"][1], res["B"][1])), default=0.0)
max_carry_s = max((mx.abs(a[1] - b[1]).max().item() for a, b in
                   zip(res["A"][1], res["B"][1])), default=0.0)
max_ring = max((mx.abs(a - b).max().item() for a, b in
                zip(res["A"][2], res["B"][2])), default=0.0)
print(f"[pW26] PARITY: logits exact={same_logits} max|d|={d:.3g} | "
      f"carry kv max|d|={max_carry:.3g} score max|d|={max_carry_s:.3g} | "
      f"window ring max|d|={max_ring:.3g}", flush=True)
print(f"[pW26] peak {mx.get_peak_memory()/1e9:.2f}GB PW26_DONE", flush=True)
