#!/usr/bin/env python3
"""pW27 -- acceptance run for PF.decode_prime at 8K and 16K.

For each context C in (8192, 16384), same 30-layer subset:
  A baseline: prefill -> 4 decode steps timed
  B primed:   prefill (with the driver's new default boundary) -> 4 steps timed
  PARITY: fresh A and B runs, 3 teacher-forced steps, compare ALL step logits
          bit-exactly (mx.array_equal), carry buffers nan-safely
          (array_equal, since -inf - -inf is nan), and attribute the window
          ring difference to the single probe-written slot (offset % window).
Acceptance: B first step <= 1.2x B steady, and parity exact on every step.

Env: PW_LAYERS (0..29), PW_CTXS (8192,16384), PW_STEPS (4)
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
CTXS = [int(x) for x in os.environ.get("PW_CTXS", "8192,16384").split(",")]
STEPS = int(os.environ.get("PW_STEPS", "4"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW27] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
WIN = int(model.args.window_size)


def fresh_cache(C):
    c = model.make_cache(1, max_seq_len=C + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def prefill_to(cache, C):
    ids = (base * (C // len(base) + 1))[:C]
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
    mx.eval(am)
    return am


def timed_arm(C, prime):
    # drive the boundary explicitly: the driver default is now ON, so the
    # baseline arm must disable it via the env the driver reads
    os.environ["DSV41_DECODE_PRIME"] = "1" if prime else "0"
    cache = fresh_cache(C)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = prefill_to(cache, C)
    tp = time.perf_counter() - t0
    os.environ["DSV41_DECODE_PRIME"] = "1"
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
    first = out[0][0] + out[0][1]
    print(f"[pW27] C={C} {'primed' if prime else 'baseline'}: prefill {tp:.1f}s | " + " ".join(
        f"s{i} {b_:7.2f}+{e:5.2f}" for i, (b_, e) in enumerate(out))
        + f" | first {first:7.2f}ms steady {steady:6.2f}ms ratio {first/steady:.2f}x",
        flush=True)
    del cache
    mx.clear_cache()
    return first, steady


for C in CTXS:
    timed_arm(C, True)
    timed_arm(C, False)

print("[pW27] === parity (driver default boundary vs disabled) ===", flush=True)
for C in CTXS:
    res = {}
    for name, prime in (("A", False), ("B", True)):
        os.environ["DSV41_DECODE_PRIME"] = "1" if prime else "0"
        cache = fresh_cache(C)
        am = prefill_to(cache, C)
        pos = int(cache.offset)
        carry = [(mx.array(lc.comp_state.kv_state), mx.array(lc.comp_state.score_state))
                 for lc in cache.layers if lc.comp_state is not None]
        ring = [mx.array(lc.win_kv) for lc in cache.layers]
        nxt = am.reshape(-1)[-1:].astype(mx.int32)
        rows = []
        for i in range(3):
            lg = model(nxt[None], cache, last_logit_only=True)
            row = lg[0, 0].astype(mx.float32)
            mx.eval(row)
            rows.append(row)
            nxt = mx.argmax(row, axis=-1).astype(mx.int32)[None]
        res[name] = (mx.stack(rows), carry, ring, pos)
        del cache
        mx.clear_cache()
    os.environ["DSV41_DECODE_PRIME"] = "1"
    a, b = res["A"], res["B"]
    exact = bool(mx.array_equal(a[0], b[0]).item())
    d = float(mx.abs(a[0] - b[0]).max().item())
    kv_eq = all(bool(mx.array_equal(x[0], y[0]).item()) for x, y in zip(a[1], b[1]))
    sc_eq = all(bool(mx.array_equal(x[1], y[1]).item()) for x, y in zip(a[1], b[1]))
    ring_delta = [int(mx.sum(mx.not_equal(x, y)).item()) for x, y in zip(a[2], b[2])]
    slot = a[3] % WIN
    # verify every difference is in the probe-written slot of that layer's ring
    slot_only = []
    for x, y in zip(a[2], b[2]):
        diff = mx.not_equal(x, y)[0]
        idxs = set(int(v) for v in np.array(mx.argwhere(diff)[:, 1])) if bool(mx.any(diff).item()) else set()
        slot_only.append(idxs <= {slot})
    print(f"[pW27] C={C} PARITY: 3-step logits exact={exact} max|d|={d:.3g} | "
          f"carry kv_exact={kv_eq} score_exact={sc_eq} | "
          f"ring differing cols per layer={ring_delta} (probe slot {slot}) "
          f"slot-only={all(slot_only)}", flush=True)
print(f"[pW27] peak {mx.get_peak_memory()/1e9:.2f}GB PW27_DONE", flush=True)
