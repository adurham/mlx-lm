#!/usr/bin/env python3
"""pW16 -- definitive A/B for the post-prefill decode boundary (20 layers, 16K).

Protocol per rep:
  * fresh cache, 16K chunked prefill via PF.prefill (so the pool ends up
    holding prefill-shaped buffers, exactly the production boundary),
  * then 5 decode steps; report step0 vs steady (median of steps 3-4),
  * arms interleaved A,B,A,B,... so drift cannot fake a win.
Arm A: boundary as-is.
Arm B: PF.decode_prime(model, cache) at the boundary.
Then a PARITY pass: same teacher-forced feed, A vs B, logits compared with
mx.array_equal + max|d| (the prime must not change any number).

Env: PW_LAYERS (0..19), PW_CTX (16384), PW_REPS (3), PW_STEPS (5)
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
                                         ",".join(str(i) for i in range(20))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))
REPS = int(os.environ.get("PW_REPS", "3"))

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW16] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def boundary(prime):
    cache = fresh_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                    final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    tp2 = 0.0
    if prime:
        tp2 = PF.decode_prime(model, cache)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    ts = []
    for i in range(STEPS):
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        mx.eval(lg)
        ts.append((time.perf_counter() - t0) * 1e3)
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    peak = mx.get_peak_memory()
    del cache
    mx.clear_cache()
    return tp, tp2, ts, peak


res = {"A": [], "B": []}
for r in range(REPS):
    for arm, prime in (("A", False), ("B", True)):
        tp, tp2, ts, peak = boundary(prime)
        steady = float(np.median(ts[3:]))
        res[arm].append((ts[0], steady, tp, tp2, peak))
        print(f"[pW16] rep{r} {arm}{'+prime' if prime else ''}: prefill {tp:.1f}s "
              f"prime {tp2*1e3:.1f}ms first {ts[0]:7.2f}ms steady {steady:6.2f}ms "
              f"ratio {ts[0]/steady:.2f}x  steps " + " ".join(f"{t:.1f}" for t in ts), flush=True)

print("[pW16] === summary (median of reps) ===", flush=True)
for arm in ("A", "B"):
    firsts = [x[0] for x in res[arm]]
    steadies = [x[1] for x in res[arm]]
    primes = [x[3] for x in res[arm]]
    print(f"[pW16] {arm}: first {np.median(firsts):7.2f}ms  steady {np.median(steadies):6.2f}ms  "
          f"ratio {np.median(firsts)/np.median(steadies):.2f}x  prime {np.median(primes)*1e3:.1f}ms  "
          f"peak {max(x[4] for x in res[arm])/1e9:.2f}GB", flush=True)

# ---- parity: no-prime vs prime, identical feed, bit compare ----------------
feed = (base * 8)[:STEPS + 1]
logs = {}
for arm, prime in (("A", False), ("B", True)):
    cache = fresh_cache()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    if prime:
        PF.decode_prime(model, cache)
    rows = []
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    for i in range(STEPS):
        lg = model(nxt[None], cache, last_logit_only=True)
        lg = lg[:, -1].astype(mx.float32)
        mx.eval(lg)
        rows.append(lg)
        nxt = mx.argmax(lg, axis=-1, keepdims=True).astype(mx.int32)
    logs[arm] = mx.stack(rows)
    del cache
    mx.clear_cache()
same = bool(mx.array_equal(logs["A"], logs["B"]).item())
d = float(mx.abs(logs["A"] - logs["B"]).max().item())
amatch = float((mx.argmax(logs["A"], -1) == mx.argmax(logs["B"], -1)).astype(mx.float32).mean().item())
print(f"[pW16] PARITY A-vs-B over {STEPS} teacher-forced steps: exact={same} max|d|={d:.3g} "
      f"argmax_agree={amatch*100:.1f}%", flush=True)
print(f"[pW16] peak {mx.get_peak_memory()/1e9:.2f}GB PW16_DONE", flush=True)
