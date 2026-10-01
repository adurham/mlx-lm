#!/usr/bin/env python3
"""pW18 -- combined: (1) 30-layer scale test of the boundary premium,
(2) A/B parity for PF.decode_prime (must be bit-exact).

Timing arms at ctx=16384, 5 decode steps, fresh cache each:
  A default   B boundary-clear   C prime   D idle-10s
Then parity: identical teacher-forced feed with and without the prime;
logits compared with mx.array_equal.

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (5)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW18_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def misses(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
dev = mx.device_info()
print(f"[pW18] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB "
      f"working_set={dev['max_recommended_working_set_size']/1e9:.1f}GB",
      flush=True)
t0 = time.perf_counter()
PF.warmup(model)
print(f"[pW18] warmup {time.perf_counter()-t0:.1f}s active {mx.get_active_memory()/1e9:.2f}GB", flush=True)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def arm(name, *, clear=False, prime=False, idle_s=0.0):
    cache = fresh_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=clear)
    mx.eval(am)
    tp = time.perf_counter() - t0
    pf_peak = mx.get_peak_memory()
    tp2 = PF.decode_prime(model, cache) if prime else 0.0
    if idle_s:
        time.sleep(idle_s)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    ts = []
    for i in range(STEPS):
        o0 = off()
        mx.metal.reset_gpu_time()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        sz = misses(o0)
        ts.append((tb * 1e3, (dt - tb) * 1e3, mx.metal.gpu_time_ns() / 1e6, len(sz)))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    steady = float(np.median([t[0] + t[1] for t in ts[3:]]))
    print(f"[pW18] {name}: prefill {tp:.1f}s peak {pf_peak/1e9:.2f}GB prime {tp2*1e3:.1f}ms", flush=True)
    for i, (b_, e, g, nb) in enumerate(ts):
        print(f"[pW18]   step{i} build {b_:8.2f} eval {e:6.2f} gpu {g:7.2f} "
              f"newBuf {nb:4d} active {mx.get_active_memory()/1e9:.2f} pool {mx.get_cache_memory()/1e9:.2f}",
              flush=True)
    print(f"[pW18] {name}: first {ts[0][0]+ts[0][1]:.2f}ms steady {steady:.2f}ms "
          f"ratio {(ts[0][0]+ts[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()


arm("A default")
arm("B boundary-clear", clear=True)
arm("C prime", prime=True)
arm("D idle-10s", idle_s=10.0)

# ---- parity: default vs prime, identical teacher-forced feed ---------------
print("[pW18] === parity A-vs-B (prime) ===", flush=True)
logs = {}
for name, prime in (("A", False), ("B", True)):
    cache = fresh_cache()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    if prime:
        PF.decode_prime(model, cache)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    rows = []
    for i in range(STEPS):
        lg = model(nxt[None], cache, last_logit_only=True)      # fp32 logits [1,1,V]
        row = lg[0, 0].astype(mx.float32)
        mx.eval(row)
        rows.append(row)
        nxt = mx.array([[int(base[i])]], dtype=mx.int32)        # teacher forced
    logs[name] = mx.stack(rows)
    del cache
    mx.clear_cache()
same = bool(mx.array_equal(logs["A"], logs["B"]).item())
d = float(mx.abs(logs["A"] - logs["B"]).max().item())
agree = float((mx.argmax(logs["A"], -1) == mx.argmax(logs["B"], -1))
              .astype(mx.float32).mean().item())
print(f"[pW18] PARITY over {STEPS} steps: exact={same} max|d|={d:.3g} argmax_agree={agree*100:.1f}%",
      flush=True)
print(f"[pW18] peak {mx.get_peak_memory()/1e9:.2f}GB PW18_DONE", flush=True)
