#!/usr/bin/env python3
"""pW12 -- structural fix for the first post-prefill decode step: PRIME the
Metal buffer pool with the sizes the decode step will ask for.

Mechanism (measured pW9/pW10/pW11): MLX's Metal buffer cache only reuses a
buffer whose size sits in [_SIZE_MIN, min(2*size, size+2*pagesize)] of the
request, so a pool filled with PREFILL-shaped tiles never satisfies the decode
step's context-sized transients (kv_all / idxs concats: 8-17 MB at 16K). The
first decode step therefore allocates ~130-270 fresh MTLBuffers, the next 2-3
steps decay, steady state reuses. That is the whole first-step premium: eval/GPU
time is flat (3 ms) and every mx.compile key is already warm.

Fix under test: at the prefill->decode boundary, allocate-and-free the decode
working set once so those buffers land in the Metal pool and the first decode
step reuses them instead of allocating.

Arms (20-layer subset, prefill 16K, 5 decode steps each):
  A baseline
  B prime = exact size histogram of A's step-0 misses
  C prime = generic pow2 buckets 256B..32MB
  D prime = exact, after the driver's trailing clear_cache()
Env: PW_LAYERS (0..19), PW_CTX (16384), PW_STEPS (5)
"""
import collections
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW12_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(20))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "5"))
PRIME_MODE = os.environ.get("PW_PRIME", "auto")

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW12] layers={LAYERS} n={len(LAYERS)} active={mx.get_active_memory()/1e9:.2f}GB "
      f"prime={PRIME_MODE}", flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]

# size histogram of the baseline's step-0 misses, captured by the first arm
EXACT_SIZES = []


def off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def misses(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        return [int(x) for x in f.read().split(b"\n") if x.strip()]


def prime(sizes):
    """Allocate + free one buffer per size so the Metal pool holds them."""
    if not sizes:
        return 0.0
    t0 = time.perf_counter()
    bufs = [mx.zeros((n,), dtype=mx.uint8) for n in sizes]
    mx.eval(bufs)
    del bufs
    return time.perf_counter() - t0


def pow2_sizes():
    s, out = 256, []
    while s <= (32 << 20):
        out += [s] * 2
        s *= 2
    # plus the context-sized concat transients seen at 16K
    out += [16793600, 16924672, 8536064, 8388608, 7864320, 2113536, 1310720]
    return out


def run(name, mode):
    global EXACT_SIZES
    cache = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                    final_clear=(mode == "exact-after-clear"))
    mx.eval(am)
    tp = time.perf_counter() - t0
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    tprime = 0.0
    if mode == "exact" or mode == "exact-after-clear":
        tprime = prime(EXACT_SIZES)
    elif mode == "pow2":
        tprime = prime(pow2_sizes())
    rows = []
    for i in range(STEPS):
        o0 = off()
        p0 = mx.get_cache_memory()
        t0 = time.perf_counter()
        lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
        tb = time.perf_counter() - t0
        mx.eval(lg)
        dt = time.perf_counter() - t0
        sz = misses(o0)
        rows.append((tb * 1e3, (dt - tb) * 1e3, len(sz), sum(sz) / 1e6, p0 / 1e9))
        nxt = lg.reshape(-1)[-1:].astype(mx.int32)
    if not EXACT_SIZES:
        EXACT_SIZES = [s for s in misses(0)] if False else sz_snapshot
    print(f"[pW12] {name}: prefill {tp:.2f}s prime {tprime*1e3:.1f}ms "
          f"pool {mx.get_cache_memory()/1e9:.2f}GB peak {mx.get_peak_memory()/1e9:.2f}GB", flush=True)
    for i, (b_, e, c, mb, p0) in enumerate(rows):
        print(f"[pW12]   step{i} build {b_:7.2f} eval {e:5.2f} total {b_+e:7.2f}ms "
              f"newBuf {c:4d}/{mb:7.2f}MB pool {p0:.2f}GB", flush=True)
    steady = float(np.median([r[0] + r[1] for r in rows[3:]])) if len(rows) > 3 else float("nan")
    print(f"[pW12]   FIRST {(rows[0][0]+rows[0][1]):.2f}ms  steady {steady:.2f}ms  "
          f"ratio {(rows[0][0]+rows[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()
    return sz_snapshot if False else None


sz_snapshot = []
# arm A: baseline, capture the step-0 miss histogram
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
mx.eval(am)
tp = time.perf_counter() - t0
nxt = am.reshape(-1)[-1:].astype(mx.int32)
rows = []
for i in range(STEPS):
    o0 = off()
    p0 = mx.get_cache_memory()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    sz = misses(o0)
    if i == 0:
        sz_snapshot = sz
    rows.append((tb * 1e3, (dt - tb) * 1e3, len(sz), sum(sz) / 1e6, p0 / 1e9))
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)
EXACT_SIZES = list(sz_snapshot)
print(f"[pW12] A baseline: prefill {tp:.2f}s pool {mx.get_cache_memory()/1e9:.2f}GB "
      f"peak {mx.get_peak_memory()/1e9:.2f}GB", flush=True)
for i, (b_, e, c, mb, p0) in enumerate(rows):
    print(f"[pW12]   step{i} build {b_:7.2f} eval {e:5.2f} total {b_+e:7.2f}ms "
          f"newBuf {c:4d}/{mb:7.2f}MB pool {p0:.2f}GB", flush=True)
steady = float(np.median([r[0] + r[1] for r in rows[3:]]))
print(f"[pW12]   FIRST {(rows[0][0]+rows[0][1]):.2f}ms  steady {steady:.2f}ms  "
      f"ratio {(rows[0][0]+rows[0][1])/steady:.2f}x", flush=True)
del cache
mx.clear_cache()

run("B prime-exact", "exact")
run("C prime-pow2", "pow2")
run("D prime-exact-after-clear", "exact-after-clear")
print(f"[pW12] peak {mx.get_peak_memory()/1e9:.2f}GB PW12_DONE", flush=True)
