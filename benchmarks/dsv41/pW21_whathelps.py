#!/usr/bin/env python3
"""pW21 -- 30 layers / 16K: what actually removes the +534 ms first step?

Facts to explain (pW18-pW20): step 0 build 594 ms vs steady 60 ms; eval and GPU
flat; every mx.compile key warm; 145-195 fresh device newBuffers on step 0 vs 0
steady; the premium scales super-linearly with the model's active memory
(+15 ms at 23 GB / 8 layers, +45 at 55 GB / 20, +534 at 80 GB / 30).

Arms (fresh cache + 16K prefill each):
  A  default                         baseline
  B  prime = exact step-0 miss sizes (captured at this build), pool cleared first
  C  prime = same, WITHOUT clearing the pool
  D  prime = generic pow2 buckets 256B..32MB, cleared first
  E  default, then roll back to the prefill offset and run step 0 AGAIN
     (separates "per-cache-state" from "per-step-work": if the repeat is fast,
      the cost is a one-time state effect at the boundary)

Env: PW_LAYERS (0..29), PW_CTX (16384), PW_STEPS (4)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW21_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
STEPS = int(os.environ.get("PW_STEPS", "4"))

# step-0 miss sizes measured at 30 layers / 16K (pW20)
HIST = ([16924672, 16793600, 8536064, 2113536]
        + [212992, 163840, 163840]
        + [81920] * 11 + [16384] * 5 + [8192] * 6
        + [5120, 4608, 4608, 4096, 4096, 4096]
        + [2560] * 7 + [2048] * 6 + [1024] * 27
        + [640, 640, 512] + [512] * 11 + [256] * 56)


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
print(f"[pW21] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]


def pow2():
    s, out = 256, []
    while s <= (64 << 20):
        out += [s] * 2
        s *= 2
    return out


def prime(sizes):
    t0 = time.perf_counter()
    bufs = [mx.zeros((n,), dtype=mx.uint8) for n in sizes]
    mx.eval(bufs)
    del bufs
    mx.eval(mx.zeros((8,), dtype=mx.uint8))
    return time.perf_counter() - t0


def fresh_cache():
    c = model.make_cache(1, max_seq_len=CTX + 512)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


def step(cache, nxt):
    o0 = off()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    sz = misses(o0)
    return (tb * 1e3, (dt - tb) * 1e3, len(sz), nxt)


def run(name, *, sizes=None, clear=False):
    cache = fresh_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
    mx.eval(am)
    tp = time.perf_counter() - t0
    tp2 = 0.0
    if sizes is not None:
        if clear:
            mx.clear_cache()
        tp2 = prime(sizes)
    nxt = am.reshape(-1)[-1:].astype(mx.int32)
    out = []
    for i in range(STEPS):
        b_, e, nb, nxt = step(cache, nxt)
        out.append((b_, e, nb))
    steady = float(np.median([t[0] + t[1] for t in out[2:]]))
    print(f"[pW21] {name}: prefill {tp:.1f}s prime {tp2*1e3:6.1f}ms  " + "  ".join(
        f"s{i} {b_:7.2f}+{e:5.2f} nb{nb:4d}" for i, (b_, e, nb) in enumerate(out))
        + f"  first {out[0][0]+out[0][1]:.1f}ms steady {steady:.1f}ms "
          f"ratio {(out[0][0]+out[0][1])/steady:.2f}x", flush=True)
    del cache
    mx.clear_cache()
    return out


run("A default")
run("B hist+clear", sizes=HIST, clear=True)
run("C hist+noclear", sizes=HIST, clear=False)
run("D pow2+clear", sizes=pow2(), clear=True)

# ---- E: roll back to the prefill offset and repeat step 0 -----------------
cache = fresh_cache()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
mx.eval(am)
pos = int(cache.offset)
sn = SP.snap(cache, pos)
nxt = am.reshape(-1)[-1:].astype(mx.int32)
b0, e0, nb0, _ = step(cache, nxt)
st = SP.stashes(cache)
SP.rollback(cache, sn, pos, st)
b1, e1, nb1, _ = step(cache, nxt)
# repeat once more to see if it keeps getting cheaper
sn2 = SP.snap(cache, pos)
b2, e2, nb2, _ = step(cache, nxt)
print(f"[pW21] E step0-twice at ctx={pos}: first {b0:.1f}+{e0:.2f} nb{nb0}  "
      f"repeat {b1:.1f}+{e1:.2f} nb{nb1}  repeat2 {b2:.1f}+{e2:.2f} nb{nb2}", flush=True)

print(f"[pW21] peak {mx.get_peak_memory()/1e9:.2f}GB PW21_DONE", flush=True)
