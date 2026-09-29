#!/usr/bin/env python3
"""pW20 -- localize the 535 ms first-step cost at 30 layers / 16K.

Three measurements in one GPU job:
  1. per-Block timings for decode step 0 vs a steady step (mx.eval after each
     block) -> which layers/mods carry the premium;
  2. the exact size histogram of step-0 device newBuffer misses (what decode
     asks for that the pool cannot serve);
  3. a pure-allocator microbench at this footprint: 200 fresh allocations
     (cache_limit=0) timed at ~80 GB active, to price one miss.

Env: PW_LAYERS (0..29), PW_CTX (16384)
"""
import collections
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW20_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import model as M  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTX = int(os.environ.get("PW_CTX", "16384"))
LBLK = []          # per-block ms for the current step


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
print(f"[pW20] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
      flush=True)
PF.warmup(model)
base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]

_orig_block = M.Block.__call__


def _timed(self, x, pre_mix, start_pos, cache, shared):
    t0 = time.perf_counter()
    h, pm = _orig_block(self, x, pre_mix, start_pos, cache, shared)
    mx.eval(h, pm)
    LBLK.append((self.layer_id, (time.perf_counter() - t0) * 1e3))
    return h, pm


M.Block.__call__ = _timed

cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True, final_clear=False)
mx.eval(am)
print(f"[pW20] prefill {CTX} in {time.perf_counter()-t0:.2f}s peak {mx.get_peak_memory()/1e9:.2f}GB",
      flush=True)
nxt = am.reshape(-1)[-1:].astype(mx.int32)

for i in range(4):
    LBLK.clear()
    o0 = off()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    mx.eval(lg)
    dt = time.perf_counter() - t0
    sz = misses(o0)
    tot = sum(b_[1] for b_ in LBLK)
    top = sorted(LBLK, key=lambda kv: -kv[1])[:6]
    print(f"[pW20] step{i} build {tb*1e3:7.2f} eval {(dt-tb)*1e3:6.2f} "
          f"sum(blocks) {tot:7.2f} newBuf {len(sz):4d}/{sum(sz)/1e6:6.2f}MB top-blocks "
          + " ".join(f"L{k}:{v:.1f}" for k, v in top), flush=True)
    if i == 0:
        c = collections.Counter(sz)
        print(f"[pW20] step0 miss histogram ({len(c)} distinct):", flush=True)
        for k, n in sorted(c.items(), reverse=True)[:22]:
            print(f"[pW20]     {k:10d} B x{n}", flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)

# ---- price one allocation at this footprint --------------------------------
active = mx.get_active_memory()
MX = []
for _ in range(3):
    mx.set_cache_limit(0)
    t0 = time.perf_counter()
    a = [mx.zeros((1 << 20,), dtype=mx.uint8) for _ in range(50)]
    mx.eval(a)
    dt = time.perf_counter() - t0
    MX.append(dt * 1e3 / 50)
    del a
mx.set_cache_limit(mx.get_memory_limit())
print(f"[pW20] fresh-alloc at active={active/1e9:.1f}GB: {np.median(MX):.3f} ms/alloc "
      f"(50 allocs per rep, 3 reps)", flush=True)
mx.clear_cache()
print(f"[pW20] peak {mx.get_peak_memory()/1e9:.2f}GB PW20_DONE", flush=True)
