#!/usr/bin/env python3
"""pW28 -- parity-only rerun: driver boundary ON vs OFF must be a no-op.

Three teacher-forced steps per arm at C in (8192, 16384), 30-layer subset:
  * all step logits bit-identical (mx.array_equal, not |d| -- logits carry -inf)
  * compressor carry bit-identical (array_equal; -inf - -inf would be nan)
  * window-ring differences confined to the single probe-written slot
    (offset % window), i.e. never read before overwrite
Env: PW_LAYERS (0..29), PW_CTXS (8192,16384)
"""
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bppw".replace("9bppw", "9bpw")
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS",
                                         ",".join(str(i) for i in range(30))).split(",")]
CTXS = [int(x) for x in os.environ.get("PW_CTXS", "8192,16384").split(",")]

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW28] layers={LAYERS} n={len(model.layers)} active={mx.get_active_memory()/1e9:.2f}GB",
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


for C in CTXS:
    res = {}
    for name, on in (("A-off", False), ("B-on", True)):
        os.environ["DSV41_DECODE_PRIME"] = "1" if on else "0"
        cache = fresh_cache(C)
        ids = (base * (C // len(base) + 1))[:C]
        am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True)
        mx.eval(am)
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
    a, b = res["A-off"], res["B-on"]
    exact = bool(mx.array_equal(a[0], b[0]).item())
    kv_eq = all(bool(mx.array_equal(x[0], y[0]).item()) for x, y in zip(a[1], b[1]))
    sc_eq = all(bool(mx.array_equal(x[1], y[1]).item()) for x, y in zip(a[1], b[1]))
    slot = a[3] % WIN
    slot_only, ndiff = [], []
    for x, y in zip(a[2], b[2]):
        d = np.array(mx.not_equal(x, y))          # [1, window, d]
        n = int(d.sum())
        ndiff.append(n)
        if n:
            cols = set(int(v) for v in np.unique(np.nonzero(d)[1]))
            slot_only.append(cols <= {slot})
        else:
            slot_only.append(True)
    print(f"[pW28] C={C} PARITY: 3-step logits exact={exact} | carry kv_exact={kv_eq} "
          f"score_exact={sc_eq} | ring differing elems/layer={ndiff} probe-slot={slot} "
          f"slot-only={all(slot_only)}", flush=True)
print(f"[pW28] peak {mx.get_peak_memory()/1e9:.2f}GB PW28_DONE", flush=True)
