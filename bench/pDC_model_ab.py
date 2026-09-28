#!/usr/bin/env python3
"""pDC -- model-level prefill A/B: same 531-token prompt, EXL3_MM_SEG=v19c vs v19e.

Builds the layer subset p48 uses (0,1,2,3,20,21,24,25) rank 0 of world 2, runs
ONE 531-token forward (the prefill path, rows=531 > 16) and saves the logits.
Two processes (one per kernel version) must produce bit-identical logits.

Also reports the forward wall time per version (first call and steady) so the
MoE prefill segment gain shows up at model scale, and the rows<=8 decode path
timing (must be unchanged: it never touches the segmented kernel).

Env: PD_PKG, PD_SEG (v19c|v19e), PD_JSON, PD_LAYERS, PD_SAVE (npy path).
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M
import mlx_lm.models.exl3.gemv_metal as G

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PD_LAYERS", "0,1,2,3,20,21").split(",")]
OUT = os.environ.get("PD_JSON")
SAVE = os.environ.get("PD_SAVE")


def log(*a):
    print("[pDC]", *a, flush=True)


model, rep = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
log(f"seg={G._MM_SEG_VERSION} layers={LAYERS} active={mx.get_active_memory()/1e9:.1f}GB")
import resource
mx.reset_peak_memory()

ids = json.load(open(HOME + "/p30_prompt_ids.json"))[:531]
cache = model.make_cache(1, max_seq_len=len(ids) + 64)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None

res = {"seg": G._MM_SEG_VERSION, "layers": LAYERS, "n": len(ids)}
t0 = time.perf_counter()
lg = model(mx.array([ids]), cache, last_logit_only=True)
mx.eval(lg)
res["first_ms"] = (time.perf_counter() - t0) * 1e3
ts = []
for _ in range(5):
    cache2 = model.make_cache(1, max_seq_len=len(ids) + 64)
    for li, lc in enumerate(cache2.layers):
        if li not in LAYERS:
            lc.comp_state = None
    t = time.perf_counter()
    lg = model(mx.array([ids]), cache2, last_logit_only=True)
    mx.eval(lg)
    ts.append(time.perf_counter() - t)
    del cache2
ts.sort()
res["prefill_ms"] = ts[len(ts) // 2] * 1e3
res["prefill_tok_s"] = len(ids) / (ts[len(ts) // 2])
res["logit_sum"] = float(lg.astype(mx.float32).sum())
res["argmax"] = int(mx.argmax(lg[0, -1]))
log(f"first={res['first_ms']:.0f}ms prefill={res['prefill_ms']:.1f}ms "
    f"({res['prefill_tok_s']:.0f} tok/s over {len(ids)} tok, {len(LAYERS)} layers) "
    f"logit_sum={res['logit_sum']:.8e} argmax={res['argmax']}")

# decode rows<=8 path (spec verify shape) must be untouched
from mlx_lm.models.deepseek_v41 import spec as SP
cache3 = model.make_cache(1, max_seq_len=len(ids) + 256)
for li, lc in enumerate(cache3.layers):
    if li not in LAYERS:
        lc.comp_state = None
model(mx.array([ids]), cache3, last_logit_only=True)
feed = json.load(open(HOME + "/p30_prompt_ids.json"))[64:72]
dts = []
for i in range(8):
    pos = cache3.offset
    sn = SP.snap(cache3, pos)
    chunk = [feed[(i + j) % len(feed)] for j in range(8)]
    t = time.perf_counter()
    lg8 = model(mx.array([chunk]), cache3)
    mx.eval(lg8)
    dts.append(time.perf_counter() - t)
    st = SP.stashes(cache3)
    SP.rollback(cache3, sn, pos + 1, st)
    mx.eval([lc.comp_state.kv_state for lc in cache3.layers if lc.comp_state is not None])
dts.sort()
res["verify8_ms"] = dts[len(dts) // 2] * 1e3
res["verify8_sum"] = float(lg8.astype(mx.float32).sum())
log(f"rows=8 verify: {res['verify8_ms']:.1f} ms sum={res['verify8_sum']:.8e}")

res["mlx_peak_GB"] = mx.get_peak_memory() / 1e9
res["rss_peak_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
log(f"peaks: mlx={res['mlx_peak_GB']:.2f}GB rss={res['rss_peak_GB']:.2f}GB")
if SAVE:
    np.save(SAVE, np.array(lg.astype(mx.float32)))
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
