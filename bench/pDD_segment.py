#!/usr/bin/env python3
"""pDD -- model-level segment attribution for the MoE/MLP prefill cost.

Times ONE 531-token forward of an MoE-heavy layer subset, then the same forward
with the routed-experts call stubbed out (p48's 'experts' ablation). The
difference IS the MoE segment, measured at model scale for both kernel versions.

Env: PD_SEG, PD_PKG, PD_JSON.
"""
import json, os, sys, time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PD_PKG", HOME + "/dsv41-ws/D"))
import numpy as np
import mlx.core as mx

from mlx_lm.models.deepseek_v41 import exl3_build as eb
import mlx_lm.models.deepseek_v41.moe as MO
import mlx_lm.models.exl3.gemv_metal as G

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PD_LAYERS", "0,1,2,20,21").split(",")]
STUB = os.environ.get("PD_STUB", "")
OUT = os.environ.get("PD_JSON")


def log(*a):
    print("[pDD]", *a, flush=True)


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ids = json.load(open(HOME + "/p30_prompt_ids.json"))[:531]


def one_forward(stub=False):
    if stub:
        real = eb.Exl3Experts.__call__
        eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(
            x[:, None, :] * 0, idx.shape + (x.shape[-1],))
    try:
        cache = model.make_cache(1, max_seq_len=len(ids) + 64)
        for li, lc in enumerate(cache.layers):
            if li not in LAYERS:
                lc.comp_state = None
        lg = model(mx.array([ids]), cache, last_logit_only=True)
        mx.eval(lg)
        del cache
    finally:
        if stub:
            eb.Exl3Experts.__call__ = real
    return lg


def timeit(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


res = {"seg": G._MM_SEG_VERSION, "layers": LAYERS, "n": len(ids)}
full = timeit(lambda: one_forward(False))
stubbed = timeit(lambda: one_forward(True))
res["full_ms"] = full
res["no_experts_ms"] = stubbed
res["moe_segment_ms"] = full - stubbed
res["moe_share"] = (full - stubbed) / full
res["layers_per_fwd"] = len(LAYERS)
log(f"full={full:.1f}ms  experts-stubbed={stubbed:.1f}ms  "
    f"MoE segment={res['moe_segment_ms']:.1f}ms ({res['moe_share']*100:.0f}% of forward) "
    f"over {len(LAYERS)} layers")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
log("wrote", OUT)
