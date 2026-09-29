#!/usr/bin/env python3
"""pU0 -- memory probe: how big is a body layer subset + the draft head?

Stream U needs a runnable 1-3 layer subset that still feeds the DSpark taps
(37,38,39) and owns their compressed-KV source (20). This measures the peak of
that build at rank 0 / world 2 (half-width experts, no all_sum), so the real
harness can stay under the 8 GB cap.

Env: PU0_LAYERS, PU0_SKIP_HEAD=1 to stop after the body.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PU0_PKG", HOME + "/dsv41-ws2/U"))

import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PU0_LAYERS", "20,37,38,39").split(",")]


def log(*a):
    print("[pU0]", *a, flush=True)


mx.reset_peak_memory()
t0 = time.time()
model, rep = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
log(f"body layers={LAYERS} {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")

if os.environ.get("PU0_SKIP_HEAD") != "1":
    t0 = time.time()
    ck = Exl3Checkpoint(MODEL)
    head = eb.build_mtp(ck, model.args, rank=0, world=2)
    mx.eval(mx.ones(1))
    log(f"+draft head {time.time() - t0:.0f}s "
        f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")

# quick smoke: one prefill + one verify round on the subset
if os.environ.get("PU0_SMOKE", "1") == "1":
    from mlx_lm.models.deepseek_v41 import spec as SP
    prompt = json.load(open(HOME + "/p30_prompt_ids.json"))[:32]
    cache = model.make_cache(1, max_seq_len=256)
    am, taps = model(mx.array([prompt]), cache, last_logit_only=True,
                     return_taps=True, argmax=True)
    dsc = head.make_cache(1)
    tc = mx.concatenate([taps[L] for L in model.args.dspark_target_layer_ids], axis=-1)
    head.append_ctx(tc, dsc)
    nxt = am[:, -1].astype(mx.int32)
    mx.eval(nxt)
    t0 = time.time()
    d, conf = head.draft(nxt, model.embed, model.head, dsc, width=3)
    mx.eval(d)
    log(f"draft w3 {1000 * (time.time() - t0):.1f} ms tokens={d.tolist()} conf={conf.shape}")
    pos = cache.offset
    sn = SP.snap(cache, pos)
    vin = mx.concatenate([nxt.reshape(1, 1), d.astype(mx.int32)], axis=1)
    t0 = time.time()
    am2, taps2 = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(am2)
    log(f"verify R4 {1000 * (time.time() - t0):.1f} ms args={am2.tolist()}")
    SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
    log(f"rollback ok offset={cache.offset}")
    log(f"final active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU0_DONE")
