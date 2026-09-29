#!/usr/bin/env python3
"""pU0m -- memory probe: head-only / per-layer / per-subset peaks (stream U).

Prints active+peak after each build step so the harness subset can be chosen to
fit the 8 GB cap. Env: PU0M_MODE=head|layers|both, PU0M_LAYERS, PU0M_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PU0M_PKG", HOME + "/dsv41-ws2/U"))

import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"


def log(*a):
    print("[pU0m]", *a, flush=True)


mode = os.environ.get("PU0M_MODE", "both")
layers = [int(x) for x in os.environ.get("PU0M_LAYERS", "20,37").split(",") if x]
WORLD = int(os.environ.get("PU0M_WORLD", "2"))

mx.reset_peak_memory()
log(f"mode={mode} layers={layers} world={WORLD}")
if mode in ("layers", "both") and layers:
    t0 = time.time()
    model, rep = eb.build_model(MODEL, native_dir=NATIVE, layers=layers, rank=0,
                                world=WORLD)
    model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
    mx.eval(mx.ones(1))
    log(f"body {time.time() - t0:.0f}s active={mx.get_active_memory() / 1e9:.2f}GB "
        f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
if mode in ("head", "both"):
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    ck0 = Exl3Checkpoint(MODEL)
    args = ModelArgs.from_dict(ck0.config)
    n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0
    t0 = time.time()
    ck = Exl3Checkpoint(MODEL)
    head = eb.build_mtp(ck, args, rank=0, world=WORLD)
    mx.eval(mx.ones(1))
    log(f"head {time.time() - t0:.0f}s active={mx.get_active_memory() / 1e9:.2f}GB "
        f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU0M_DONE")
