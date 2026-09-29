#!/usr/bin/env python3
"""pU0c -- pick a <=8 GB config for the stream-U round harness.

Builds a body layer subset and/or the DSpark draft head with the expert count
capped (the only way a body subset + the head fits the 8 GB rule): the MoE
router is capped through ``args.n_routed_experts`` / ``args.dspark_n_experts``
and the loader is patched to match, so the model is internally consistent --
same classes, same slicing, same collective structure, fewer experts.

Env: PU0_CFG=body|head|both, PU0_LAYERS, PU0_BODY_EXPERTS, PU0_HEAD_EXPERTS,
     PU0_WORLD, PU0_PKG.
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

cfg = os.environ.get("PU0_CFG", "both")
layers = [int(x) for x in os.environ.get("PU0_LAYERS", "20,37").split(",") if x]
NB = os.environ.get("PU0_BODY_EXPERTS")
NE = os.environ.get("PU0_HEAD_EXPERTS")
WORLD = int(os.environ.get("PU0_WORLD", "1"))


def log(*a):
    print("[pU0c]", *a, flush=True)


# consistent expert cap: router size and loaded tables agree.
_real_load = eb.load_experts
cap = int(NB) if (NB and cfg in ("body", "both")) else None
cap_h = int(NE) if (NE and cfg in ("head", "both")) else None


def _patched(ckpt, layer_id, *, n_experts=None, prefix=None, **kw):
    if n_experts is None:
        if prefix and prefix.startswith("mtp.") and cap_h:
            n_experts = cap_h
        elif cap:
            n_experts = cap
    return _real_load(ckpt, layer_id, n_experts=n_experts, prefix=prefix, **kw)


eb.load_experts = _patched

mx.reset_peak_memory()
log(f"cfg={cfg} layers={layers} body_cap={cap} head_cap={cap_h} world={WORLD}")

if cfg in ("body", "both") and layers:
    ck = Exl3Checkpoint(MODEL)
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    args = ModelArgs.from_dict(ck.config)
    if cap:
        args.n_routed_experts = cap
    n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0
    t0 = time.time()
    model, rep = eb.build_model(MODEL, native_dir=NATIVE, layers=layers, rank=0,
                                world=WORLD)
    model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
    mx.eval(mx.ones(1))
    log(f"body {time.time() - t0:.0f}s active={mx.get_active_memory() / 1e9:.2f}GB "
        f"peak={mx.get_peak_memory() / 1e9:.2f}GB")

if cfg in ("head", "both"):
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    ck = Exl3Checkpoint(MODEL)
    args = ModelArgs.from_dict(ck.config)
    if cap_h:
        args.dspark_n_experts = cap_h
    n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0
    t0 = time.time()
    head = eb.build_mtp(ck, args, rank=0, world=WORLD)
    mx.eval(mx.ones(1))
    log(f"head {time.time() - t0:.0f}s active={mx.get_active_memory() / 1e9:.2f}GB "
        f"peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU0C_DONE")
