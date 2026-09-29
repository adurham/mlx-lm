#!/usr/bin/env python3
"""pU0f -- memory of the FULL single-node model slice (stream U).

Builds all 40 layers at rank 0 / world 2 (half-width experts, no collectives --
the same geometry p48_bench uses) plus the draft head, and prints active/peak so
the round harness can run at full depth inside the node limit.

Env: PU0F_BODY_EXPERTS (cap, default none), PU0F_HEAD_EXPERTS, PU0F_PKG.
NOTE: ``mx.eval`` is MLX's graph evaluation, not python's builtin.
"""
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
PKG = os.environ.get("PU0F_PKG", HOME + "/dsv41-ws2/U")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "bench"))

import mlx.core as mx  # noqa: E402

import pU_cap  # noqa: E402

b = os.environ.get("PU0F_BODY_EXPERTS")
h = os.environ.get("PU0F_HEAD_EXPERTS")
pU_cap.apply_caps(int(b) if b else None, int(h) if h else None)

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"

mx.reset_peak_memory()
log = lambda *a: print("[pU0f]", *a, flush=True)

t0 = time.time()
model, _ = eb.build_model(MODEL, native_dir=NATIVE, rank=0, world=2)
mx.eval(mx.ones(1))
log(f"body 40 layers (cap={b or 'none'}) {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")

ck = Exl3Checkpoint(MODEL)
t0 = time.time()
head = eb.build_mtp(ck, model.args, rank=0, world=2)
mx.eval(mx.ones(1))
log(f"+draft head (cap={h or 'none'}) {time.time() - t0:.0f}s "
    f"active={mx.get_active_memory() / 1e9:.2f}GB peak={mx.get_peak_memory() / 1e9:.2f}GB")
log("PU0F_DONE")
