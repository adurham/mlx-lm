#!/usr/bin/env python3
"""pF0 -- before/after greedy parity for the sampling change.

Runs the SAME greedy spec loop twice with two different trees on sys.path:

  base -> the pre-sampling commit (spec.generate as it was)
  new  -> the working tree (spec.generate with the temperature dispatch)

Same prompt, same layer subset, same seed of the world; the greedy path is
supposed to be *identical*: same tokens, and the only difference in stats is the
extra keys. The comparison itself happens on the driver side (the two runs get
different process trees), so this script just prints one canonical result line.

Usage (m4-1, one GPU job at a time):
    EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 PF0_PKG=<tree> PF0_TAG=base \\
      lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python pF0_greedy_parity.py
"""
import hashlib
import json
import os
import sys

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ["PF0_PKG"])
tag = os.environ.get("PF0_TAG", "?")
print(f"[pF0 {tag}] pkg={os.environ['PF0_PKG']} "
      f"spec={os.path.abspath(sys.modules['mlx_lm'].__file__ if 'mlx_lm' in sys.modules else '')}",
      flush=True)
import mlx.core as mx  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
N = int(os.environ.get("PF0_N", "32"))
LAYERS = [int(x) for x in os.environ.get("PF0_LAYERS", "0,1,2,3,37,38,39").split(",")]

from mlx_lm.models.deepseek_v41 import exl3_build as eb   # noqa: E402
from mlx_lm.models.deepseek_v41 import spec as SP         # noqa: E402
from mlx_lm.models.exl3.loader import Exl3Checkpoint      # noqa: E402

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=1)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
ck = Exl3Checkpoint(MODEL)
head = eb.build_mtp(ck, model.args, rank=0, world=1)
mx.eval(mx.ones(1))

_orig = model.make_cache


def _mc(*a, **k):
    c = _orig(*a, **k)
    for li, lc in enumerate(c.layers):
        if li not in LAYERS:
            lc.comp_state = None
    return c


model.make_cache = _mc

allids = json.load(open(HOME + "/p30_prompt_ids.json"))
prompt = allids[:64]

outs = []
for rep in range(2):
    out, stats = SP.generate(model, head, prompt, N, gamma=3, adaptive=False)
    outs.append(out)
    print(f"[pF0 {tag}] rep{rep}: {len(out)-1} tok ms_round={stats['ms_round']:.1f} "
          f"acc={stats['mean_acc']:.3f} stats={sorted(stats.keys())}", flush=True)

digest = hashlib.sha256(bytes(str(outs[0]).encode())).hexdigest()[:16]
print(f"[pF0 {tag}] reproducible_within_run={outs[0] == outs[1]} "
      f"sha256[:16]={digest} tokens={outs[0][:24]}", flush=True)
print(f"[pF0 {tag}] mem active={mx.get_active_memory()/1e9:.2f}GB "
      f"peak={mx.get_peak_memory()/1e9:.2f}GB PF0_DONE", flush=True)
