#!/usr/bin/env python3
"""pV13 -- R=1 bit-identity baseline + verify-window parity, 2 layers.

Saves the exact logits a verify-shaped forward produces at R=1..4 and at R=1
with the MoE going through _decode_fused2 vs the unfused _decode path, so any
kernel change can be checked as "R=1 bit-identical, R>=2 cos >= 0.9998".

Run (production down, 2 layers, ~7 GB):
  PV13_PKG=~/dsv41-ws2/V PV13_TAG=before EXL3_MM_MAX_ROWS=100000 \
   MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
   ~/repos/exo/.venv/bin/python bench/pV13_parity.py
Then re-run with PV13_TAG=after PV13_CHECK=~/pV13_before.npz
"""
import json, os, sys
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV13_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import spec as SP

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV13_LAYERS", "20,21").split(",")]
TAG = os.environ.get("PV13_TAG", "run")
CHECK = os.environ.get("PV13_CHECK")
ROWS = [int(x) for x in os.environ.get("PV13_ROWS", "1,2,3,4").split(",")]


def log(*a):
    print(f"[pV13 {TAG}]", *a, flush=True)


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                          world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
log(f"built layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB")

allids = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = allids[:64]
feed = allids[64:64 + 40]
cache = model.make_cache(1, max_seq_len=len(ids) + 256)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
model(mx.array([ids]), cache, last_logit_only=True, argmax=True)
mx.eval(mx.zeros(1))
pos0 = cache.offset
log(f"prefill OK offset={pos0}")

outs = {}
for R in ROWS:
    cache.offset = pos0
    lg = None
    for i in range(4):
        pos = cache.offset
        sn = SP.snap(cache, pos)
        vin = mx.array([[feed[(i + j) % len(feed)] for j in range(R)]], dtype=mx.int32)
        lg = model(vin, cache, logits_only=True) if False else model(vin, cache)
        mx.eval(lg)
        SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
    outs[f"R{R}"] = np.array(lg[0].astype(mx.float32))
    log(f"  R={R}: captured logits {outs[f'R{R}'].shape}")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")

if CHECK:
    ref = dict(np.load(CHECK))
    for k in sorted(ref):
        if k not in outs:
            continue
        a, b = ref[k], outs[k]
        if a.shape != b.shape:
            log(f"  {k}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        exact = bool(np.array_equal(a, b))
        cos = float((a * b).sum(-1) /
                    (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30))
        d = float(np.abs(a - b).max())
        am = float((a.argmax(-1) == b.argmax(-1)).mean())
        log(f"  {k}: exact={exact} cos={cos:.7f} max|d|={d:.4g} "
            f"argmax_agree={am*100:.1f}%  {'<<< MUST BE EXACT' if k=='R1' else ''}")
else:
    p = HOME + f"/pV13_{TAG}.npz"
    np.savez(p, **outs)
    log(f"saved {p}")
log("PV13_DONE")
