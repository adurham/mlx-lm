#!/usr/bin/env python3
"""pV6 -- phase-accurate marginal cost of a verify row (single layer, <=8 GB).

Instead of per-op wrappers (whose inner mx.eval flushes upstream work and
over-attributes), this harness evaluates at explicit phase boundaries inside
Block._fused_call: mixes / attn / hc_expand / mixes2 / FFN(MoE) / hc_expand.
One sync per boundary, identical count at every R, so the DELTA between R=1
and R=k per phase is the true marginal cost of the extra rows.

Run per R (one process each, so nothing is shared/cached across shapes):
  PV6_LAYERS=20 PV6_ROWS=1 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
    lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV6_phases.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV6_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A
from mlx_lm.models.deepseek_v41 import spec as SP
from mlx_lm.models.deepseek_v41.hc_fused import hc_expand, mixes_and_collapse

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV6_LAYERS", "20").split(",")]
R = int(os.environ.get("PV6_ROWS", "1"))
REPS = int(os.environ.get("PV6_REPS", "12"))
STUB = os.environ.get("PV6_STUB", "none")


def log(*a):
    print(f"[pV6 R={R} stub={STUB}]", *a, flush=True)


if STUB in ("dense", "both"):
    def _z(x, n):
        return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (n,)).astype(x.dtype)
    eb.Exl3Proj.__call__ = lambda self, x: _z(x, self._lin.out_features)
    eb.Exl3Member.__call__ = lambda self, x: _z(x, self.out_features)
    eb.Exl3GroupedStack.__call__ = lambda self, x: _z(x, self._g.outs[0])
if STUB in ("experts", "both"):
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(
        x[:, None, :] * 0, idx.shape + (x.shape[-1],))


PHASES = ("mix1", "attn", "expand1", "mix2", "ffn", "expand2", "head")


def timed_fused_call(self, x, pre_mix, start_pos, cache, shared):
    import time as _t
    acc = _cur["acc"]

    def seg(name, fn):
        t0 = _t.perf_counter()
        out = fn()
        mx.eval(out)
        acc[name] += _t.perf_counter() - t0
        return out

    h = seg("mix1", lambda: mixes_and_collapse(
        x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix,
        self.hc_iters, self.norm_eps, self.hc_eps))
    h, attn_pre, attn_post, attn_comb = h
    h = seg("attn", lambda: self.attn(self.attn_norm(h), start_pos, cache, shared))
    x = seg("expand1", lambda: hc_expand(h, x, attn_post, attn_comb))
    h = seg("mix2", lambda: mixes_and_collapse(
        x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre,
        self.hc_iters, self.norm_eps, self.hc_eps))
    h, ffn_pre, ffn_post, ffn_comb = h
    h = seg("ffn", lambda: self.ffn(self.ffn_norm(h)))
    x = seg("expand2", lambda: hc_expand(h, x, ffn_post, ffn_comb))
    return x, ffn_pre


_cur = {"acc": {k: 0.0 for k in PHASES}}
M.Block._fused_call = timed_fused_call

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
mx.eval(model.parameters())
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

n = len(LAYERS)
step_ms, heads = [], []
for i in range(REPS):
    pos = cache.offset
    sn = SP.snap(cache, pos)
    vin = mx.array([[feed[(i + j) % len(feed)] for j in range(R)]], dtype=mx.int32)
    for k in PHASES:
        _cur["acc"][k] = 0.0
    s0 = time.perf_counter()
    out = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(out[0])
    step_ms.append((time.perf_counter() - s0) * 1e3)
    heads.append((_cur["acc"]["head"], 0.0))
    SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
    if i == 3:
        _snap = {k: v for k, v in _cur["acc"].items()}
step = float(np.median(step_ms[3:]))
# _snap values are SECONDS (perf_counter deltas); `step` is MILLISECONDS.
snap_ms = {k: v * 1e3 for k, v in _snap.items()}
log(f"step median {step:7.3f} ms over layers={LAYERS}")
for k in PHASES:
    if snap_ms[k] > 0:
        log(f"   {k:8s} {snap_ms[k]/n:8.3f} ms/layer   ({snap_ms[k]/step*100:5.1f}% of step)")
tot = sum(snap_ms.values())
log(f"   accounted {tot/step*100:.1f}% of step ({tot:.3f} of {step:.3f} ms); "
    f"unattributed {(step-tot)/n:+.3f} ms/layer")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV6_DONE")
