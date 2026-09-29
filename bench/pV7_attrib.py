#!/usr/bin/env python3
"""pV7 -- fine verify-marginal attribution, layer 20, non-degenerate.

Splits Block._fused_call into: hc, attn-EXL3-linears, attn-other, experts,
moe-glue, and times each with ONE eval per segment (same count at every R, so
deltas are meaningful). No stubs: real data everywhere.

Env: PV7_LAYERS (default 20), PV7_ROWS (single int), PV7_REPS, PV7_SEG=1 to
disable the extra EXL3 sub-timing inside attention.

Run:
  PV7_LAYERS=20 PV7_ROWS=1 EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
   lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV7_attrib.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV7_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A
from mlx_lm.models.deepseek_v41 import spec as SP
from mlx_lm.models.deepseek_v41.hc_fused import hc_expand, mixes_and_collapse

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV7_LAYERS", "20").split(",")]
R = int(os.environ.get("PV7_ROWS", "1"))
REPS = int(os.environ.get("PV7_REPS", "10"))
TAG = f"R={R}"


def log(*a):
    print(f"[pV7 {TAG}]", *a, flush=True)


SEGS = ("hc1", "attn", "hc2", "hc3", "ffn", "hc4")
_acc = {k: 0.0 for k in SEGS}
_sub = {}


def _seg(name, fn):
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    dt = time.perf_counter() - t0
    _acc[name] += dt
    return out


# --- sub-timing inside attention: one eval per EXL3 linear ------------------
def _wrap(mod, nm, attr="__call__"):
    fn = getattr(mod, attr)

    def w(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        mx.eval(r)
        dt = time.perf_counter() - t0
        _sub[nm] = _sub.get(nm, 0.0) + dt
        return r
    setattr(mod, attr, w)


_orig_fused_call = M.Block._fused_call


def timed_fused_call(self, x, pre_mix, start_pos, cache, shared):
    h = _seg("hc1", lambda: mixes_and_collapse(
        x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, pre_mix,
        self.hc_iters, self.norm_eps, self.hc_eps))
    h, attn_pre, attn_post, attn_comb = h

    def _attn():
        return self.attn(self.attn_norm(h), start_pos, cache, shared)
    a = _seg("attn", _attn)
    x = _seg("hc2", lambda: hc_expand(a, x, attn_post, attn_comb))
    h = _seg("hc3", lambda: mixes_and_collapse(
        x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, attn_pre,
        self.hc_iters, self.norm_eps, self.hc_eps))
    h, ffn_pre, ffn_post, ffn_comb = h

    def _ffn():
        return self.ffn(self.ffn_norm(h))
    y = _seg("ffn", _ffn)
    x = _seg("hc4", lambda: hc_expand(y, x, ffn_post, ffn_comb))
    return x, ffn_pre


M.Block._fused_call = timed_fused_call

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
mx.eval(model.parameters())
# sub-wrap every EXL3 module of the built layers
def walk(mod, path=""):
    for name, child in mod.children().items():
        p = f"{path}.{name}" if path else name
        yield p, child
        yield from walk(child, p)


for blk in model.layers:
    for p, child in walk(blk):
        if isinstance(child, (eb.Exl3Proj, eb.Exl3Member, eb.Exl3GroupedStack)):
            if getattr(child, "_pv7_wrapped", False):
                continue
            _wrap(child, f"L{blk.layer_id}." + p.split(".")[-1])
            child._pv7_wrapped = True
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

step_ms = []
snap = None
snapsub = None
for i in range(REPS):
    pos = cache.offset
    sn = SP.snap(cache, pos)
    vin = mx.array([[feed[(i + j) % len(feed)] for j in range(R)]], dtype=mx.int32)
    for k in SEGS:
        _acc[k] = 0.0
    t0 = time.perf_counter()
    out = model(vin, cache, return_taps=True, argmax=True)
    mx.eval(out[0])
    step_ms.append((time.perf_counter() - t0) * 1e3)
    if i == 3:
        snap = dict(_acc)
        snapsub = dict(_sub)
    SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
step = float(np.median(step_ms[3:]))
n = len(LAYERS)
log(f"step median {step:7.3f} ms ({n} layer)")
acc_ms = 0.0
for k in SEGS:
    ms = snap[k] / n * 1e3
    acc_ms += ms
    log(f"   {k:9s} {ms:8.3f} ms/layer")
log(f"   accounted {acc_ms:8.3f} ms/layer, unattributed {step/n - acc_ms:+.3f}")
log("   -- attn EXL3 linears by name (ms/layer) --")
for k, v in sorted(snapsub.items(), key=lambda kv: -kv[1]):
    log(f"      {k:22s} {v/n*1e3:7.3f}")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV7_DONE")
