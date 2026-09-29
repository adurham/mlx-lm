#!/usr/bin/env python3
"""pV2 -- per-op census of the marginal verify row on a layer subset.

Runs p48's build (P48_PKG tree) with P48_ROWS=1 and 4, wrapping the EXL3
pieces so each reports its own cumulative GPU time via mx.metal.dispatch_count
bracketing... simpler: bracket with mx.eval + perf_counter around each wrapped
call (they are leaves; the harness already evaluates the full step). We instead
time each wrapped call in isolation with an explicit sync, then report per-op
ms for R=1 and R=4 and the delta. The step total is also printed from the
unwrapped path so the ablation numbers can be sanity-checked.

Run:
  P48_PKG=~/dsv41-ws2/V P48_ROWS=1 PV2_MODE=census \
    EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
    ~/repos/exo/.venv/bin/python bench/pV2_census.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P48_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A
from mlx_lm.models.deepseek_v41 import spec as SP

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV2_LAYERS", "20,21").split(",")]
STUB = os.environ.get("PV2_STUB", "none")


def log(*a):
    print("[pV2]", *a, flush=True)


model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
log(f"built layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB stub={STUB}")

# ---- per-op timing wrapper: explicit sync around each leaf call -------------
TIMING = os.environ.get("PV2_TIMING", "1") == "1"
stats = {}
_cur = {"keys": []}


def wrap(mod, name, attr="__call__"):
    fn = getattr(mod, attr)

    def w(*a, **k):
        if not TIMING:
            return fn(*a, **k)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        outs = r if isinstance(r, (tuple, list)) else [r]
        ev = [t for t in outs if isinstance(t, mx.array)]
        t1 = time.perf_counter()
        if ev:
            mx.eval(ev)
        t2 = time.perf_counter()
        key = (name, _cur["keys"][-1] if _cur["keys"] else "")
        e = stats.setdefault(key, {"n": 0, "ms": 0.0, "ms_eval": 0.0})
        e["n"] += 1
        e["ms"] += (t1 - t0) * 1e3
        e["ms_eval"] += (t2 - t1) * 1e3
        return r
    setattr(mod, attr, w)


if TIMING:
    wrap(MO.MoE, "MoE(total)")
    wrap(eb.Exl3Experts, "Exl3Experts")
    wrap(eb.Exl3Proj, "Exl3Proj(single)")
    wrap(eb.Exl3Member, "Exl3Member(fused grp)")
    wrap(eb.Exl3GroupedStack, "wo_a stack")
    wrap(eb.Exl3FusedGroup, "FusedGroup.call", "run_stacked")
    from mlx_lm.models.deepseek_v41 import layers as LY
    wrap(LY.RMSNorm, "RMSNorm")
    from mlx_lm.models.deepseek_v41 import sparse_attention as SA
    from mlx_lm.models.deepseek_v41 import indexer as IX
    from mlx_lm.models.deepseek_v41 import compressor as CP
    from mlx_lm.models.deepseek_v41 import engram as EN
    from mlx_lm.models.deepseek_v41 import hc_fused as HF
    wrap(SA, "sparse_attn", "sparse_attn")
    wrap(IX.Indexer, "Indexer(total)")
    wrap(CP.Compressor, "Compressor(total)")
    wrap(EN.Engram, "Engram")
    wrap(M, "mixes_and_collapse", "mixes_and_collapse")
    wrap(M, "hc_expand", "hc_expand")
    for L in LAYERS:
        _cur["keys"].append(f"L{L}")

if STUB == "experts":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[:, None, :] * 0, idx.shape + (x.shape[-1],))
    log("stubbed experts")
elif STUB == "dense":
    def _z(x, n):
        return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (n,)).astype(x.dtype)
    eb.Exl3Proj.__call__ = lambda self, x: _z(x, self._lin.out_features)
    eb.Exl3Member.__call__ = lambda self, x: _z(x, self.out_features)
    eb.Exl3GroupedStack.__call__ = lambda self, x: _z(x, self._g.outs[0])
    log("stubbed dense EXL3")
elif STUB == "both":
    eb.Exl3Experts.__call__ = lambda self, x, idx: mx.broadcast_to(x[:, None, :] * 0, idx.shape + (x.shape[-1],))
    def _z(x, n):
        return mx.broadcast_to(x[..., :1] * 0, x.shape[:-1] + (n,)).astype(x.dtype)
    eb.Exl3Proj.__call__ = lambda self, x: _z(x, self._lin.out_features)
    eb.Exl3Member.__call__ = lambda self, x: _z(x, self.out_features)
    eb.Exl3GroupedStack.__call__ = lambda self, x: _z(self._g.outs[0]).astype(x.dtype)
    log("stubbed experts + dense")

allids = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = allids[:64]
feed = allids[64:64 + 128]
cache = model.make_cache(1, max_seq_len=len(ids) + 128 * 8 + 16)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None
logits = model(mx.array([ids]), cache, last_logit_only=True)
mx.eval(logits)
log("prefill done")

RES = {}
for R in [int(x) for x in os.environ.get("PV2_ROWS", "1,4").split(",")]:
    steps = []
    for i in range(int(os.environ.get("PV2_STEPS", "12"))):
        pos = cache.offset
        sn = SP.snap(cache, pos)
        chunk = [feed[(i + j) % len(feed)] for j in range(R)]
        s0 = time.perf_counter()
        lg = model(mx.array([chunk]), cache)
        mx.eval(lg)
        steps.append((time.perf_counter() - s0) * 1e3)
        st = SP.stashes(cache)
        SP.rollback(cache, sn, pos + 1, st)
        mx.eval([lc.comp_state.kv_state for lc in cache.layers if lc.comp_state is not None])
    RES[f"R{R}"] = {"median_ms": float(np.median(steps[3:]))}
    log(f"R={R}: {np.median(steps[3:]):.3f} ms/step (all {[f'{x:.1f}' for x in steps]})")

if TIMING:
    log("per-op cumulative ms/step for this build:")
    tot = 0.0
    for (nm, lk), e in sorted(stats.items(), key=lambda kv: -(kv[1]["ms"] + kv[1]["ms_eval"])):
        per = (e["ms"] + e["ms_eval"]) / max(e["n"] // 2, 1)
        tot += 0.0
        log(f"   {nm:24s} {lk:5s} n={e['n']//2:4d} per-call graph {e['ms']/max(e['n']//2,1):7.3f} "
            f"eval {e['ms_eval']/max(e['n']//2,1):7.3f} total {per:8.3f} ms")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV2_DONE")
