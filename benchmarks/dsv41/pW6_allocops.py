#!/usr/bin/env python3
"""pW6 -- which op allocates NEW device buffers during the post-prefill decode?

MLX_LOG_NEW_BUFFER_PATH logs every Metal device-level newBuffer (cache miss).
Wrapping each op and reading the log offset gives a per-op allocation census for
decode steps 0..3 after a prefill, plus the same for the "pool warm" steady
steps. That says exactly which tensors' shapes are unstable / pool-evicted.

Env: PW_LAYERS (2,20), PW_CTX (8192), PW_STEPS (4), PW_CLEAR (4|0)
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PW_PKG", HOME + "/dsv41-ws2/W"))
os.environ.setdefault("MLX_LOG_NEW_BUFFER_PATH", "/tmp/pW6_bufs.log")
import mlx.core as mx  # noqa: E402

BUFL = os.environ["MLX_LOG_NEW_BUFFER_PATH"]
CUR = ["?"]
ALLOC = {}


def _off():
    try:
        return os.path.getsize(BUFL)
    except OSError:
        return 0


def _delta(o0):
    with open(BUFL, "rb") as f:
        f.seek(o0)
        vals = [int(x) for x in f.read().split(b"\n") if x.strip()]
    return len(vals), sum(vals)


from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as A, indexer as IX, prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import model as M, moe as MO, hc_fused as HF  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PW_LAYERS", "2,20").split(",")]
CTX = int(os.environ.get("PW_CTX", "8192"))
STEPS = int(os.environ.get("PW_STEPS", "4"))
CLEAR = int(os.environ.get("PW_CLEAR", "4"))

TAGS = []
for name, mod, attr in (("RMSNorm", None, None),):
    pass


def wrap(tag, obj, attr):
    orig = getattr(obj, attr)

    def w(*a, **k):
        o0 = _off()
        t0 = time.perf_counter()
        r = orig(*a, **k)
        # only the host-side graph build happens here; allocation happens at eval
        _flush(tag, o0, (time.perf_counter() - t0) * 1e3)
        return r
    setattr(obj, attr, w)
    TAGS.append(tag)


def _flush(tag, o0, ms):
    c, b = _delta(o0)
    if c:
        d = ALLOC.setdefault(tag, [0, 0, 0.0])
        d[0] += c
        d[1] += b
        d[2] += ms


# eval-time attribution is what we want: wrap the *whole* step instead, and
# attribute allocations by wrapping each op with a deferred eval fence.
def wrap_eval(tag, obj, attr):
    """Fence after the op and attribute the allocations that landed."""
    orig = getattr(obj, attr)

    def w(*a, **k):
        o0 = _off()
        r = orig(*a, **k)
        outs = r if isinstance(r, (list, tuple)) else [r]
        mx.eval([t for t in outs if isinstance(t, mx.array)])
        _flush(tag, o0, 0.0)
        return r
    setattr(obj, attr, w)
    TAGS.append(tag)


wrap_eval("Attention", A.Attention, "__call__")
wrap_eval("Indexer", IX.Indexer, "__call__")
wrap_eval("Indexer.publish_keys", IX.Indexer, "publish_keys")
from mlx_lm.models.deepseek_v41 import compressor as CP  # noqa: E402
wrap_eval("Compressor", CP.Compressor, "__call__")
wrap_eval("sparse_attn", A, "sparse_attn")
wrap_eval("MoE", MO.MoE, "__call__")
wrap_eval("Gate", MO.Gate, "__call__")
wrap_eval("SharedExpert", MO.SharedExpert, "__call__")
wrap_eval("Exl3Experts", eb.Exl3Experts, "__call__")
wrap_eval("Exl3Proj", eb.Exl3Proj, "__call__")
wrap_eval("Exl3Member", eb.Exl3Member, "__call__")
wrap_eval("Exl3GroupedStack", eb.Exl3GroupedStack, "__call__")
wrap_eval("hc_mixes_and_collapse", HF, "mixes_and_collapse")
wrap_eval("hc_expand", HF, "hc_expand")
wrap_eval("rope_tail", A, "rope_tail")
from mlx_lm.models.deepseek_v41.layers import RMSNorm  # noqa: E402
wrap_eval("RMSNorm", RMSNorm, "__call__")
from mlx_lm.models.exl3.exl3_moe import EXL3SwitchGLU  # noqa: E402
try:
    wrap_eval("EXL3SwitchGLU", EXL3SwitchGLU, "__call__")
except Exception as e:  # noqa: BLE001
    print("[pW6] skip switchglu", e)

model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0, world=2, group=None)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
print(f"[pW6] layers={LAYERS} active={mx.get_active_memory()/1e9:.2f}GB", flush=True)
PF.warmup(model)

base = json.load(open(HOME + "/p30_prompt_ids.json"))
ids = (base * (CTX // len(base) + 1))[:CTX]
cache = model.make_cache(1, max_seq_len=CTX + 512)
for li, lc in enumerate(cache.layers):
    if li not in LAYERS:
        lc.comp_state = None

mx.reset_peak_memory()
t0 = time.perf_counter()
am = PF.prefill(model, ids, cache, last_logit_only=True, argmax=True,
                clear_cache_every=CLEAR)
mx.eval(am)
print(f"[pW6] prefill {CTX} in {time.perf_counter()-t0:.2f}s "
      f"clear_every={CLEAR} peak={mx.get_peak_memory()/1e9:.2f}GB", flush=True)

nxt = am.reshape(-1)[-1:].astype(mx.int32)
for i in range(STEPS):
    for k in ALLOC:
        ALLOC[k] = [0, 0, 0.0]
    o0 = _off()
    t0 = time.perf_counter()
    lg = model(nxt[None], cache, last_logit_only=True, argmax=True)
    tb = time.perf_counter() - t0
    d0 = mx.metal.dispatch_count()
    mx.eval(lg)
    te = time.perf_counter() - t0
    tot_c, tot_b = _delta(o0)
    print(f"[pW6] step{i} build {tb*1e3:7.2f}ms eval {(te-tb)*1e3:7.2f}ms "
          f"TOTAL newBuffer {tot_c} calls / {tot_b/1e6:.2f} MB", flush=True)
    for k, v in sorted(ALLOC.items(), key=lambda kv: -kv[1][1]):
        if v[0]:
            print(f"[pW6]     {k:26s} {v[0]:5d} calls {v[1]/1e6:8.3f} MB", flush=True)
    nxt = lg.reshape(-1)[-1:].astype(mx.int32)
print(f"[pW6] peak {mx.get_peak_memory()/1e9:.2f}GB active {mx.get_active_memory()/1e9:.2f}GB "
      f"cache {mx.get_cache_memory()/1e9:.2f}GB PW6_DONE", flush=True)
