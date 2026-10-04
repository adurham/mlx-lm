#!/usr/bin/env python3
"""pV18 -- where attention's per-row growth comes from (layer 20, single layer).

pV7 measured attention at 1.382 ms (R=1) -> 1.651 (R=4) -> 1.777 (R=6) for
layer 20, i.e. +0.395 ms of the window's marginal. This probe splits the 190
lines of Attention.__call__ into their real terms and sweeps R, so the growth
is attributed rather than guessed:

  q path      : wq_a -> q_norm -> wq_b -> rope_tail
  kv path     : wkv -> kv_norm -> rope -> fake_quant -> concat with window
  idx build   : window_idx_matrix + cidx concat (host-side index math)
  sparse_attn : the actual attention kernel over window+compressed
  out path    : rope inverse, wo_a grouped stack, wo_b

Timing method: one eval per segment, identical segment COUNT at every R, so the
R-to-R delta per segment is meaningful. Run one R per process.

Env: PV18_LAYERS (default 20), PV18_ROWS (int), PV18_REPS.
Run:
  PV18_PKG=~/dsv41-ws2/V PV18_LAYERS=20 PV18_ROWS=4 EXL3_MM_MAX_ROWS=100000 \
   MTL_DISABLE_TIMEOUT=1 lockf -k ~/dsv41-gpu.lock \
   ~/repos/exo/.venv/bin/python bench/pV18_attn.py
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("PV18_PKG", HOME + "/dsv41-ws2/V"))
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import spec as SP
from mlx_lm.models.deepseek_v41 import attention as A
from mlx_lm.models.deepseek_v41 import sparse_attention as SA
from mlx_lm.models.deepseek_v41.layers import rope_tail
from mlx_lm.models.deepseek_v41.fakequant import (
    fake_quant_fp4_e4m3, fake_quant_fp8_ue8m0)

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PV18_LAYERS", "20").split(",")]
R = int(os.environ.get("PV18_ROWS", "1"))
REPS = int(os.environ.get("PV18_REPS", "10"))

SEGS = ("qpath", "kvpath", "idx", "sparse", "outpath", "other")
_acc = {k: 0.0 for k in SEGS}


def log(*a):
    print(f"[pV18 R={R}]", *a, flush=True)


def _seg(name, fn):
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    _acc[name] += (time.perf_counter() - t0) * 1e3
    return out


def timed_attn(self, x, start_pos, cache, shared):
    bsz, n, _ = x.shape
    rd = self.rope_head_dim
    end_pos = start_pos + n
    # Per-call RoPE: query rows are exactly this forward's n positions,
    # computed on demand from the layer's [32] freq vector (no cached table).
    c_q, s_q = self._cos_sin(mx.arange(start_pos, end_pos))

    qr_box = {}

    def _q():
        qr_box["qr"] = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr_box["qr"]).reshape(bsz, n, self.n_heads, self.head_dim)
        return rope_tail(q, rd, c_q, s_q)
    q = _seg("qpath", _q)
    qr = qr_box["qr"]

    def _kv():
        kv = self.kv_norm(self.wkv(x))
        kv = rope_tail(kv, rd, c_q, s_q)
        return fake_quant_fp8_ue8m0(kv, 32)
    kv = _seg("kvpath", _kv)

    lc = cache.layers[self.layer_id]
    prev = lc.window_chrono(start_pos)
    wp = prev.shape[1]
    kv_all = mx.concatenate([prev.astype(kv.dtype), kv], axis=1) if wp else kv
    idxs = mx.broadcast_to(A.window_idx_matrix(wp, n, self.window_size)[None],
                           (bsz, n, min(self.window_size, wp + n)))
    mx.eval(idxs)
    lc.write_window(start_pos, kv)
    offset = wp + n

    if self.ratio:
        src = shared.kv_src_cache
        compress_len = end_pos // self.ratio
        latents = None
        if self.is_kv_source:
            latents = self.compressor(x, start_pos, lc.comp_state)
            shared.kv_src_cache = src = lc
        if self.is_index_source:
            if self.indexer.owns_k:
                if latents is not None:
                    self.indexer.publish_keys(latents, start_pos, self._freqvec, lc)
                shared.index_src_cache = lc
            if compress_len == 0:
                cidx = mx.zeros((bsz, n, 0), dtype=mx.int32)
            else:
                index_k = shared.index_src_cache.index_k[:bsz, :compress_len]
                cidx = self.indexer(x, qr, start_pos, offset, self._freqvec,
                                    index_k, shared)
            shared.topk_idxs = cidx
        else:
            cidx = shared.topk_idxs
        if latents is not None:
            g0 = start_pos // self.ratio
            g = latents.shape[1]
            pos = (g0 + mx.arange(g)) * self.ratio
            l_cos, l_sin = self._cos_sin(pos)
            latents = rope_tail(latents, rd, l_cos, l_sin)
            latents = fake_quant_fp4_e4m3(latents, 16)
            lc.comp_kv[:bsz, g0:g0 + g] = latents.astype(lc.dtype)
        if compress_len:
            def _idx():
                comp = src.comp_kv[:bsz, :compress_len].astype(kv.dtype)
                kva = mx.concatenate([kv_all, comp], axis=1)
                return mx.concatenate([idxs, cidx], axis=-1), kva
            idxs, kv_all = _seg("idx", _idx)

    sink = mx.zeros_like(self.attn_sink) if self._break_sink else self.attn_sink
    o = _seg("sparse", lambda: SA.sparse_attn(q, kv_all, sink, idxs, self.softmax_scale))

    def _out():
        oo = o
        if not self._break_rope_inverse:
            oo = rope_tail(oo, rd, c_q, s_q, inverse=True)
        oo = oo.reshape(bsz, n, self.n_groups, -1)
        if isinstance(self.wo_a, nn.Linear):
            wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)
            oo = mx.einsum("bsgd,grd->bsgr", oo.astype(mx.float32), wa.astype(mx.float32))
        else:
            oo = self.wo_a(oo)
        return self.wo_b(oo.reshape(bsz, n, -1).astype(x.dtype))
    out = _seg("outpath", _out)
    return out


import mlx.nn as nn  # noqa: E402

A.Attention.__call__ = timed_attn

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

step_ms = []
snap = None
for i in range(REPS):
    pos = cache.offset
    sn = SP.snap(cache, pos)
    vin = mx.array([[feed[(i + j) % len(feed)] for j in range(R)]], dtype=mx.int32)
    for k in SEGS:
        _acc[k] = 0.0
    t0 = time.perf_counter()
    out = model(vin, cache, argmax=True)
    mx.eval(out)
    step_ms.append((time.perf_counter() - t0) * 1e3)
    if i == 3:
        snap = dict(_acc)
    SP.rollback(cache, sn, pos + 1, SP.stashes(cache))
step = float(np.median(step_ms[3:]))
n = len(LAYERS)
log(f"step median {step:7.3f} ms ({n} layer)")
tot = 0.0
for k in SEGS:
    ms = snap[k] / n
    tot += ms
    log(f"   {k:8s} {ms:8.3f} ms/layer  ({ms/step*100:5.1f}%)")
log(f"   attention accounted {tot:.3f} of {step:.3f} ms step "
    f"({tot/step*100:.1f}%; the rest is the MoE + hyper-connections)")
log(f"peak={mx.get_peak_memory()/1e9:.2f}GB")
log("PV18_DONE")
