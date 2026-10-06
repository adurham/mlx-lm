# LOCAL ADDITION. Parity for the DSV41_SPARSE_PREFILL_CHUNK knob (attention.py).
"""Proves the prefill query-tile cap is pure batching:

* Test A -- ``sparse_attn`` output is BIT-IDENTICAL across chunk=64/256/2048
  (same ops per row; only the number of tiles and per-tile mx.eval fences move;
  the module docstring already establishes fenced/unfenced bit-identity).
* Test B -- the attention module's default stays 64 with the env unset (the
  shipped behavior), and follows the env when set (read at import).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import sparse_attention as sa


def _qkv(seed: int = 7):
    rng = np.random.default_rng(seed)
    b, m, h, d = 1, 96, 4, 64
    k = 40
    q = mx.array(rng.standard_normal((b, m, h, d)).astype(np.float32))
    n = 256
    kv = mx.array(rng.standard_normal((b, n, d)).astype(np.float32)).astype(mx.bfloat16)
    sink = mx.array(rng.standard_normal((h,)).astype(np.float32))
    idx = mx.array(rng.integers(0, n, size=(b, m, k)).astype(np.int32))
    idx[:, :, :5] = -1                      # exercise the masked slots
    return q, kv, sink, idx


def test_sparse_attn_chunk_invariance_bit_identical():
    q, kv, sink, idx = _qkv()
    outs = []
    for chunk in (64, 256, 2048):
        o = sa.sparse_attn(q, kv, sink, idx, 0.125, chunk=chunk)
        mx.eval(o)
        outs.append(np.array(o))
    assert np.array_equal(outs[0], outs[1]), "chunk 64 vs 256 diverged"
    assert np.array_equal(outs[0], outs[2]), "chunk 64 vs 2048 diverged"


def test_attention_default_chunk_is_256_and_follows_env():
    # default 256 since 2026-10-06 (shipped with the qtile=256 winner; bigger
    # tiles => fewer per-tile fences). 64 remains the escape-hatch value.
    from mlx_lm.models.deepseek_v41 import attention as at
    if os.environ.get("DSV41_SPARSE_PREFILL_CHUNK") is None:
        assert at._PREFILL_CHUNK == 256, "shipped default must be 256"
    # env-follows, in a clean interpreter
    r = subprocess.run(
        [sys.executable, "-c",
         "import mlx_lm.models.deepseek_v41.attention as A;"
         "print(A._PREFILL_CHUNK)"],
        capture_output=True, text=True,
        env={**os.environ, "DSV41_SPARSE_PREFILL_CHUNK": "256"},
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.stdout.strip() == "256", f"env not honored: {r.stdout!r} {r.stderr!r}"
