# Copyright © 2026 Adam Durham (hermes-gw)
"""Gate parity for the two C1/C3 sparse-attention optimizations (both default OFF).

C1 -- column-partitioned two-source gather (``DSV41_SPARSE_COLSPLIT``). With the
gate ON, ``sparse_attn`` routes each key tile's columns to ONE source: columns of
the window block (``< split``) gather only from ``kv``, top-k columns
(``>= split``) only from ``kv2``, and a straddling tile gathers the two column
ranges separately -- no ``mx.where`` select. The boundary is derived from the
indices once per call; on any invariant violation the call falls back to the
exact where-select path.

C3 -- async per-tile fence (``DSV41_SPARSE_ASYNC_FENCE``). With the gate ON the
per-tile ``mx.eval`` becomes ``mx.async_eval`` (queue, no host round-trip) plus
ONE blocking ``mx.eval`` after the final tile, preserving the evaluated-return
contract while removing the per-tile syncs.

Pinned here:
1. both gates default OFF;
2. C1 OFF + colsplit passed == the call without colsplit (gate is inert);
3. C1 ON == OFF, bit-exact, on realistic idxs (window block ``< split`` first,
   top-k block ``>= split`` after, ``-1`` masks in both) across split sizes,
   dtypes, query/key tiling and the reference impl;
4. the invariant-violation fallback: a crafted idxs with a window-space value
   inside the top-k block proves ``_column_boundary`` returns ``None`` and the
   fallback output equals the where-select output (NOT a mis-gather);
5. C3 ON == OFF, bit-exact, across chunk=64/256 and every fence variant;
6. all four C1xC3 combinations agree bit-for-bit;
7. the real ``Attention.__call__`` path is bit-exact with the gates ON.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_sparse_opt_gates.py -q
"""
from __future__ import annotations

import itertools
import os
import zlib

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import sparse_attention as sa

D = 512          # head_dim
M = 48           # query rows
K = 40           # topk width
H = 8            # heads for the sink

_GATES = ("_COLSPLIT", "_FENCE_ASYNC")
_KNOBS = ("_QTILE", "_KTILE", "_WDTYPE", "_IMPL", "_COMPILE", "_BUDGET_BYTES",
          "_FENCE", "_FENCE_MIN_ROWS")


def _bits(x: mx.array) -> bytes:
    """Bit view: bf16/fp16 through uint16, everything else as numpy."""
    if x.dtype in (mx.bfloat16, mx.float16):
        return np.array(x.view(mx.uint16)).tobytes()
    return np.array(x).tobytes()


def _rand(rng, shape, dtype):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


def _realistic_idx(rng, split, n2, nwin, b=1, m=M, k=K, break_invariant=False):
    """idxs laid out exactly like attention.py: [window (<split) | topk (>=split)].

    Window columns ``[0, nwin)`` take values in ``[0, split)`` (``-1`` padding
    allowed), top-k columns ``[nwin, k)`` take values in ``[split, split+n2)``
    (``-1`` masks allowed). ``break_invariant`` plants a window-space value in
    the top-k block to force the fallback.
    """
    idx = np.zeros((b, m, k), np.int32)
    idx[:, :, :nwin] = rng.integers(0, max(split, 1), size=(b, m, nwin))
    idx[:, :, nwin:] = rng.integers(split, split + n2, size=(b, m, k - nwin))
    # masks: the indexer emits -1 in both regions
    idx[rng.random((b, m, k)) < 0.15] = -1
    # styled rows (space permitting): all-masked, all-window, all-topk
    if m >= 3:
        idx[:, m - 3, :] = -1
        idx[:, m - 2, :nwin] = rng.integers(0, max(split, 1), size=(b, nwin))
        idx[:, m - 2, nwin:] = -1
        idx[:, m - 1, :nwin] = -1
        idx[:, m - 1, nwin:] = rng.integers(split, split + n2, size=(b, k - nwin))
    # boundary values on row 0
    if k > nwin:
        idx[:, 0, nwin - 1] = split - 1
        idx[:, 0, nwin] = split
    if break_invariant and k > nwin:
        idx[:, 0, nwin + 1] = min(3, split - 1)      # real window value misplaced
    return mx.array(idx)


@pytest.fixture(autouse=True)
def _restore_gates():
    saved = {n: getattr(sa, n) for n in _GATES + _KNOBS}
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(sa, n, v)


def _set(*, colsplit=False, async_fence=False, qtile=64, ktile=0,
         wdtype="auto", impl="tiled", compile_=True, fence="qtile"):
    sa._COLSPLIT, sa._FENCE_ASYNC = colsplit, async_fence
    sa._QTILE, sa._KTILE, sa._WDTYPE = qtile, ktile, wdtype
    sa._IMPL, sa._COMPILE, sa._FENCE = impl, compile_, fence


def _run(q, kv1, kv2, sink, idx, split, nwin, *, chunk=64, colsplit_param=True):
    return sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=chunk, kv2=kv2,
                          split=split, colsplit=(nwin if colsplit_param else None))


# -- 1. defaults -------------------------------------------------------------
def test_gates_default_off():
    if os.environ.get("DSV41_SPARSE_COLSPLIT") is None:
        assert sa._COLSPLIT is False, "C1 gate must default OFF"
    if os.environ.get("DSV41_SPARSE_ASYNC_FENCE") is None:
        assert sa._FENCE_ASYNC is False, "C3 gate must default OFF"


def test_gates_follow_env_at_import():
    """Each gate is read at import (mirrors the file's own os.environ pattern)."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import mlx_lm.models.deepseek_v41.sparse_attention as S;"
        "import mlx_lm.models.deepseek_v41.attention as A;"
        "print(int(S._COLSPLIT), int(S._FENCE_ASYNC), int(A._COLSPLIT))")
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={**os.environ, "DSV41_SPARSE_COLSPLIT": "1",
             "DSV41_SPARSE_ASYNC_FENCE": "1"},
        cwd=root)
    assert r.stdout.strip() == "1 1 1", f"gates not honored: {r.stdout!r} {r.stderr!r}"


# -- 2. C1 gate OFF is inert ------------------------------------------------
@pytest.mark.parametrize("split,n2,nwin", [(60, 20, 16), (6, 9, 4), (25, 25, 8)])
def test_c1_gate_off_ignores_colsplit(split, n2, nwin):
    """Gate OFF: passing colsplit must be byte-identical to passing None."""
    _set(colsplit=False)
    rng = np.random.default_rng(zlib.crc32(f"off{split}".encode()))
    kv1, kv2 = _rand(rng, (1, split, D), mx.bfloat16), _rand(rng, (1, n2, D), mx.bfloat16)
    q = _rand(rng, (1, M, H, D), mx.bfloat16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin)
    a = _run(q, kv1, kv2, sink, idx, split, nwin, colsplit_param=True)
    b = _run(q, kv1, kv2, sink, idx, split, nwin, colsplit_param=False)
    mx.eval(a, b)
    assert _bits(a) == _bits(b), "gate OFF changed output when colsplit was passed"


# -- 3. C1 ON vs OFF bit-exact ----------------------------------------------
@pytest.mark.parametrize("split,n2,nwin,ktile,qtile,dtype", [
    (60, 20, 16, 0, 64, mx.bfloat16),
    (60, 20, 16, 7, 8, mx.bfloat16),     # straddling key tiles + many query tiles
    (60, 20, 16, 3, 8, mx.bfloat16),     # tile boundary well inside the window block
    (6, 9, 4, 0, 64, mx.bfloat16),
    (1, 30, 1, 0, 64, mx.bfloat16),      # window block width 1
    (25, 25, 8, 0, 64, mx.float16),
    (K - 1, 12, 16, 0, 64, mx.bfloat16),
])
def test_c1_on_off_bit_exact(split, n2, nwin, ktile, qtile, dtype):
    rng = np.random.default_rng(zlib.crc32(f"c1{split}/{n2}/{nwin}/{ktile}".encode()))
    kv1, kv2 = _rand(rng, (1, split, D), dtype), _rand(rng, (1, n2, D), dtype)
    q = _rand(rng, (1, M, H, D), dtype)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin)
    _set(colsplit=False, ktile=ktile, qtile=qtile)
    off = _run(q, kv1, kv2, sink, idx, split, nwin)
    _set(colsplit=True, ktile=ktile, qtile=qtile)
    on = _run(q, kv1, kv2, sink, idx, split, nwin)
    mx.eval(off, on)
    assert on.shape == off.shape and on.dtype == off.dtype
    assert _bits(off) == _bits(on), f"C1 not bit-exact ({split},{n2},{nwin},kt={ktile})"


def test_c1_boundary_derived_matches_declared():
    """The derived (colsplit=-1) boundary equals the declared window width."""
    rng = np.random.default_rng(5)
    idx = _realistic_idx(rng, 60, 20, 16)
    assert sa._column_boundary(idx, 60, 16) == 16
    assert sa._column_boundary(idx, 60, -1) == 16


def test_c1_boundary_inside_window_block_is_rejected():
    """A declared width short of the real boundary leaves window values in the
    right region -> the provenance check must refuse it."""
    rng = np.random.default_rng(6)
    idx = _realistic_idx(rng, 60, 20, 16)
    assert sa._column_boundary(idx, 60, 8) is None


# -- 4. invariant-violation fallback ----------------------------------------
@pytest.mark.parametrize("ktile", [0, 7])
def test_c1_fallback_on_invariant_violation(ktile):
    """Crafted idxs with a window value in the top-k block -> exact fallback."""
    rng = np.random.default_rng(zlib.crc32(f"violate{ktile}".encode()))
    split, n2, nwin = 60, 20, 16
    kv1, kv2 = _rand(rng, (1, split, D), mx.bfloat16), _rand(rng, (1, n2, D), mx.bfloat16)
    q = _rand(rng, (1, M, H, D), mx.bfloat16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin, break_invariant=True)
    # the declared boundary is now provably unclean -> fall back
    assert sa._column_boundary(idx, split, nwin) is None
    assert sa._column_boundary(idx, split, -1) is None
    _set(colsplit=True, ktile=ktile)
    got = _run(q, kv1, kv2, sink, idx, split, nwin)
    _set(colsplit=False, ktile=ktile)
    ref = _run(q, kv1, kv2, sink, idx, split, nwin)
    mx.eval(got, ref)
    assert _bits(got) == _bits(ref), "fallback did not reproduce the where-select output"


def test_c1_fallback_matches_dense_concat_reference():
    """The fallback output equals the legacy dense-concat semantics bit-for-bit."""
    rng = np.random.default_rng(11)
    split, n2, nwin = 60, 20, 16
    kv1, kv2 = _rand(rng, (1, split, D), mx.bfloat16), _rand(rng, (1, n2, D), mx.bfloat16)
    q = _rand(rng, (1, M, H, D), mx.bfloat16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin, break_invariant=True)
    _set(colsplit=True)
    got = _run(q, kv1, kv2, sink, idx, split, nwin)     # falls back to where-select
    # legacy semantics: gather concat(kv1, kv2) with no second source (tiled)
    _set(colsplit=False)
    kv_cat = mx.concatenate([kv1, kv2], axis=1)
    ref = sa.sparse_attn(q, kv_cat, sink, idx, D ** -0.5, chunk=64)
    mx.eval(got, ref)
    assert _bits(got) == _bits(ref)


# -- 5. C3 ON vs OFF ---------------------------------------------------------
@pytest.mark.parametrize("chunk,fence", [
    (64, "qtile"), (64, "ktile"), (256, "qtile"), (256, "ktile"), (64, "0"),
])
def test_c3_on_off_bit_exact(chunk, fence):
    rng = np.random.default_rng(zlib.crc32(f"c3{chunk}{fence}".encode()))
    split, n2, nwin = 60, 20, 16
    kv1, kv2 = _rand(rng, (1, split, D), mx.bfloat16), _rand(rng, (1, n2, D), mx.bfloat16)
    q = _rand(rng, (1, M, H, D), mx.bfloat16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin)
    _set(async_fence=False, fence=fence)
    off = _run(q, kv1, kv2, sink, idx, split, nwin, chunk=chunk)
    _set(async_fence=True, fence=fence)
    on = _run(q, kv1, kv2, sink, idx, split, nwin, chunk=chunk)
    mx.eval(off, on)
    assert _bits(off) == _bits(on), f"C3 not bit-exact (chunk={chunk}, fence={fence})"


# -- 6. all four C1xC3 combinations -----------------------------------------
@pytest.mark.parametrize("colsplit,async_fence", list(itertools.product((False, True), repeat=2)))
def test_all_gate_combinations_parity(colsplit, async_fence):
    rng = np.random.default_rng(zlib.crc32(f"combo{colsplit}{async_fence}".encode()))
    split, n2, nwin = 60, 20, 16
    kv1, kv2 = _rand(rng, (1, split, D), mx.bfloat16), _rand(rng, (1, n2, D), mx.bfloat16)
    q = _rand(rng, (1, M, H, D), mx.bfloat16)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    idx = _realistic_idx(rng, split, n2, nwin)
    _set(colsplit=colsplit, async_fence=async_fence, qtile=16)
    got = _run(q, kv1, kv2, sink, idx, split, nwin)
    mx.eval(got)
    assert got.shape == (1, M, H, D)
    # compare against the all-OFF arm
    _set(colsplit=False, async_fence=False, qtile=16)
    base = _run(q, kv1, kv2, sink, idx, split, nwin)
    mx.eval(base)
    assert _bits(got) == _bits(base), f"combo C1={colsplit} C3={async_fence} diverged"


# -- 7. real Attention.__call__ path ----------------------------------------
@pytest.mark.parametrize("colsplit,async_fence", list(itertools.product((False, True), repeat=2)))
def test_attention_path_bit_exact_with_gates(colsplit, async_fence):
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    from mlx_lm.models.deepseek_v41.cache import ModelCache
    from mlx_lm.models.deepseek_v41.attention import Attention
    from mlx_lm.models.deepseek_v41.model import SharedState
    from mlx_lm.models.deepseek_v41 import attention as attmod

    args = ModelArgs(
        dim=256, n_layers=3, n_heads=8, head_dim=512, rope_head_dim=64,
        q_lora_rank=64, o_lora_rank=64, o_groups=8, window_size=8,
        compress_ratios=(2, 2, 2), kv_source_layers=(0,),
        index_source_layers=(0,), index_n_heads=4, index_head_dim=128,
        index_topk=512, max_seq_len=64)
    att = Attention(0, args)
    att.attn_sink = att.attn_sink + 0.1
    mx.eval(att.parameters())

    def _run_att():
        n0 = 8
        mx.random.seed(1234)
        x = mx.random.normal((1, n0, 256)) * 0.5
        c = ModelCache(args, 1, 64)
        c.ensure_capacity(n0)
        acc = [att(x, 0, c, SharedState())]
        for step in range(4):
            mx.random.seed(100 + step)
            xx = mx.random.normal((1, 1, 256)) * 0.5
            acc.append(att(xx, n0 + step, c, SharedState()))
        out = mx.concatenate(acc, axis=1)
        mx.eval(out)
        return out

    saved_att = attmod._COLSPLIT
    try:
        sa._COLSPLIT, sa._FENCE_ASYNC, attmod._COLSPLIT = False, False, False
        base = _run_att()
        sa._COLSPLIT, sa._FENCE_ASYNC, attmod._COLSPLIT = colsplit, async_fence, colsplit
        got = _run_att()
    finally:
        attmod._COLSPLIT = saved_att
    assert mx.array_equal(base, got).item(), \
        f"Attention path diverged (C1={colsplit}, C3={async_fence})"
