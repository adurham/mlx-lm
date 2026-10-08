# Copyright © 2026 Adam Durham (hermes-gw)
"""Small-m guard for the C1 column-split gather (``DSV41_SPARSE_COLSPLIT``).

BACKGROUND (source-verified). The two twin C1 gates disagreed:

* ``sparse_attention.py`` sets ``_COLSPLIT = env.get("DSV41_SPARSE_COLSPLIT",
  "1") == "1"``  -- default **ON**;
* ``attention.py`` sets ``_COLSPLIT = env.get("DSV41_SPARSE_COLSPLIT", "0") ==
  "1"``  -- default **OFF**.

``attention.py`` passes ``colsplit`` to ``sparse_attn`` ONLY when ITS gate is on
(``extra = {"colsplit": n_window} if _COLSPLIT else {}``), so at the production
env (unset) the call arrives with ``colsplit=None``. ``sparse_attn`` then
DERIVED the boundary via ``_column_boundary(..., -1)`` ->
``_leading_window_columns`` (``int(mx.min(...))``, one host round-trip) plus the
provenance check (``bool(mx.all(checks).item())``, a second). Those two host
syncs ran on EVERY compressing layer of EVERY decode (m=1) and 4-row verify
forward, for nothing -- the C1 gather only pays off where a tile would
otherwise gather BOTH sources, i.e. large-m prefill.

THE FIX (this change). Gate the whole boundary derivation+check on the call's
row count, reusing the existing ``_FENCE_MIN_ROWS`` (default 16, sparse_attention
line ~117, already used to pick prefill-vs-small): ``kv2 is not None and
_COLSPLIT and m > _FENCE_MIN_ROWS``. Large m keeps C1 exactly as before; small m
falls back to ``_gather_split``, which is VALUE-IDENTICAL -- for a real index
(``>= 0``) the selected gather is the same source buffer row, and a ``-1`` mask
pad resolves to row 0 of ``kv`` whose logit is forced to ``-inf`` downstream
(``_tile_init``/``_tile_step``) and cannot contribute.

What this file pins:
1. the shipped gate defaults (sparse_attention ON, attention OFF) are
   unchanged and env-following (subprocess, clean interpreter);
2. the DEFAULT (env-unset equivalence) small-m call is bit-identical to the
   same call forced through the C1 column-partitioned gather, for m=1 (decode)
   and m=4 (verify), with and without ``-1`` pads, and for both the declared
   (``colsplit=n_window``) and derived (``colsplit=None``) boundary arms;
3. ``_column_boundary`` is NOT called for m <= _FENCE_MIN_ROWS and IS called
   for m > _FENCE_MIN_ROWS (the gate is observable);
4. large-m behaviour is unchanged: C1 on vs off is bit-exact above the
   threshold (regression guard);
5. the real ``Attention.__call__`` decode path issues no ``_column_boundary``
   call in its small forwards and is bit-exact with the C1 gate either way.

Run (mlx-lm worktree root; src/ first so the exo conftest landmine guard passes):

    PYTHONPATH=<repo>/src:<repo>/mlx-lm <venv>/bin/python -m pytest \\
        tests/test_dsv41_sparse_smallm_colsplit.py -q
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import sparse_attention as sa

B = 1
H = 2            # heads for the sink
D = 8            # head_dim
K = 16           # topk width
SPLIT = 6        # window block width (kv rows); compressed source is kv2
N2 = 9           # kv2 rows
NWIN = 4         # window columns in the index matrix (the declared boundary)
THRESH = sa._FENCE_MIN_ROWS

_GATES = ("_COLSPLIT",)
_KNOBS = ("_FENCE_MIN_ROWS", "_QTILE", "_KTILE", "_WDTYPE", "_IMPL",
          "_COMPILE", "_BUDGET_BYTES", "_FENCE", "_FENCE_ASYNC")


@pytest.fixture(autouse=True)
def _restore_gates():
    saved = {n: getattr(sa, n) for n in _GATES + _KNOBS}
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(sa, n, v)


def _idx(rng, m, *, pads=True, break_invariant=False) -> mx.array:
    """Index matrix laid out like attention.py: [window (<split) | topk (>=split)].

    Window columns ``[0, NWIN)`` carry values in ``[0, SPLIT)``, top-k columns
    ``[NWIN, K)`` carry values in ``[SPLIT, SPLIT+N2)``; ``-1`` masks are sprinkled
    in both regions. ``pads=False`` removes every mask so the ``-1`` handling is
    also covered by a mask-free arm.
    """
    idx = np.zeros((B, m, K), np.int32)
    idx[:, :, :NWIN] = rng.integers(0, SPLIT, size=(B, m, NWIN))
    idx[:, :, NWIN:] = rng.integers(SPLIT, SPLIT + N2, size=(B, m, K - NWIN))
    if pads:
        idx[rng.random((B, m, K)) < 0.25] = -1
    if break_invariant:
        idx[:, 0, NWIN + 1] = min(2, SPLIT - 1)      # real window value, wrong side
    return mx.array(idx)


def _qkv(rng, m, *, pads=True, break_invariant=False):
    kv1 = mx.array(rng.standard_normal((B, SPLIT, D)).astype(np.float32))
    kv2 = mx.array(rng.standard_normal((B, N2, D)).astype(np.float32))
    q = mx.array(rng.standard_normal((B, m, H, D)).astype(np.float32))
    sink = mx.array(rng.standard_normal((H,)).astype(np.float32))
    return q, kv1, kv2, sink, _idx(rng, m, pads=pads, break_invariant=break_invariant)


# -- 1. shipped gate defaults -------------------------------------------------
def test_twin_gates_defaults_unchanged():
    """sparse_attention C1 stays ON, attention C1 stays OFF (env unset)."""
    from mlx_lm.models.deepseek_v41 import attention as attmod
    if os.environ.get("DSV41_SPARSE_COLSPLIT") is None:
        assert sa._COLSPLIT is True, "sparse_attention C1 must default ON (ships C1)"
        assert attmod._COLSPLIT is False, "attention C1 must default OFF (byte-identical call site)"


def test_twin_gates_follow_env_at_import():
    """Each gate is read at import; both follow the same env var (clean interp)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import mlx_lm.models.deepseek_v41.sparse_attention as S;"
            "import mlx_lm.models.deepseek_v41.attention as A;"
            "print(int(S._COLSPLIT), int(A._COLSPLIT))")
    for val, want in (("1", "1 1"), ("0", "0 0")):
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={**os.environ, "DSV41_SPARSE_COLSPLIT": val}, cwd=root)
        assert r.stdout.strip() == want, f"env={val}: {r.stdout!r} {r.stderr!r}"


# -- 2. small-m default == forced C1, bit-exact -------------------------------
@pytest.mark.parametrize("m,pads,declared", list(itertools.product(
    (1, 4), (True, False), (True, False))))
def test_smallm_default_bit_identical_to_forced_c1(m, pads, declared):
    """The default small-m fallback equals the forced column-partitioned gather.

    Arm A (default): threshold left at its shipped 16 -> the boundary block is
    skipped, ``colsplit`` is never used -> ``_gather_split``. Arm B (forced C1):
    threshold lowered to 0 and a valid ``colsplit`` supplied (declared NWIN, or
    derived from ``None``) -> ``_gather_cols``. The layout satisfies the C1
    invariant, so arm B really enters the column-partitioned gather.
    """
    rng = np.random.default_rng(zlib.crc32(f"smallm{m}{pads}{declared}".encode()))
    q, kv1, kv2, sink, idx = _qkv(rng, m, pads=pads)
    sa._COLSPLIT = True

    # the crafted layout is a clean C1 layout: boundary is NWIN either way
    assert sa._column_boundary(idx, SPLIT, NWIN) == NWIN
    assert sa._column_boundary(idx, SPLIT, -1) == NWIN

    sa._FENCE_MIN_ROWS = THRESH                 # shipped default -> skip
    default = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                             split=SPLIT)
    sa._FENCE_MIN_ROWS = 0                      # force the boundary block
    forced = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                            split=SPLIT, colsplit=(NWIN if declared else None))
    mx.eval(default, forced)

    assert default.shape == forced.shape and default.dtype == forced.dtype
    assert bool(mx.array_equal(default, forced).item()), (
        f"small-m default diverged from forced C1 (m={m}, pads={pads}, "
        f"declared={declared})")


def test_smallm_declared_out_of_range_boundary_is_inert():
    """A stale/bogus declared boundary can never mis-gather: it is not consulted
    at all for small m, and the output still equals the plain where-select."""
    rng = np.random.default_rng(4242)
    q, kv1, kv2, sink, idx = _qkv(rng, 4, pads=True)
    sa._COLSPLIT, sa._FENCE_MIN_ROWS = True, THRESH
    default = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                             split=SPLIT, colsplit=K + 5)   # nonsense declared
    sa._COLSPLIT = False
    ref = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                         split=SPLIT)
    mx.eval(default, ref)
    assert bool(mx.array_equal(default, ref).item())


# -- 3. the gate is observable ------------------------------------------------
def test_boundary_not_called_for_small_m():
    """m <= _FENCE_MIN_ROWS -> _column_boundary is never reached."""
    calls = []
    orig = sa._column_boundary

    def spy(icb, split, colsplit):
        calls.append((tuple(icb.shape), split, colsplit))
        return orig(icb, split, colsplit)

    sa._column_boundary = spy
    sa._COLSPLIT, sa._FENCE_MIN_ROWS = True, THRESH
    try:
        for m in (1, 4, THRESH):
            calls.clear()
            rng = np.random.default_rng(100 + m)
            q, kv1, kv2, sink, idx = _qkv(rng, m)
            out = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, kv2=kv2,
                                  split=SPLIT, colsplit=NWIN)
            mx.eval(out)
            assert calls == [], f"_column_boundary ran at m={m}: {calls}"
    finally:
        sa._column_boundary = orig


@pytest.mark.parametrize("declared", [True, False])
def test_boundary_consulted_for_large_m(declared):
    """m > _FENCE_MIN_ROWS -> the C1 boundary block runs (prefill unchanged)."""
    calls = []
    orig = sa._column_boundary

    def spy(icb, split, colsplit):
        calls.append((tuple(icb.shape), split, colsplit))
        return orig(icb, split, colsplit)

    sa._column_boundary = spy
    sa._COLSPLIT, sa._FENCE_MIN_ROWS = True, THRESH
    try:
        rng = np.random.default_rng(7)
        q, kv1, kv2, sink, idx = _qkv(rng, THRESH + 32)
        out = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, kv2=kv2,
                              split=SPLIT, colsplit=(NWIN if declared else None))
        mx.eval(out)
        assert calls, "C1 boundary block did not run for m > threshold"
        assert calls[0][2] == (NWIN if declared else -1)
    finally:
        sa._column_boundary = orig


# -- 4. large-m behaviour unchanged -------------------------------------------
@pytest.mark.parametrize("m", [THRESH + 1, 64])
def test_large_m_c1_on_off_bit_exact(m):
    """Above the threshold the guard is a no-op: C1 on == off, bit-exact."""
    rng = np.random.default_rng(zlib.crc32(f"largem{m}".encode()))
    q, kv1, kv2, sink, idx = _qkv(rng, m)
    sa._FENCE_MIN_ROWS = THRESH
    sa._COLSPLIT = False
    off = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                         split=SPLIT)
    sa._COLSPLIT = True
    on = sa.sparse_attn(q, kv1, sink, idx, D ** -0.5, chunk=64, kv2=kv2,
                        split=SPLIT, colsplit=NWIN)
    mx.eval(off, on)
    assert bool(mx.array_equal(off, on).item()), f"large-m C1 not bit-exact (m={m})"


# -- 5. real Attention.__call__ path -----------------------------------------
def test_attention_smallm_no_boundary_call_and_bit_exact():
    """End-to-end: the decode path issues no boundary call and is gate-agnostic.

    Builds the real ``Attention`` (compressor + indexer + ring), runs one 8-row
    forward then four 1-row decodes (all <= threshold), and asserts (a) the
    default gates -- attention OFF, sparse_attention ON -- produce NO
    ``_column_boundary`` call, and (b) the output is bit-identical with the C1
    gate forced on at the call site.
    """
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

    calls = []
    orig_boundary = sa._column_boundary

    def spy_boundary(icb, split, colsplit):
        calls.append(tuple(icb.shape))
        return orig_boundary(icb, split, colsplit)

    saved_att = attmod._COLSPLIT
    sa._column_boundary = spy_boundary
    sa._COLSPLIT, sa._FENCE_MIN_ROWS = True, THRESH   # shipped defaults

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

    try:
        attmod._COLSPLIT = False            # shipped attention default
        calls.clear()
        base = _run_att()
        assert calls == [], f"boundary ran on the small-m decode path: {calls}"
        attmod._COLSPLIT = True             # gate forced on (declares colsplit)
        got = _run_att()
    finally:
        attmod._COLSPLIT = saved_att
        sa._column_boundary = orig_boundary

    assert bool(mx.array_equal(base, got).item()), \
        "Attention small-m output diverged across the C1 gate"
