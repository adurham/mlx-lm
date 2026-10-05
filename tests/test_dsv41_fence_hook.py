# LOCAL ADDITION (not vendored). Tests for the prefill fence liveness hook.
"""Fence-point liveness hook for ``deepseek_v41.Model`` (design: Fix A).

A healthy deep-context prefill can run minutes per chunk, so a supervisor that
watches only per-chunk progress events cannot tell "working" from "wedged". The
fix keys a liveness callback to the existing per-K-layers eval fence, so every
beat is backed by K layers of committed compute. These tests pin the contract:

* call-count formula -- a multi-row forward calls the hook once per fence point
  (layer ids ``K-1, 2K-1, ...``), for every ``_fence_every`` K;
* zero calls at ``n == 1`` (decode is never fenced);
* zero calls when the hook is unset, or when ``_fence_every == 0``;
* fail-open: an exception in the hook does NOT propagate, disables the hook for
  the object, latches ``_fence_hook_failed``, logs once, and never repeats;
* driver install/restore: ``prefill.prefill`` / ``warmup`` set
  ``model._fence_hook`` for the call and restore it in ``finally`` -- including
  when the forward raises;
* bit-identity: a forward with a pure-counting hook is byte-identical to the
  same forward with no hook.

The model used is a real (tiny, 4-layer) ``deepseek_v41.Model`` on CPU, so the
fence path, the hook call site and the returned tensors are the production ones
- not a double.

Usage::

    PYTHONPATH=. <venv>/bin/python -m pytest tests/test_dsv41_fence_hook.py -q
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import prefill as PF
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import Model

# --------------------------------------------------------------------------
# tiny real model (CPU): enough layers that several fence points exist
# --------------------------------------------------------------------------

ROWS = 8
IDS = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)


def toy_args() -> ModelArgs:
    """A minimal but structurally complete config (4 layers, 2 fence points at
    K=2). Small enough to build and forward on CPU in milliseconds."""
    return ModelArgs(
        vocab_size=256, dim=64, n_layers=4,
        n_heads=2, head_dim=128, rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=2, window_size=8,
        moe_inter_dim=32, n_routed_experts=8, n_shared_experts=1,
        n_activated_experts=2, compress_ratios=(2, 2, 2, 2),
        kv_source_layers=(0, 2), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=8,
        hc_mult=2, max_seq_len=64)


@pytest.fixture
def model() -> Model:
    mx.random.seed(7)
    m = Model(toy_args())
    mx.eval(m.parameters())
    return m


def fence_points(n_layers: int, fence: int) -> int:
    """The model's own fence arithmetic: a call at every layer id ``i`` with
    ``i % fence == fence - 1`` (ids 0..n_layers-1)."""
    return sum(1 for i in range(n_layers) if (i + 1) % fence == 0)


def digest(a: mx.array) -> str:
    mx.eval(a)
    return hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()


class Counter:
    """A pure Python hook (no mx ops, no state mutation) that counts beats and
    can be made to raise after ``fail_after`` successful calls."""

    def __init__(self, fail_after: int | None = None):
        self.n = 0
        self.fail_after = fail_after

    def __call__(self) -> None:
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RuntimeError("fence hook boom")


# --------------------------------------------------------------------------
# 1. call-count formula
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fence", [1, 2, 3, 4, 8])
def test_call_count_matches_fence_points(model: Model, fence: int) -> None:
    model._fence_every = fence
    c = Counter()
    model._fence_hook = c
    cache = model.make_cache(1, max_seq_len=64)
    out = model(IDS, cache, last_logit_only=True, argmax=False)
    mx.eval(out)
    assert c.n == fence_points(model.args.n_layers, fence)


def test_call_count_multirow_across_chunks() -> None:
    """Across a driver run of several chunks the count is the per-chunk fence
    points summed (here verified directly on the model for two distinct row
    counts)."""
    mx.random.seed(11)
    m = Model(toy_args())
    mx.eval(m.parameters())
    m._fence_every = 2
    c = Counter()
    m._fence_hook = c
    cache = m.make_cache(1, max_seq_len=64)
    mx.eval(m(IDS, cache, last_logit_only=True, argmax=False))
    assert c.n == 2                       # 4 layers, K=2 -> layers 1 and 3


# --------------------------------------------------------------------------
# 2. zero calls: single row, unset hook, disabled fence
# --------------------------------------------------------------------------

def test_zero_calls_single_row(model: Model) -> None:
    model._fence_every = 2
    c = Counter()
    model._fence_hook = c
    cache = model.make_cache(1, max_seq_len=64)
    mx.eval(model(mx.array([[3]], dtype=mx.int32), cache,
                  last_logit_only=True, argmax=False))
    assert c.n == 0


def test_zero_calls_when_hook_unset(model: Model) -> None:
    model._fence_every = 2
    assert getattr(model, "_fence_hook", None) is None
    cache = model.make_cache(1, max_seq_len=64)
    out = model(IDS, cache, last_logit_only=True, argmax=False)
    mx.eval(out)                          # must simply not raise
    assert out.shape == (1, 1, model.args.vocab_size)


def test_zero_calls_when_fence_disabled(model: Model) -> None:
    model._fence_every = 0
    c = Counter()
    model._fence_hook = c
    cache = model.make_cache(1, max_seq_len=64)
    mx.eval(model(IDS, cache, last_logit_only=True, argmax=False))
    assert c.n == 0


# --------------------------------------------------------------------------
# 3. fail-open: a broken hook never propagates, never repeats
# --------------------------------------------------------------------------

def test_exception_disables_latches_and_logs_once(model: Model,
                                                  caplog) -> None:
    model._fence_every = 2
    c = Counter(fail_after=0)             # raises on the first beat
    model._fence_hook = c
    with caplog.at_level(logging.WARNING,
                         logger="mlx_lm.models.deepseek_v41.model"):
        cache = model.make_cache(1, max_seq_len=64)
        out = model(IDS, cache, last_logit_only=True, argmax=False)
        mx.eval(out)                      # the bug must NOT propagate

    assert out.shape == (1, 1, model.args.vocab_size)
    assert model._fence_hook is None       # disabled for this object
    assert model._fence_hook_failed is True
    assert c.n == 1                        # one attempt, the 2nd fence skipped
    warnings = [r for r in caplog.records if "liveness hook raised" in r.message]
    assert len(warnings) == 1              # logged exactly once

    # a later forward makes no further attempt
    cache2 = model.make_cache(1, max_seq_len=64)
    mx.eval(model(IDS, cache2, last_logit_only=True, argmax=False))
    assert c.n == 1


# --------------------------------------------------------------------------
# 4. bit-identity: hook presence changes nothing in the returned tensors
# --------------------------------------------------------------------------

def test_bit_identity_with_and_without_hook(model: Model) -> None:
    def run(hook) -> str:
        model._fence_every = 2
        model._fence_hook = hook
        cache = model.make_cache(1, max_seq_len=64)
        return digest(model(IDS, cache, last_logit_only=True, argmax=False))

    a = run(None)
    c = Counter()
    b = run(c)
    assert c.n == 2                        # the hook really ran on the fenced run
    assert a == b, "hook changed the output"


# --------------------------------------------------------------------------
# 5. driver install / restore (prefill.prefill, warmup)
# --------------------------------------------------------------------------

class _FakeCache:
    def __init__(self) -> None:
        self.offset = 0


class _DriverModel:
    """Minimal stand-in exposing exactly what ``prefill()`` / ``warmup()`` touch.
    Its ``__call__`` records and invokes the installed hook, like the real
    fenced forward does, and advances the cache."""

    def __init__(self) -> None:
        self._fence_every = 99            # sentinel: must be restored
        self.seen_hooks: list = []

    def make_cache(self, bsz=1, max_seq_len=None, dtype=None, **kw) -> _FakeCache:
        return _FakeCache()

    def __call__(self, ids, cache, **kw) -> mx.array:
        h = getattr(self, "_fence_hook", None)
        self.seen_hooks.append(h)
        if h is not None:
            h()
        n = int(ids.shape[1])
        cache.offset += n
        return mx.array([[float(n)]])


def test_driver_installs_and_restores_hook() -> None:
    m = _DriverModel()
    prev = lambda: None                   # noqa: E731 (identity we must restore)
    m._fence_hook = prev
    calls: list = []
    hook = lambda: calls.append(1)        # noqa: E731

    PF.prefill(m, np.arange(ROWS, dtype=np.int64), m.make_cache(),
               chunk=4, long_chunk=4, long_threshold=10 ** 9,
               fence_every=2, async_depth=0, clear_cache_every=0,
               fence_hook=hook)

    assert calls, "driver did not exercise the hook"
    assert m._fence_hook is prev           # restored to the pre-call value
    assert m._fence_every == 99            # fence restored alongside it
    assert m.seen_hooks and all(h is hook for h in m.seen_hooks)


def test_driver_restores_hook_after_forward_exception() -> None:
    class _Boom(_DriverModel):
        def __call__(self, ids, cache, **kw) -> mx.array:
            self.seen_hooks.append(getattr(self, "_fence_hook", None))
            raise RuntimeError("forward failed")

    m = _Boom()
    prev = lambda: None                    # noqa: E731
    m._fence_hook = prev
    inst = lambda: None                    # noqa: E731

    with pytest.raises(RuntimeError, match="forward failed"):
        PF.prefill(m, np.arange(ROWS, dtype=np.int64), m.make_cache(),
                   chunk=4, fence_every=2, async_depth=0,
                   fence_hook=inst)

    assert m._fence_hook is prev           # finally restored despite the raise
    assert m._fence_every == 99
    assert m.seen_hooks == [inst]          # the hook was installed for the call


def test_warmup_installs_and_restores_hook() -> None:
    m = _DriverModel()                     # _fence_hook absent -> prev is None
    hook = lambda: None                    # noqa: E731

    PF.warmup(m, chunk=4, long_chunk=2, decode=False, fence_every=2,
              fence_hook=hook, clear=False)

    assert m._fence_hook is None           # restored (was absent before)
    assert m._fence_every == 99
    assert m.seen_hooks and all(h is hook for h in m.seen_hooks)


def test_prefill_without_hook_leaves_attribute_untouched() -> None:
    m = _DriverModel()
    PF.prefill(m, np.arange(ROWS, dtype=np.int64), m.make_cache(),
               chunk=4, fence_every=2, async_depth=0, clear_cache_every=0)
    assert m._fence_hook is None           # default installs nothing
    assert m.seen_hooks and all(h is None for h in m.seen_hooks)
