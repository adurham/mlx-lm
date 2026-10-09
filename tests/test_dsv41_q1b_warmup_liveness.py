# Copyright © 2026 Adam Durham (hermes-gw)
"""ROUND-Q1B: the sharded-affine world=2 "first-forward stall".

The node logs showed this was not a collective pairing deadlock. Both ranks logged
``[Event::wait] slow wait`` at the same millisecond every 4-7 s, with zero
``Timed out`` under MLX_EVENT_WAIT_TIMEOUT_MS=20000, so the collectives kept
completing. The exo supervisor's silence watchdog (45 s + one 20 s growth
probe) killed a load warmup that was healthy but slow and emitted nothing:

* exl3: chunk512_a = 48 s, about 52.7 s of silence, survived with ~13 s to spare;
* affine6: more than 66 s of silence, killed 4/4 times.

What this file pins:

1. **collective parity**: a TP forward issues the SAME collective sequence
   (site, dtype, shape) whether the sharded dense projections are EXL3-style
   modules or ``AffineProj``. The projection type is never a collective
   input, and the fused-EXL3 helpers contain no collective call.
2. **liveness plumbing**: ``prefill.load_warmup`` forwards ``fence_hook`` to
   :func:`warmup`. With no hook (the default) it calls ``warmup`` exactly as
   before (``fence_hook=None``).
3. **build residue**: constructing an ``AffineProj`` leaves no fp16 build
   intermediates in MLX's buffer cache.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_q1b_warmup_liveness.py -q
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from mlx_lm.models.deepseek_v41 import collective as _coll
from mlx_lm.models.deepseek_v41 import exl3_build as eb
from mlx_lm.models.deepseek_v41 import prefill as PF
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import Model

IDS = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)


def toy_args() -> ModelArgs:
    return ModelArgs(
        vocab_size=256, dim=64, n_layers=4,
        n_heads=2, head_dim=128, rope_head_dim=16,
        q_lora_rank=64, o_lora_rank=32, o_groups=2, window_size=8,
        moe_inter_dim=64, n_routed_experts=8, n_shared_experts=1,
        n_activated_experts=2, compress_ratios=(2, 2, 2, 2),
        kv_source_layers=(0, 2), index_source_layers=(0,),
        index_n_heads=2, index_head_dim=32, index_topk=8,
        hc_mult=2, max_seq_len=64)


class _FakeGroup:
    """Stands in for the TP group: only identity matters to the model."""

    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0


def _tp_model(dense: str) -> Model:
    """Tiny real model wired like ``build_block`` with attn_tp + shared_tp on.

    ``dense="affine"`` replaces the projections that ``build_block`` slices
    (attn.wq_b, attn.wo_b, ffn.shared_experts.w1/w2/w3) with ``AffineProj``,
    which is what ``_dense_slice`` returns in affineN mode. ``dense="exl3"``
    keeps the stand-in linears (``Exl3Proj`` in production).
    """
    mx.random.seed(11)
    m = Model(toy_args())
    mx.eval(m.parameters())
    g = _FakeGroup()
    for blk in m.layers:
        blk.attn.group = g
        blk.ffn.group = g
        blk.ffn.shared_sharded = True
        if dense == "affine":
            for path in ("attn.wq_b", "attn.wo_b", "ffn.shared_experts.w1",
                         "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
                obj = blk
                parts = path.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                lin = getattr(obj, parts[-1])
                setattr(obj, parts[-1],
                        eb.AffineProj.from_weight(lin.weight, 8, 32,
                                                  expect=tuple(lin.weight.shape)))
    return m


def _collective_trace(model: Model, monkeypatch) -> list[tuple]:
    trace: list[tuple] = []

    def rec(x, group=None):
        site = inspect.stack()[1]
        trace.append((site.filename.rsplit("/", 1)[-1], site.function,
                      str(x.dtype), tuple(x.shape)))
        return x                                  # world-1 sum: identity

    monkeypatch.setattr(_coll, "all_sum", rec)
    cache = model.make_cache(1, max_seq_len=64)
    mx.eval(model(IDS, cache, last_logit_only=True, argmax=False))
    return trace


def test_tp_collective_sequence_is_independent_of_dense_format(monkeypatch):
    exl3 = _collective_trace(_tp_model("exl3"), monkeypatch)
    affine = _collective_trace(_tp_model("affine"), monkeypatch)
    # per layer: attention partial sum then the MoE tail (shared added BEFORE)
    assert len(exl3) == 2 * toy_args().n_layers
    sites = [(f, fn) for f, fn, _, _ in exl3]
    assert sites == [("attention.py", "__call__"), ("moe.py", "_all_sum_tail")] * 4
    assert exl3 == affine, "affine changed the collective sequence"


def test_fused_exl3_helpers_issue_no_collectives():
    """``_fuse_block`` is exl3-only (``DENSE_MODE == "exl3"``); its members must
    not contribute a collective, or exl3 and affine would diverge."""
    for obj in (eb._fuse_block, eb.Exl3FusedGroup, eb.Exl3Member,
                eb.Exl3GroupedStack, eb.AffineProj, eb._Grouped):
        src = inspect.getsource(obj)
        assert "all_sum" not in src and "distributed" not in src, obj


# --------------------------------------------------------------------------
# load_warmup liveness plumbing
# --------------------------------------------------------------------------

class _FakeCache:
    def __init__(self) -> None:
        self.offset = 0


class _DriverModel:
    """What ``warmup`` / ``load_warmup`` touch; the forward fires the hook like
    the real fenced forward does."""

    def __init__(self) -> None:
        self._fence_every = 99
        self.seen_hooks: list = []

    def make_cache(self, bsz=1, max_seq_len=None, dtype=None, **kw):
        return _FakeCache()

    def __call__(self, ids, cache, **kw):
        h = getattr(self, "_fence_hook", None)
        self.seen_hooks.append(h)
        if h is not None:
            h()
        cache.offset += int(ids.shape[1])
        return mx.array([[0.0]])


def test_load_warmup_forwards_fence_hook():
    m = _DriverModel()
    beats: list = []
    hook = lambda: beats.append(1)              # noqa: E731
    PF.load_warmup(m, None, chunk=4, long_chunk=2, decode=True,
                   fence_every=2, clear=False, fence_hook=hook)
    assert beats, "load_warmup did not deliver the fence hook to the forward"
    multi = [h for h in m.seen_hooks if h is not None]
    assert multi and all(h is hook for h in multi)
    assert m._fence_hook is None and m._fence_every == 99     # restored


def test_load_warmup_default_is_unchanged(monkeypatch):
    seen: dict = {}

    def fake_warmup(model, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(PF, "warmup", fake_warmup)
    PF.load_warmup(_DriverModel(), None, clear=False)
    assert seen.get("fence_hook", "MISSING") is None
    assert "fence_hook" in inspect.signature(PF.load_warmup).parameters


def test_load_warmup_hook_sees_host_synced_collectives():
    """The beat comes from inside ``sync_collectives``, i.e. after host-synced
    collectives: a beat implies the peer has paired every collective so far."""
    states: list = []
    m = _DriverModel()
    PF.load_warmup(m, None, chunk=4, long_chunk=2, decode=False, fence_every=2,
                   clear=False, fence_hook=lambda: states.append(_coll.active()))
    assert states and all(states)


# --------------------------------------------------------------------------
# AffineProj build residue
# --------------------------------------------------------------------------

@pytest.mark.skipif(not mx.metal.is_available(), reason="buffer cache is Metal")
def test_affine_proj_build_leaves_no_cache_residue():
    w = mx.random.normal((1024, 2048)).astype(mx.float16)
    mx.eval(w)
    mx.clear_cache()
    # bf16 source -> _set_weight builds an fp16 cast intermediate (the
    # reconstruct/slice intermediates in production); the caller keeps ``src``.
    src = w.astype(mx.bfloat16)
    mx.eval(src)
    p = eb.AffineProj.from_weight(src, 6, 64)
    assert mx.get_cache_memory() == 0, "AffineProj build left buffer-cache residue"
    assert p.out_features == 1024
    del src
