# Copyright © 2026 Adam Durham (hermes-gw)
"""Speculative rollback must leave nothing of the REJECTED drafts in the cache.

The window ring is position-addressed by ``p % window``, so a rejected verify row
at ``p`` overwrites the KV that a live read of the already-committed ``p -
window`` needs (``cache.py`` ``ring_snapshot`` documents the aliasing). A verify
forward writes ALL ``gamma + 1`` rows (``attention.py``: ``window_chrono`` reads,
then ``write_window`` writes), so a rollback that only rebuilds the compressor
carry leaks the rejected rows into the next forward's attention -- and, on a stack
with a layer above, into that layer's compressed / index caches, which are
committed and never recomputed (``docs/dsv41-state-machinery-audit-2026-10-14.md``
M1, exo repo).

These tests pin the FIXED invariant: for every acceptance count ``m`` in
``0..gamma`` (``m == 0`` is a FULL reject) two runs that differ ONLY in the content
of the rejected draft rows must end bit-identical in

* ``cache.offset``,
* every layer's window ring,
* every layer's ``comp_kv`` / ``index_k`` (the compressed and indexer caches),
* every layer's compressor carry,

and must produce bit-identical rows from the forward that follows the rollback.
The stack is 2 attention layers, so the upper layer *writes* compressed / index
caches from hidden states that came out of the lower layer's attention over the
ring -- the propagation channel a one-layer fixture cannot see at all.

``test_control_...`` disables only the ring restore and proves the assertions above
are load-bearing (the runs then differ, in the ring AND in the upper layer's
compressed cache).

Ports of the two audit repros in ``~/.hermes/cache/scratch/``
(``dsv41_spec_ring_alias_repro.py``, W=8; ``dsv41_spec_ring_alias_W128_repro.py``,
W=128), inverted to assert the fixed invariant, with the compressed-cache and
full-reject cases added.

Run (mlx-lm worktree root):

    PYTHONPATH=$PWD <venv>/bin/python -m pytest tests/test_dsv41_spec_rollback_ring.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mlx_lm.models.deepseek_v41 import spec
from mlx_lm.models.deepseek_v41.attention import Attention
from mlx_lm.models.deepseek_v41.cache import ModelCache
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.model import SharedState

WINDOWS = (8, 128)          # the audit repro's toy window, and production's
GAMMA = 3                   # the live deploy's dsv41 gamma


def _pos(window: int) -> int:
    """Absolute offset: the audit repros' values (20 / production-like 1000)."""
    return 20 if window == 8 else 1000


def _args(window: int, ratio: int = 2, n_layers: int = 2) -> ModelArgs:
    """A tiny two-layer subset: both layers are kv AND index sources, so each owns
    a compressor, a window ring, a compressed-KV cache and an index-key cache."""
    return ModelArgs(dim=64, n_heads=4, head_dim=32, rope_head_dim=8,
                     q_lora_rank=32, o_lora_rank=16, o_groups=2,
                     window_size=window, compress_ratios=(ratio,) * n_layers,
                     kv_source_layers=(0, 1), index_source_layers=(0, 1),
                     n_layers=n_layers)


def _filler(window: int, pos: int | None = None) -> mx.array:
    """Deterministic input rows, long enough for ``pos`` plus a window of reads."""
    pos = _pos(window) if pos is None else pos
    mx.random.seed(11)
    return mx.random.normal((1, pos + 4 * window + 32, 64))


class _Stack:
    """``n_layers`` attention layers stacked over one real ``ModelCache``.

    No MoE / norms: the point is the cache machinery. Layer ``i > 0`` takes layer
    ``i-1``'s attention output as its input, which is exactly how a stale window
    read reaches an upper layer's compressor / indexer. Advances the cache offset
    the way ``Model._forward`` does.
    """

    def __init__(self, args: ModelArgs, seed: int = 7):
        mx.random.seed(seed)
        self.layers = [Attention(i, args) for i in range(args.n_layers)]
        for layer in self.layers:
            mx.eval(layer.parameters())

    def __call__(self, x: mx.array, start_pos: int, cache) -> mx.array:
        shared = SharedState()
        h = x
        for layer in self.layers:
            h = layer(h, start_pos, cache, shared)
        mx.eval(h)
        cache.offset = start_pos + x.shape[1]
        return h


def _capture(cache) -> dict:
    """Every piece of state a rollback has to leave content-independent."""
    out: dict = {"offset": np.array(int(cache.offset))}
    for i, lc in enumerate(cache.layers):
        out[f"ring{i}"] = np.array(lc.win_kv.astype(mx.float32))
        out[f"comp_kv{i}"] = (None if lc.comp_kv is None
                              else np.array(lc.comp_kv.astype(mx.float32)))
        out[f"index_k{i}"] = (None if lc.index_k is None
                              else np.array(lc.index_k.astype(mx.float32)))
        cs = lc.comp_state
        out[f"carry_kv{i}"] = None if cs is None else np.array(cs.kv_state)
        out[f"carry_score{i}"] = None if cs is None else np.array(cs.score_state)
    return out


def _assert_identical(a: dict, b: dict, tag: str) -> None:
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None:
            assert b[key] is None, f"{tag}: {key} exists in one run only"
            continue
        assert np.array_equal(a[key], b[key]), (
            f"{tag}: {key} depends on the REJECTED drafts' content; "
            f"max|diff| = {np.abs(a[key].astype(np.float64) - b[key]).max()}")


def _spec_round(args: ModelArgs, x: mx.array, junk_seed: int, *, gamma: int, acc: int,
                next_rows: int, pos: int | None = None):
    """prefill to ``pos`` -> snap -> verify(anchor + gamma drafts) -> rollback to
    the first rejected position -> one more real forward, which reads the ring.

    The two runs of a test differ ONLY in ``junk_seed``, i.e. in the random content
    of the ``gamma - acc`` REJECTED draft rows.
    """
    window = args.window_size
    pos = _pos(window) if pos is None else pos
    cache = ModelCache(args, 1, pos + 256)
    stack = _Stack(args, seed=7)
    stack(x[:, :pos], 0, cache)
    assert int(cache.offset) == pos
    sn = spec.snap(cache, pos, gamma + 1)
    mx.random.seed(junk_seed)
    junk = mx.random.normal((1, gamma - acc, x.shape[-1]))
    verify_in = mx.concatenate([x[:, pos:pos + acc + 1], junk], axis=1)
    stack(verify_in, pos, cache)               # writes ALL gamma + 1 ring rows
    target = pos + acc + 1
    spec.rollback(cache, sn, target, spec.stashes(cache))
    assert int(cache.offset) == target
    out = stack(x[:, target:target + next_rows], target, cache)   # reads the ring
    return cache, np.array(out.astype(mx.float32))


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("acc", [0, 1, 2, 3])
def test_rejected_drafts_leave_no_trace(window, acc):
    """``acc == 0`` is a FULL reject: every verify row is discarded."""
    args, x = _args(window), _filler(window)
    cache_a, out_a = _spec_round(args, x, 100, gamma=GAMMA, acc=acc, next_rows=1)
    cache_b, out_b = _spec_round(args, x, 200, gamma=GAMMA, acc=acc, next_rows=1)
    _assert_identical(_capture(cache_a), _capture(cache_b), f"W={window} acc={acc}")
    assert np.array_equal(out_a, out_b), (
        f"W={window} acc={acc}: the forward AFTER the rollback returns different "
        "rows for different rejected-draft content")


@pytest.mark.parametrize("window", WINDOWS)
def test_two_layer_compressed_caches_are_content_independent(window):
    """The >= 2 layer assertion. The upper layer's ``comp_kv`` / ``index_k`` are
    written from hidden states that came out of the lower layer's attention over
    the ring, so with the leak they encode the rejected drafts permanently. Both
    the compressed caches and one extra forward are compared, not just the ring."""
    args, x = _args(window), _filler(window)
    cap_a, out_a = _spec_round(args, x, 100, gamma=GAMMA, acc=1, next_rows=4)
    cap_b, out_b = _spec_round(args, x, 200, gamma=GAMMA, acc=1, next_rows=4)
    _assert_identical(_capture(cap_a), _capture(cap_b), f"W={window} two-layer")
    assert np.array_equal(out_a, out_b)
    # Non-vacuous: the upper layer's compressed / index caches really are used.
    cap = _capture(cap_a)
    assert np.any(cap["comp_kv1"] != 0), "layer-1 comp_kv was never written"
    assert np.any(cap["index_k1"] != 0), "layer-1 index_k was never written"


@pytest.mark.parametrize("window", WINDOWS)
def test_accepted_rows_are_kept_and_rejected_slots_restored(window):
    """The fix restores ONLY the rejected slots: the accepted positions must keep
    the verify values, the rejected ones must get the pre-verify bytes back."""
    args, x = _args(window), _filler(window)
    pos, gamma, acc = _pos(window), GAMMA, 1
    target = pos + acc + 1
    cache = ModelCache(args, 1, pos + 256)
    stack = _Stack(args, seed=7)
    stack(x[:, :pos], 0, cache)
    before = [np.array(lc.win_kv.astype(mx.float32)) for lc in cache.layers]
    sn = spec.snap(cache, pos, gamma + 1)
    mx.random.seed(100)
    junk = mx.random.normal((1, gamma - acc, x.shape[-1]))
    verify_in = mx.concatenate([x[:, pos:target], junk], axis=1)
    stack(verify_in, pos, cache)
    mid = [np.array(lc.win_kv.astype(mx.float32)) for lc in cache.layers]
    spec.rollback(cache, sn, target, spec.stashes(cache))
    for lc, b, m in zip(cache.layers, before, mid):
        after = np.array(lc.win_kv.astype(mx.float32))
        w = lc.window
        accepted = [p % w for p in range(pos, target)]
        rejected = [p % w for p in range(target, pos + gamma + 1)]
        for slot in rejected:
            assert np.array_equal(after[:, slot], b[:, slot]), (
                f"W={window}: rejected position's slot {slot} was not restored")
        for slot in accepted:
            assert np.array_equal(after[:, slot], m[:, slot]), (
                f"W={window}: accepted position's slot {slot} was changed by the "
                "rollback (it must keep the verify value)")
        assert any(not np.array_equal(m[:, s], b[:, s]) for s in accepted), (
            f"W={window}: the verify forward changed no accepted slot - the check "
            "above is vacuous")


@pytest.mark.parametrize("ratio,pos,acc", [
    (3, 1003, 1),      # rollback target lands ON a compressor group boundary
    (4, 1003, 1),      # the verify leaves rejected rows in the carry's dead rows
])
def test_rejected_drafts_do_not_survive_in_the_compressor_carry(ratio, pos, acc):
    """The compressor carry is rebuilt from the carried + accepted rows, and the
    rows above the live count are canonicalized. Both parameters are chosen so the
    verify's rejected rows really do land in the carry buffer AND survive the
    following forward (which rewrites only the lowest rows): with ``ratio=3`` the
    boundary branch is live (``ratio=2`` with ``gamma+1=4`` never writes the carry
    at all), and ``ratio=4`` exercises the dead-upper-row branch. Without the
    canonicalization the two runs differ here (sabotage arm D)."""
    args, x = _args(8, ratio=ratio), _filler(8, pos)
    cap_a, out_a = _spec_round(args, x, 100, gamma=GAMMA, acc=acc, next_rows=1,
                               pos=pos)
    cap_b, out_b = _spec_round(args, x, 200, gamma=GAMMA, acc=acc, next_rows=1,
                               pos=pos)
    _assert_identical(_capture(cap_a), _capture(cap_b),
                      f"ratio={ratio} pos={pos} acc={acc} carry")
    assert np.array_equal(out_a, out_b)


def test_control_without_the_ring_restore_the_invariant_does_break(monkeypatch):
    """Negative control: with ONLY the ring restore disabled (the pre-fix
    behaviour) the two runs MUST differ. If this ever passes, the assertions above
    prove nothing. Two shapes are pinned:

    * ``next_rows=1``: the stale slot the following forward does not overwrite
      stays in the ring and changes that forward's own output;
    * ``next_rows=4``: the ring is overwritten by the forward itself, so the leak
      survives ONLY in the upper layer's compressed / index caches -- the exact
      channel a one-layer fixture cannot see.
    """
    monkeypatch.setattr(spec, "_restore_ring", lambda *a, **k: None)
    args, x = _args(8), _filler(8)

    c1, o1 = _spec_round(args, x, 100, gamma=GAMMA, acc=1, next_rows=1)
    c2, o2 = _spec_round(args, x, 200, gamma=GAMMA, acc=1, next_rows=1)
    assert not np.array_equal(_capture(c1)["ring0"], _capture(c2)["ring0"]), (
        "control: the un-restored run did not leak into the lowest layer's ring; "
        "the scenario does not exercise the aliasing")
    assert not np.array_equal(o1, o2), (
        "control: the leak did not change the following forward's output")
    with pytest.raises(AssertionError):
        _assert_identical(_capture(c1), _capture(c2), "control")

    c3 = _capture(_spec_round(args, x, 100, gamma=GAMMA, acc=1, next_rows=4)[0])
    c4 = _capture(_spec_round(args, x, 200, gamma=GAMMA, acc=1, next_rows=4)[0])
    assert not np.array_equal(c3["comp_kv1"], c4["comp_kv1"]), (
        "control: the leak did not reach the upper layer's compressed cache; the "
        "two-layer assertion above is not load-bearing")
    assert not np.array_equal(c3["index_k1"], c4["index_k1"]), (
        "control: the leak did not reach the upper layer's index-key cache")


def test_rollback_refuses_a_snapshot_that_does_not_cover_the_write():
    """``snap(verify_rows)`` must cover the forward's row count. An under-sized
    snapshot cannot restore the rejected rows, so rollback has to fail loudly
    instead of leaving them in the ring."""
    args, x = _args(8), _filler(8)
    pos = _pos(8)
    cache = ModelCache(args, 1, pos + 256)
    stack = _Stack(args, seed=7)
    stack(x[:, :pos], 0, cache)
    sn = spec.snap(cache, pos, 2)              # too small: the forward writes 4
    stack(x[:, pos:pos + GAMMA + 1], pos, cache)
    with pytest.raises(RuntimeError, match="cannot be restored"):
        spec.rollback(cache, sn, pos + 2, spec.stashes(cache))


def test_window_write_slots_matches_the_write_it_describes():
    """``write_window`` and ``window_write_slots`` must agree exactly: the slot
    set the rollback snapshots is the slot set the write touches, and position
    ``p`` lands in slot ``p % window``."""
    cache = ModelCache(_args(8), 1, 64)
    lc = cache.layers[0]
    w = lc.window
    for pos, n in ((0, 1), (6, 4), (1000, 5), (1000, 2 * w + 3)):
        before = np.array(lc.win_kv.astype(mx.float32))
        lc.write_window(pos, mx.full((1, n, lc.win_kv.shape[-1]), 5.0))
        after = np.array(lc.win_kv.astype(mx.float32))
        slots = sorted(set(np.array(lc.window_write_slots(pos, n)).tolist()))
        changed = np.flatnonzero((after != before).any(axis=(0, 2))).tolist()
        assert changed == slots, (
            f"pos={pos} n={n}: write touched {changed}, "
            f"window_write_slots says {slots}")
        for k in range(max(0, n - w), n):
            assert after[0, (pos + k) % w, 0] == 5.0
        lc.win_kv = mx.zeros_like(lc.win_kv)
