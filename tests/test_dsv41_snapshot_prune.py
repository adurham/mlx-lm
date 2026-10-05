# LOCAL ADDITION (not vendored). Snapshot retention: never prune offset 0.
"""Retention-policy tests for ``SessionCache.snapshot``'s prune loop.

``snapshot()`` keeps at most ``max_snapshots`` checkpoints. This file pins the
amended policy (see docs/dsv41-reuse-ladder-design-2026-10-05.md, Fable review
item 4): the checkpoint at offset 0 is the permanent fallback boundary and is
NEVER evicted; when over budget the oldest *non-zero* checkpoint goes first.
Without that rule a periodic checkpoint ladder turns a legitimate near-origin
rewind into a ``RollbackError`` once 32+ higher offsets have swept through the
cap.

Reuses the production ``ModelCache`` over the tiny layer subset and the
position-derived stub prefill from ``test_dsv41_session_cache`` (same package),
so the prune assertions exercise the real class, not a double.

    PYTHONPATH=. python -m pytest tests/test_dsv41_snapshot_prune.py -q
"""

from __future__ import annotations

import numpy as np

from mlx_lm.models.deepseek_v41 import session_cache as SC
from tests.test_dsv41_session_cache import (
    StubModel,
    feed,
    state_diff,
    stub_prefill,
)

MAX_SEQ = 512


def _session(max_snapshots: int) -> SC.SessionCache:
    return SC.SessionCache(
        StubModel(), max_seq_len=MAX_SEQ, prefill_fn=stub_prefill,
        max_snapshots=max_snapshots,
    )


# --------------------------------------------------------------------------
# 1) over-budget prune keeps offset 0 + the N newest non-zero checkpoints
# --------------------------------------------------------------------------

def test_prune_keeps_zero_and_newest_nonzero() -> None:
    """max_snapshots=3, snapshots at 0,10,20,30,40 -> survivors {0,30,40}."""
    s = _session(max_snapshots=3)
    ids = np.arange(64, dtype=np.int64)
    for n in (10, 20, 30, 40):
        s.append_turn(ids[:n])
    # the constructor's pos-0 boundary must have survived every eviction
    assert s.boundaries == [0, 30, 40], s.boundaries
    assert len(s.boundaries) <= s.max_snapshots
    # and every surviving boundary is a restorable checkpoint
    assert set(s.boundaries) <= set(s._snaps)


def test_prune_keeps_zero_at_larger_cap() -> None:
    """max_snapshots=4, offsets 0..50 -> survivors {0,30,40,50}."""
    s = _session(max_snapshots=4)
    ids = np.arange(80, dtype=np.int64)
    for n in (10, 20, 30, 40, 50):
        s.append_turn(ids[:n])
    assert s.boundaries == [0, 30, 40, 50], s.boundaries


def test_zero_survives_many_evictions() -> None:
    """A long ladder (40 rungs) never drops the origin."""
    s = SC.SessionCache(StubModel(), max_seq_len=4096, prefill_fn=stub_prefill,
                        max_snapshots=8)
    ids = np.arange(3000, dtype=np.int64)
    offs = list(range(64, 64 * 41, 64))          # 40 snapshots, 64..2560
    for n in offs:
        s.append_turn(ids[:n])
    assert 0 in s.boundaries
    assert len(s.boundaries) == 8                      # cap still honoured
    assert s.boundaries[0] == 0
    assert s.boundaries[-1] == offs[-1]                # newest stays
    assert s.boundaries[1:] == offs[-(8 - 1):]         # newest 7 non-zero


# --------------------------------------------------------------------------
# 2) regression: rewind to offset 0 still works after the cap has overflowed
# --------------------------------------------------------------------------

def test_rewind_to_zero_after_overflow() -> None:
    """The motivating scenario: an lcp near 0 after many ladder rungs.

    Pure-newest eviction would have dropped the offset-0 checkpoint, so this
    ``append_turn`` (prefix mismatch at row 1 -> lcp=1 -> boundary 0) would
    raise ``RollbackError`` and force a full cold prefill. With keep-0 it is a
    correct rewind-to-0 + refill, and the state matches a fresh feed bitwise.
    """
    ids = np.arange(300, dtype=np.int64)
    s = _session(max_snapshots=3)
    for n in (64, 128, 192, 256):                # overflow 3-checkpoint budget
        s.append_turn(ids[:n])
    assert 0 in s.boundaries, s.boundaries

    mut = ids.copy()
    mut[1] = 999                                 # diverge one row past the origin
    r = s.append_turn(mut[:256])                 # must NOT raise
    assert r.turn_start == 0 and r.rolled_back_from == 256, f"{r}"
    assert r.tokens_prefilled == 256             # full refill from the origin
    assert not state_diff(s.cache, feed(StubModel(), [mut[:256]])), \
        "rewind-to-0 state must equal a fresh feed"


def test_rewind_to_zero_without_mutation_is_plain_truncation() -> None:
    """Rewinding to a surviving 0 also covers the no-mutation truncation path."""
    ids = np.arange(300, dtype=np.int64)
    s = _session(max_snapshots=3)
    for n in (64, 128, 192, 256):
        s.append_turn(ids[:n])
    r = s.append_turn(ids[:1])                   # shares row 0 -> rewinds to 0
    assert r.turn_start == 0, f"{r}"
    assert r.tokens_prefilled == 1


# --------------------------------------------------------------------------
# 3) cap semantics unchanged for the non-origin part
# --------------------------------------------------------------------------

def test_no_overflow_leaves_all_boundaries() -> None:
    """Under budget, nothing is evicted -- identical to the old behaviour."""
    s = _session(max_snapshots=5)
    ids = np.arange(80, dtype=np.int64)
    for n in (10, 20, 30):
        s.append_turn(ids[:n])
    assert s.boundaries == [0, 10, 20, 30], s.boundaries


def test_exactly_at_cap_no_eviction() -> None:
    s = _session(max_snapshots=4)
    ids = np.arange(80, dtype=np.int64)
    for n in (10, 20, 30):
        s.append_turn(ids[:n])
    assert s.boundaries == [0, 10, 20, 30], s.boundaries
    assert len(s.boundaries) == s.max_snapshots


def test_min_cap_two_keeps_zero_and_newest() -> None:
    """The floor (max_snapshots>=2) leaves {0, newest}: never empty."""
    s = _session(max_snapshots=2)
    ids = np.arange(120, dtype=np.int64)
    for n in (10, 20, 30, 40):
        s.append_turn(ids[:n])
    assert s.boundaries == [0, 40], s.boundaries
