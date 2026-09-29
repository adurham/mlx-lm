"""Cross-rank collectives for the DSv4.1 TP model, with a GPU-watchdog guard.

Why this exists (measured 2026-09-29, 2x M4 Max, JACCL over RDMA): JACCL runs
``all_sum`` on a CPU stream, and any GPU work that consumes its result is
encoded behind a wait on that event *inside a Metal command buffer*. If the
peer rank reaches the collective more than ~5 s later, the waiting rank's
command buffer trips the Metal watchdog ("Caused GPU Timeout Error"), and
``MTL_DISABLE_TIMEOUT=1`` does not prevent it (skew probe: 2 s skew fine, 8 s and
20 s skew -> timeouts on both ranks). Rank skew of that size happens on the
first forward after a build, when each rank JIT-compiles its custom Metal
kernels at a different speed; it hung ~half of the full-model two-node runs,
including the pre-workstream tree.

``sync_collectives()`` (or ``DSV41_SYNC_COLLECTIVES=1``) evaluates every
collective's result on the host before any GPU work is encoded behind it, so a
late peer blocks a CPU thread (no watchdog) instead of a command buffer. Use it
for warmup / the first forward; steady-state decode leaves it off.
"""
from __future__ import annotations

import contextlib
import os

import mlx.core as mx

_ALWAYS = os.environ.get("DSV41_SYNC_COLLECTIVES", "0") == "1"
_depth = [0]


def active() -> bool:
    return _ALWAYS or _depth[0] > 0


@contextlib.contextmanager
def sync_collectives(enabled: bool = True):
    """Host-synchronise every collective inside this block (see module doc)."""
    if not enabled:
        yield
        return
    _depth[0] += 1
    try:
        yield
    finally:
        _depth[0] -= 1


_WARM_CALLS = int(os.environ.get("DSV41_SYNC_WARM_CALLS", "2"))
_seen: dict = {}


def warm_guard(key):
    """Context manager: host-sync collectives for the first ``_WARM_CALLS``
    calls seen with ``key`` (a shape signature). That is when each rank
    JIT-builds its Metal kernels for the shape, i.e. when the rank skew that
    trips the watchdog happens. Later calls with the same key run unsynced.
    ``DSV41_SYNC_WARM_CALLS=0`` disables the guard."""
    c = _seen.get(key, 0)
    if c >= _WARM_CALLS:
        return contextlib.nullcontext()
    _seen[key] = c + 1
    return sync_collectives()


def all_sum(x, group=None):
    y = mx.distributed.all_sum(x, group=group)
    if active():
        mx.eval(y)
    return y
