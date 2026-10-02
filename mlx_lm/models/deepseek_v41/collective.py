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
import inspect
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


def _raw(fn):
    """The unwrapped MLX collective. Another model in this process (exo's
    deepseek_v4) wraps mx.distributed.* to downcast fp32 payloads to bf16;
    this model sends exact fp32 values (token ids in combine_argmax), so a
    wrapper silently corrupts them."""
    return inspect.unwrap(fn)


# DSV41_MOE_ALLSUM_BF16=1: shrink the MoE routed-sum collective's payload from
# fp32 to bf16 (half the wire bytes; on the 2x M4 Max JACCL link the fp32 MoE
# all_sum costs 141 us/call against 41 us for the same-size bf16 attention
# all_sum -- D1, scratch/d1/REPORT.md). NOT BIT-EXACT by construction: one
# rank's partial is rounded to bf16 before the reduction and the reduced result
# is widened back to fp32 after, so each element carries up to the bf16
# round-to-nearest relative error (2^-8) plus the reduction's own rounding. Do
# not use it for token ids or anything that must round-trip exactly; the MoE
# routed sum is a float sum where that error is acceptable (and the same
# downcast is what deepseek_v4's process-wide wrapper has always done to fp32
# collectives). Default OFF: flag off calls the exact fp32 collective.
_ALLSUM_BF16 = os.environ.get("DSV41_MOE_ALLSUM_BF16", "0") == "1"


def all_sum(x, group=None):
    y = _raw(mx.distributed.all_sum)(x, group=group)
    if active():
        mx.eval(y)
    return y


def all_sum_lowp(x, group=None):
    """``all_sum`` for fp32 float payloads that tolerate bf16 rounding.

    Flag OFF (default): exactly ``all_sum(x, group=group)`` -- same op, same
    payload, byte-identical result.
    Flag ON: cast the fp32 payload to bf16 for the transfer, widen the reduced
    result back to the input dtype. The host-sync contract (``active()``) is
    the one ``all_sum`` implements: the synchronised array is the collective's
    own output, before the widening cast.

    Only fp32 payloads are touched, so a non-fp32 input (and every caller that
    does not opt in) is unaffected.
    """
    if not _ALLSUM_BF16 or x.dtype != mx.float32:
        return all_sum(x, group=group)
    return all_sum(x.astype(mx.bfloat16), group=group).astype(x.dtype)
