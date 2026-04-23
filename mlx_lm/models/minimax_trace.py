# Copyright © 2025 Apple Inc.
"""Span-level wall-time profiler for MiniMax decode.

Gated on ``EXO_MINIMAX_TRACE=1``. Zero-cost when disabled.

The tracer is deliberately self-contained (no ``exo`` imports) so the
``mlx-lm`` fork can carry it without creating a package cycle. It records
count / total-ns / min-ns / max-ns per named span into a module-level
:class:`SpanStats` singleton, dumps on ``SIGUSR1`` and at interpreter
exit.

Wall-time accuracy requires forcing MLX evaluation at span boundaries.
The :func:`finalize` helper is a no-op when tracing is off and calls
``mx.eval`` when tracing is on — keeping the model code free of
``if ENABLED`` branches.

A second env, ``EXO_MINIMAX_NOOP_ALLSUM=1``, is *not* handled here; it
lives in :mod:`minimax` itself where the all-sum sites are. See that
file for the ceiling experiment.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator

import mlx.core as mx

ENABLED: bool = os.environ.get("EXO_MINIMAX_TRACE", "0") == "1"


class SpanStats:
    __slots__ = ("_lock", "_data")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n": 0, "total_ns": 0, "min_ns": 2**63 - 1, "max_ns": 0}
        )

    def record(self, name: str, duration_ns: int) -> None:
        with self._lock:
            d = self._data[name]
            d["n"] += 1
            d["total_ns"] += duration_ns
            if duration_ns < d["min_ns"]:
                d["min_ns"] = duration_ns
            if duration_ns > d["max_ns"]:
                d["max_ns"] = duration_ns

    def snapshot_and_reset(self) -> dict[str, dict[str, int]]:
        with self._lock:
            snap = {k: dict(v) for k, v in self._data.items()}
            self._data.clear()
        return snap

    def dump(self, *, reset: bool = True) -> None:
        snap = self.snapshot_and_reset() if reset else {
            k: dict(v) for k, v in self._data.items()
        }
        if not snap:
            return
        total_ns = sum(v["total_ns"] for v in snap.values())
        lines = [f"[MINIMAX_TRACE pid={os.getpid()}] span breakdown:"]
        header = f"  {'span':<28s} {'n':>8s} {'avg_us':>10s} {'min_us':>10s} {'max_us':>10s} {'total_ms':>10s} {'%':>6s}"
        lines.append(header)
        for name, v in sorted(snap.items(), key=lambda kv: -kv[1]["total_ns"]):
            n = v["n"]
            avg_us = (v["total_ns"] / n / 1000.0) if n else 0.0
            min_us = v["min_ns"] / 1000.0
            max_us = v["max_ns"] / 1000.0
            total_ms = v["total_ns"] / 1e6
            pct = (100.0 * v["total_ns"] / total_ns) if total_ns else 0.0
            lines.append(
                f"  {name:<28s} {n:>8d} {avg_us:>10.2f} {min_us:>10.2f} {max_us:>10.2f} {total_ms:>10.2f} {pct:>5.1f}%"
            )
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()


STATS = SpanStats()


@contextmanager
def span(name: str) -> Generator[None, None, None]:
    if not ENABLED:
        yield
        return
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        STATS.record(name, time.perf_counter_ns() - start)


def finalize(x: mx.array) -> mx.array:
    """Force MLX to materialize ``x`` when tracing is on; no-op otherwise.

    Call this at the END of a :func:`span` block on the span's output so
    that the ``perf_counter`` delta reflects real kernel wall-time
    instead of graph-build time.
    """
    if ENABLED:
        mx.eval(x)
    return x


if ENABLED:
    def _sig_dump(_signum: int, _frame: object) -> None:
        STATS.dump(reset=True)

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGUSR1, _sig_dump)
    atexit.register(lambda: STATS.dump(reset=False))
    sys.stderr.write(
        f"[MINIMAX_TRACE pid={os.getpid()}] enabled; dump on SIGUSR1 or exit.\n"
    )
    sys.stderr.flush()
