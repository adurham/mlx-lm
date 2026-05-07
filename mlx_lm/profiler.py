# Copyright © 2025 Apple Inc.
"""Generic profiler-hook interface for mlx-lm models.

A ``ProfilerHook`` lets external callers (e.g. a deployment runner) attach
optional, opt-in instrumentation to model code without the model file
itself reading environment variables. Models call the module-level
:func:`span` / :func:`finalize` / :func:`on_layer_start` / :func:`on_layer_end`
helpers; with no hook registered they short-circuit to no-ops.

The hook is intentionally minimal — three concerns the model code cannot
reach without coupling to a profiler implementation:

  * :meth:`span` — name a region whose wall time should be measured.
  * :meth:`finalize` — force ``mx.eval`` so a span boundary reflects real
    GPU wall time instead of graph-build time.
  * :meth:`on_layer_start` / :meth:`on_layer_end` — coarser per-layer
    callbacks (used by per-layer memory snapshots, etc.).

Two reference implementations live alongside:

  * :class:`SpanProfilerHook` — accumulates count/total/min/max ns per span,
    dumps on SIGUSR1 / atexit. Equivalent to the previous MiniMax tracer.
  * :class:`MemorySnapshotHook` — wraps another hook and adds Metal
    memory snapshots at layer boundaries. Equivalent to the previous
    qwen3_5 ``EXO_PROFILE_LAYERS`` path.
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
from typing import Generator, Optional, Protocol

import mlx.core as mx


class ProfilerHook(Protocol):
    """Hook surface that mlx-lm model code may call."""

    def span(self, name: str) -> "contextlib.AbstractContextManager[None]": ...

    def finalize(self, x: mx.array) -> mx.array: ...

    def on_layer_start(self, layer_idx: int, kind: str) -> None: ...

    def on_layer_end(self, layer_idx: int, kind: str) -> None: ...


_registered_hook: Optional[ProfilerHook] = None


# Singleton reusable no-op context manager so ``span()`` doesn't allocate
# a new generator-based CM per call when no hook is active. Hot decode
# paths can call span() 360+ times per token; the @contextmanager
# decorator's per-call overhead (build a generator, advance to yield,
# advance again on exit) was ~3-5 µs/call. With 60 layers × 6 spans/layer
# the overhead added up to ~1.8 ms/step on DSv4 — large enough to show
# up in the GPU% probe as raw fwd_build time.
class _NullSpan:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NULL_SPAN: _NullSpan = _NullSpan()


def register(hook: ProfilerHook) -> None:
    """Install a profiler hook. Replaces any previously registered hook."""
    global _registered_hook
    _registered_hook = hook


def unregister() -> None:
    """Remove the currently registered profiler hook (back to no-op)."""
    global _registered_hook
    _registered_hook = None


def get() -> Optional[ProfilerHook]:
    """Return the currently registered hook, or ``None``."""
    return _registered_hook


@contextmanager
def _null_span() -> Generator[None, None, None]:
    yield


def span(name: str) -> "contextlib.AbstractContextManager[None]":
    """Enter a named profiling region; no-op when no hook is registered."""
    hook = _registered_hook
    if hook is None:
        return _NULL_SPAN
    return hook.span(name)


def finalize(x: mx.array) -> mx.array:
    """Force ``mx.eval(x)`` if a hook is registered; otherwise no-op."""
    if _registered_hook is None:
        return x
    return _registered_hook.finalize(x)


def on_layer_start(layer_idx: int, kind: str) -> None:
    """Notify the registered hook that a layer ``layer_idx`` (``kind``) is starting."""
    hook = _registered_hook
    if hook is None:
        return
    hook.on_layer_start(layer_idx, kind)


def on_layer_end(layer_idx: int, kind: str) -> None:
    """Notify the registered hook that a layer ``layer_idx`` (``kind``) just finished."""
    hook = _registered_hook
    if hook is None:
        return
    hook.on_layer_end(layer_idx, kind)


# ── Reference implementations ──────────────────────────────────────────────


class SpanStats:
    """Thread-safe accumulator: count / total-ns / min-ns / max-ns per span name."""

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
        snap = (
            self.snapshot_and_reset()
            if reset
            else {k: dict(v) for k, v in self._data.items()}
        )
        if not snap:
            return
        # Wall-time denominator: sum of TOP-LEVEL spans only (entries that
        # don't contain a ".") so child spans aren't double-counted into the
        # share-of-wall-time column.
        wall_ns = sum(v["total_ns"] for k, v in snap.items() if "." not in k)
        if wall_ns == 0:
            wall_ns = sum(v["total_ns"] for v in snap.values())
        lines = [f"[PROFILER pid={os.getpid()}] span breakdown:"]
        header = (
            f"  {'span':<28s} {'n':>8s} {'avg_us':>10s} {'min_us':>10s} "
            f"{'max_us':>10s} {'total_ms':>10s} {'%':>6s}"
        )
        lines.append(header)
        for name, v in sorted(snap.items(), key=lambda kv: -kv[1]["total_ns"]):
            n = v["n"]
            avg_us = (v["total_ns"] / n / 1000.0) if n else 0.0
            min_us = v["min_ns"] / 1000.0
            max_us = v["max_ns"] / 1000.0
            total_ms = v["total_ns"] / 1e6
            pct = (100.0 * v["total_ns"] / wall_ns) if wall_ns else 0.0
            lines.append(
                f"  {name:<28s} {n:>8d} {avg_us:>10.2f} {min_us:>10.2f} "
                f"{max_us:>10.2f} {total_ms:>10.2f} {pct:>5.1f}%"
            )
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()


class SpanProfilerHook:
    """Hook that records per-span wall time and forces ``mx.eval`` at boundaries.

    Wall-time accuracy requires forcing MLX evaluation at span boundaries:
    ``finalize`` calls ``mx.eval`` so the ``perf_counter`` delta reflects
    real kernel time rather than graph-build time.
    """

    def __init__(self) -> None:
        self.stats: SpanStats = SpanStats()

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.stats.record(name, time.perf_counter_ns() - start)

    def finalize(self, x: mx.array) -> mx.array:
        mx.eval(x)
        return x

    def on_layer_start(self, layer_idx: int, kind: str) -> None:
        return

    def on_layer_end(self, layer_idx: int, kind: str) -> None:
        return

    def dump(self, *, reset: bool = True) -> None:
        self.stats.dump(reset=reset)


class MemorySnapshotHook:
    """Per-layer Metal memory-snapshot hook for prefill profiling.

    ``level=1`` prints active/peak Metal memory after every layer.
    ``level>=2`` additionally calls ``on_layer_start`` (used by callers
    that want pre-layer snapshots too).

    ``span`` / ``finalize`` are no-ops here — combine with
    :class:`SpanProfilerHook` via :class:`CompositeHook` if you want both.
    """

    def __init__(self, level: int = 1) -> None:
        self.level = max(0, int(level))
        self._base_active: float = 0.0

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        yield

    def finalize(self, x: mx.array) -> mx.array:
        return x

    def _snapshot(self, label: str) -> None:
        mx.eval(mx.zeros(1))
        active = mx.metal.get_active_memory() / 1024**3
        peak = mx.metal.get_peak_memory() / 1024**3
        sys.stderr.write(
            f"[PROFILER {label}] active={active:.3f} GB  peak={peak:.3f} GB\n"
        )
        sys.stderr.flush()
        mx.metal.reset_peak_memory()
        self._base_active = active

    def on_layer_start(self, layer_idx: int, kind: str) -> None:
        if self.level >= 2:
            self._snapshot(f"L{layer_idx}({kind}) input")

    def on_layer_end(self, layer_idx: int, kind: str) -> None:
        if self.level >= 1:
            self._snapshot(f"L{layer_idx}({kind}) end")


class CompositeHook:
    """Fan-out hook: dispatches every call to each component in order."""

    def __init__(self, *hooks: ProfilerHook) -> None:
        self._hooks: tuple[ProfilerHook, ...] = hooks

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            for h in self._hooks:
                stack.enter_context(h.span(name))
            yield

    def finalize(self, x: mx.array) -> mx.array:
        for h in self._hooks:
            x = h.finalize(x)
        return x

    def on_layer_start(self, layer_idx: int, kind: str) -> None:
        for h in self._hooks:
            h.on_layer_start(layer_idx, kind)

    def on_layer_end(self, layer_idx: int, kind: str) -> None:
        for h in self._hooks:
            h.on_layer_end(layer_idx, kind)


def install_signal_dump(hook: SpanProfilerHook) -> None:
    """Wire SIGUSR1 + atexit to dump a :class:`SpanProfilerHook`'s stats.

    Idempotent across calls — re-registers the most recent hook.
    """

    def _sig_dump(_signum: int, _frame: object) -> None:
        hook.dump(reset=True)

    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGUSR1, _sig_dump)
    atexit.register(lambda: hook.dump(reset=False))
    sys.stderr.write(
        f"[PROFILER pid={os.getpid()}] enabled; dump on SIGUSR1 or exit.\n"
    )
    sys.stderr.flush()
