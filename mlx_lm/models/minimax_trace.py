# Copyright © 2025 Apple Inc.
"""Compatibility shim — re-exports the generic profiler hook surface.

The MiniMax-specific tracer was generalised into :mod:`mlx_lm.profiler`.
Existing call sites importing ``span`` / ``finalize`` from this module
continue to work unchanged; new code should import from
:mod:`mlx_lm.profiler` directly.
"""

from __future__ import annotations

from ..profiler import finalize, span

__all__ = ["finalize", "span"]
