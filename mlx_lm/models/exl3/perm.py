# Vendored from PonyExl3 (https://github.com/beamivalice/PonyExl3; Apache-2.0,
# see LICENSE and NOTICE alongside this file).
# Local change vs upstream: absolute `ponyexl3.*` imports rewritten to relative
# imports. Numerical code is unchanged unless noted below.
"""Tensor-core tile permutation — MLX ``take`` path."""

from __future__ import annotations

import mlx.core as mx

from .ref.perm import tensor_core_perm_inverse

_inv_perm_mx: mx.array | None = None


def _inv_perm() -> mx.array:
    global _inv_perm_mx
    if _inv_perm_mx is None:
        _inv_perm_mx = mx.array(tensor_core_perm_inverse())
    return _inv_perm_mx


def kernel_order_to_row_major_mlx(values: mx.array) -> mx.array:
    """Reorder 256-element tile from kernel layout to row-major 16×16."""
    inv = _inv_perm()
    out: mx.array = mx.take(values, inv, axis=-1)
    shaped: mx.array = mx.reshape(out, (*values.shape[:-1], 16, 16))
    return shaped
