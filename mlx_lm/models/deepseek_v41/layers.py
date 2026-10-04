# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""Core layers: RMSNorm, YaRN rope with the inverse path, clamped SwiGLU.

Transcribed from the reference ``model.py``. Details that matter:

* ``rms_norm_eps`` is ``1e-20`` in the release — effectively zero, kept exact;
* rope pairs *adjacent* elements as (real, imag) — not split halves — and is
  applied to the **last** ``rope_head_dim`` channels only;
* attention applies the rope **inverse** (conjugate) to its output before the
  grouped output projection, which is why the conjugate path exists.
"""

from __future__ import annotations

import math

import os

import mlx.core as mx
import mlx.nn as nn

# Compile the small elementwise helpers (same ops, fewer launches; bit-exact).
_COMPILE = os.environ.get("DSV41_COMPILE_OPS", "1") == "1"


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, returning the input dtype (matches reference)."""

    def __init__(self, dim: int, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        if _COMPILE:
            return _rms_c(x, self.weight, self.eps)
        return _rms(x, self.weight, self.eps)


def _rms(x, weight, eps):
    dtype = x.dtype
    xf = x.astype(mx.float32)
    var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    return (weight * xf).astype(dtype)


def rope_freqs(dim: int, original_seq_len: int, base: float, factor: float,
               beta_fast: int, beta_slow: int) -> mx.array:
    """The YaRN-adjusted ``[dim//2]`` fp32 frequency (angles per token position).

    Factored out of :func:`precompute_freqs_cis` so per-call RoPE multiplies this
    ONE vector by an arbitrary position tensor. Layers used to cache a
    ``[seqlen, dim//2]`` cos/sin table per attention layer (~21 GiB at 1M
    context); now each layer holds only this 32-element vector (128 B), and
    cos/sin are computed for just the positions a call actually needs.

    ``original_seq_len == 0`` disables YaRN (pure sliding-window layers use the
    base theta). Bit-exact: this is the identical expression the table builder
    used, so a per-position value is unchanged.
    """
    def corrected_dim(rotations):
        return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = mx.clip((mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def cos_sin_at(freqvec: mx.array, positions: mx.array) -> tuple[mx.array, mx.array]:
    """``cos``/``sin`` of ``positions`` against a ``[dim//2]`` frequency vector.

    ``positions`` may be any int/float token-position array (contiguous for
    query rows, the compressor's ``(g0 + arange(g)) * ratio`` for latents) ->
    ``([..., dim//2], [...])`` fp32. Bitwise-equal to the corresponding rows of
    the old :func:`precompute_freqs_cis` table: both compute ``t * freq`` in
    fp32 then cos/sin elementwise, so a value depends only on its position.
    """
    ang = positions.astype(mx.float32)[:, None] * freqvec[None, :]
    return mx.cos(ang), mx.sin(ang)


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int, base: float,
                         factor: float, beta_fast: int, beta_slow: int):
    """YaRN-scaled rotary frequencies as a (cos, sin) pair of shape [seqlen, dim//2].

    Kept for callers/tests that still want a dense table; it is now a thin
    wrapper over :func:`rope_freqs` with the same bit values.
    """
    freqs = rope_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow)
    t = mx.arange(seqlen, dtype=mx.float32)
    ang = t[:, None] * freqs[None, :]
    return mx.cos(ang), mx.sin(ang)


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False) -> mx.array:
    """Rotate ``x`` [..., s, (h,) d] by adjacent-pair complex multiplication.

    ``cos``/``sin``: [s, d//2]. ``inverse=True`` conjugates — attention uses this
    on its output to remove the query rotation again.
    """
    dtype = x.dtype
    xf = x.astype(mx.float32)
    shape = xf.shape
    xf = xf.reshape(*shape[:-1], shape[-1] // 2, 2)
    xr, xi = xf[..., 0], xf[..., 1]

    if xr.ndim == 4:            # [b, s, h, d//2]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
    else:                        # [b, s, d//2]
        c, s = cos[None, :, :], sin[None, :, :]
    if inverse:
        s = -s

    out = mx.stack([xr * c - xi * s, xr * s + xi * c], axis=-1)
    return out.reshape(shape).astype(dtype)


def rope_tail(x: mx.array, rd: int, cos: mx.array, sin: mx.array, inverse: bool = False) -> mx.array:
    """Apply rope to the last ``rd`` channels, leave the rest untouched."""
    return mx.concatenate([x[..., :-rd], apply_rotary_emb(x[..., -rd:], cos, sin, inverse)], axis=-1)


def clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    """SwiGLU with the asymmetric clamp: ``up`` two-sided, ``gate`` upper-only. fp32."""
    g = gate.astype(mx.float32)
    u = up.astype(mx.float32)
    if limit > 0:
        u = mx.clip(u, -limit, limit)
        g = mx.minimum(g, limit)
    return (g * mx.sigmoid(g)) * u


_rms_c = mx.compile(_rms)
if _COMPILE:
    _rope_tail_ref = rope_tail
    rope_tail = mx.compile(_rope_tail_ref)
