# LOCAL ADDITION (not vendored). Fused hyper-connection kernels for V4.1.
"""Fused hyper-connection path (hc_mult == 4).

V4.1 staggers the coefficients: a sub-layer collapses its input with the
PREVIOUS sub-layer's ``pre`` and computes its own ``(pre, post, comb)`` for
later use. One sub-layer therefore needs:

  1. ``_mix_proj``   -- rms-scaled projection of the flattened stream (compiled);
  2. ``sinkhorn_collapse`` kernel -- split + Sinkhorn on the new mixes AND the
     collapse with the previous ``pre`` (one dispatch);
  3. ``hc_expand`` kernel -- ``out[k] = post[k]*x + sum_j comb[j,k]*res[j]``.

The reference ops path (``hyper_connections.py``) issues ~140 dispatches per
sub-layer; this path issues ~4. Math is the same (fp32, same formulas, same
eps placement); only reduction order differs.
"""

from __future__ import annotations

import mlx.core as mx

_SINKHORN_COLLAPSE_SRC = """
    uint tid  = thread_position_in_threadgroup.x;
    uint row  = threadgroup_position_in_grid.x;
    uint lane = tid % 32;
    uint sg   = tid / 32;
    constexpr int HC = 4;
    constexpr int MIX = (2 + HC) * HC;
    const float EPS = eps[0];

    if (sg == 0) {
        const float s0 = scale[0];
        const float s1 = scale[1];
        const float s2 = scale[2];
        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint ll = metal::min(lane, (uint)(HC - 1));
        const uint mo = row * MIX;

        float pz = mixes[mo + ll] * s0 + base[ll];
        float qz = mixes[mo + HC + ll] * s1 + base[HC + ll];
        float pre_v  = 1.0f / (1.0f + metal::precise::exp(-pz)) + EPS;
        float post_v = 2.0f / (1.0f + metal::precise::exp(-qz));
        if (lane < (uint)HC) {
            pre_out[row * HC + lane]  = pre_v;
            post_out[row * HC + lane] = post_v;
        }

        float4 v;
        for (int c = 0; c < HC; ++c) {
            v[c] = mixes[mo + 2 * HC + ll * HC + c] * s2 + base[2 * HC + ll * HC + c];
        }
        float m = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        float4 e = metal::precise::exp(v - m);
        float4 r = (e / (e.x + e.y + e.z + e.w) + EPS) * active;

        float4 cs = float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w));
        r = r / (cs + EPS);
        for (int it = 1; it < ITERS; ++it) {
            r = r / (r.x + r.y + r.z + r.w + EPS);
            cs = float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w));
            r = r / (cs + EPS);
        }
        if (lane < (uint)HC) {
            for (int c = 0; c < HC; ++c) {
                comb_out[(row * HC + lane) * HC + c] = r[c];
            }
        }
    }

    // collapse with the PREVIOUS sub-layer's pre (independent of the above)
    const float p0 = pre_prev[row * HC + 0];
    const float p1 = pre_prev[row * HC + 1];
    const float p2 = pre_prev[row * HC + 2];
    const float p3 = pre_prev[row * HC + 3];
    const uint xb = row * HC * D;
    for (uint d = tid; d < (uint)D; d += 256) {
        float acc = p0 * float(x_in[xb + d]);
        acc += p1 * float(x_in[xb + D + d]);
        acc += p2 * float(x_in[xb + 2 * D + d]);
        acc += p3 * float(x_in[xb + 3 * D + d]);
        collapsed[row * D + d] = T(acc);
    }
"""

_EXPAND_SRC = """
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    constexpr int HC = 4;
    threadgroup float post_s[HC];
    threadgroup float comb_s[HC * HC];
    if (tid < (uint)HC) post_s[tid] = post[row * HC + tid];
    if (tid < (uint)(HC * HC)) comb_s[tid] = comb[row * HC * HC + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint xb = row * D;
    const uint rb = row * HC * D;
    for (uint d = tid; d < (uint)D; d += 256) {
        float xv = float(x_in[xb + d]);
        float r0 = float(residual[rb + d]);
        float r1 = float(residual[rb + D + d]);
        float r2 = float(residual[rb + 2 * D + d]);
        float r3 = float(residual[rb + 3 * D + d]);
        for (int n = 0; n < HC; ++n) {
            float acc = post_s[n] * xv;
            acc += comb_s[0 * HC + n] * r0;
            acc += comb_s[1 * HC + n] * r1;
            acc += comb_s[2 * HC + n] * r2;
            acc += comb_s[3 * HC + n] * r3;
            out[rb + n * D + d] = T(acc);
        }
    }
"""

_kernels: dict = {}


def _kernel(name: str):
    k = _kernels.get(name)
    if k is None:
        if name == "sc":
            k = mx.fast.metal_kernel(
                name="dsv41_hc_sinkhorn_collapse",
                input_names=["x_in", "mixes", "scale", "base", "pre_prev", "eps"],
                output_names=["collapsed", "pre_out", "post_out", "comb_out"],
                source=_SINKHORN_COLLAPSE_SRC,
                ensure_row_contiguous=True,
            )
        else:
            k = mx.fast.metal_kernel(
                name="dsv41_hc_expand",
                input_names=["x_in", "residual", "post", "comb"],
                output_names=["out"],
                source=_EXPAND_SRC,
                ensure_row_contiguous=True,
            )
        _kernels[name] = k
    return k


@mx.compile
def _mix_proj(x: mx.array, fn: mx.array, norm_eps: mx.array) -> mx.array:
    xf = x.reshape(*x.shape[:2], -1).astype(mx.float32)
    rs = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + norm_eps)
    return (xf @ fn.T) * rs


def mixes_and_collapse(x, fn, scale, base, pre_prev, iters: int, norm_eps: float,
                       hc_eps: float):
    """x [b,s,4,d], pre_prev [b,s,4] -> (collapsed [b,s,d], pre, post, comb)."""
    b, s, hc, d = x.shape
    mixes = _mix_proj(x, fn, mx.array(norm_eps, dtype=mx.float32))
    rows = b * s
    outs = _kernel("sc")(
        inputs=[x, mixes, scale.astype(mx.float32), base.astype(mx.float32),
                pre_prev.astype(mx.float32), mx.array([hc_eps], dtype=mx.float32)],
        template=[("T", x.dtype), ("D", d), ("ITERS", iters)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, s, d), (b, s, hc), (b, s, hc), (b, s, hc, hc)],
        output_dtypes=[x.dtype, mx.float32, mx.float32, mx.float32],
    )
    return outs[0], outs[1], outs[2], outs[3]


def hc_expand(h, residual, post, comb):
    """out[k] = post[k]*h + sum_j comb[j,k]*residual[j]; returns h.dtype."""
    b, s, hc, d = residual.shape
    (out,) = _kernel("ex")(
        inputs=[h, residual, post, comb],
        template=[("T", h.dtype), ("D", d)],
        grid=(b * s * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, s, hc, d)],
        output_dtypes=[h.dtype],
    )
    return out
