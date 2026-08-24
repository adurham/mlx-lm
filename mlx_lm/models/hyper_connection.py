# Copyright © 2026 Apple Inc.

from typing import Tuple

import os
import mlx.core as mx
import mlx.nn as nn


def _make_hc_sinkhorn_collapse_kernel():
    """Fused sinkhorn + collapse: eliminates one dispatch per HC cycle.

    1. BRANCHLESS SINKHORN: all 32 lanes in simd group 0 execute identical
       instructions. Lanes >= HC use multiplicative mask (active=0) instead
       of divergent branches — eliminates SIMD serialization.
    2. PARALLEL SINKHORN: lanes 0-3 each own one comb row. Column norm
       via simd_sum() — free SIMD shuffle.
    3. NATIVE bfloat4 LOADS: single 64-bit load yields 4 bfloat16 values;
       cast to float4 is a free hardware conversion.
    4. FMA CHAINS: collapse uses fused multiply-add for 3 of 4 terms.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0
        //   All 32 lanes execute identical instructions. Lanes >= HC
        //   compute on clamped indices but multiply by active=0, so they
        //   contribute zero to simd_sum. No divergent branches in the loop.
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            // Pre/post sigmoids: all lanes compute, only active lanes write
            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // Comb softmax: load + mask. Inactive lanes load row 0 (safe)
            // but multiply by active=0 so they hold zeros.
            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            // Initial column normalization
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            // Sinkhorn iterations: zero branches in the loop body
            for (int iter = 1; iter < ITERS; ++iter) {
                // Row norm + re-clamp inactive lanes
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;

                // Col norm via simd_sum
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)
                ) + EPS);
                r *= col_inv;
            }

            if (lane < (uint)HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // PHASE 2: Collapse — all 256 threads, vectorized
        // ================================================================
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in
                                         + row * (HC * D);
        device U*       out_row = (device U*)collapsed
                                         + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = U4(result);
        }

        // Scalar tail for D not divisible by 4
        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                      + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
            out_row[d] = (U)val;
        }
        #endif
    """

    return mx.fast.metal_kernel(
        name="hc_sinkhorn_collapse",
        input_names=["x_in", "mixes", "scale", "base"],
        output_names=["collapsed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


def _hc_kernel(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    B, L, H, D = x.shape

    return _hc_sinkhorn_collapse_kernel(
        inputs=[x, mixes, scale, base],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("D", D),
            ("EPS_INT", round(eps / 1e-9)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, hc_mult), (B, L, hc_mult, hc_mult)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(
        mixes, scale, base, hc_mult, sinkhorn_iters, eps
    )
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, L, H, D = x.shape
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T

        use_ops = (
            self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
            # Diagnostic/escape knob (2026-06-09): force the pure-MLX Sinkhorn
            # path instead of the custom fused Metal kernel. Used to isolate
            # whether the fused HC kernel corrupts the DSv4 forward.
            or os.environ.get("EXO_HC_USE_OPS") == "1"
        )
        hc_func = _hc_ops if use_ops else _hc_kernel

        return hc_func(
            x,
            y,
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )


@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def _make_hc_expand_kernel():
    """Fused HyperConnection EXPAND: numerically equivalent to
    ``_hc_expand_op`` (fp32 accumulate, single cast to output dtype at
    store), but issued as ONE Metal kernel — avoids the fp32
    materialization of ``residual`` (a 134MB intermediate at the
    production shape [1,2048,4,4096]) and the separate broadcast/matmul
    intermediates, cutting traffic to the fused ideal:
    read ``x`` bf16 once, read ``residual`` bf16 once, read tiny
    ``post`` / ``comb`` fp32 once, write ``out`` bf16 once.

    Per-token dispatch: 256-thread threadgroup, one row per token.
    ``post`` (``HC`` fp32 values) and ``comb`` (``HC*HC`` fp32 values)
    are staged into threadgroup memory so the D4 strided loop reuses
    them from fast on-chip storage. Each thread loads x[d4] once and
    residual[m, d4] for all m, then writes all HC output streams for
    that slice — maximizes register/threadgroup reuse of every loaded
    byte across the HC*HC FMAs.

    Requires ``D`` divisible by 4 (float4-vectorized loads/stores).
    Production ``D`` is 4096; the caller must fall back to the ops path
    for other shapes.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    # Notes on address spaces:
    #   * ``x_in`` / ``residual`` are always large; a ``const device T*``
    #     cast is safe and needed for vec<T,4> loads.
    #   * ``post`` / ``comb`` may be placed in ``constant`` address space
    #     by MLX when the total number of elements is small (decode
    #     shapes). Index them through the auto-generated accessor
    #     without an explicit ``const device`` cast to work in both
    #     regimes.
    source = """
        uint tid = thread_position_in_threadgroup.x;
        uint row = threadgroup_position_in_grid.x;  // B*L row index

        // Stage tiny per-token post/comb into threadgroup memory. Every
        // D4 iteration reads them repeatedly (HC*HC FMAs per iter), so
        // keeping them on-chip removes the loads from the inner loop.
        threadgroup float post_s[HC];
        threadgroup float comb_s[HC * HC];  // comb[m,n] at m*HC + n

        if (tid < (uint)HC) {
            post_s[tid] = post[row * (uint)HC + tid];
        }
        if (tid < (uint)(HC * HC)) {
            comb_s[tid] = comb[row * (uint)HC * (uint)HC + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        constexpr uint D4 = (uint)D / 4;

        const device T* x_base   = (const device T*)x_in    + row * (uint)D;
        const device T* res_base = (const device T*)residual
                                        + row * (uint)HC * (uint)D;
        device U*       out_base = (device U*)out
                                        + row * (uint)HC * (uint)D;
        const device T4* x_row = (const device T4*)x_base;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 xv = float4(x_row[d4]);

            // Load residual[m, d4] into registers for m=0..HC-1.
            float4 rv[HC];
            for (int m = 0; m < HC; ++m) {
                const device T4* res_m =
                    (const device T4*)(res_base + (uint)m * (uint)D);
                rv[m] = float4(res_m[d4]);
            }

            // out[n] = post[n]*x + sum_m comb[m,n]*residual[m]
            // fp32 accumulate; single U-cast at the store.
            for (int n = 0; n < HC; ++n) {
                float4 acc = float4(post_s[n]) * xv;
                for (int m = 0; m < HC; ++m) {
                    acc = fma(float4(comb_s[m * HC + n]), rv[m], acc);
                }
                device U4* out_n =
                    (device U4*)(out_base + (uint)n * (uint)D);
                out_n[d4] = U4(acc);
            }
        }
    """

    return mx.fast.metal_kernel(
        name="hc_expand_fused",
        input_names=["x_in", "residual", "post", "comb"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


# Read the enable flag ONCE at import; combined with the metal/gpu check
# in ``_make_hc_expand_kernel`` this makes the DEFAULT path a literal
# call to ``_hc_expand_op`` — bit-identical to the pre-kernel behavior.
_HC_EXPAND_KERNEL_ENABLED = (
    os.environ.get("EXO_DSV4_HC_EXPAND_KERNEL") == "1"
)

_hc_expand_kernel = (
    _make_hc_expand_kernel() if _HC_EXPAND_KERNEL_ENABLED else None
)


def hc_expand(x, residual, post, comb):
    # Default OFF: EXO_DSV4_HC_EXPAND_KERNEL unset ⇒ this is the
    # original compiled call, byte-for-byte.
    if _hc_expand_kernel is None:
        return _hc_expand_op(x, residual, post, comb)

    # Kernel is float4-vectorized; guard rare off-shape ``D``. Production
    # is D=4096 (hidden_size, replicated under TP) — always divisible.
    B, L, hc_mult, D = residual.shape
    if D % 4 != 0 or L == 0:
        return _hc_expand_op(x, residual, post, comb)

    (out,) = _hc_expand_kernel(
        inputs=[x, residual, post, comb],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc_mult),
            ("D", D),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, hc_mult, D)],
        output_dtypes=[x.dtype],
    )
    return out


class HyperHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)
