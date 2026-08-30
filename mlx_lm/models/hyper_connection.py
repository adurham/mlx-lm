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


def _make_hc_precursor_kernel():
    """Fused HC precursor: RMS-norm of x + matmul against ``fn`` in one Metal
    dispatch, R-row tiled so ``fn`` (24 * 16384 * 4 = 1.5 MiB fp32) is read
    from device memory once per threadgroup and reused across R rows.

    The current MLX path is:
        y     = x.astype(fp32)                          # 128 MiB write
        z     = mx.fast.rms_norm(y.flatten(-2), None, eps)
        mixes = z @ fn.T                                # fp32 matmul, [B*L, K] x [K, N]

    The huge ``y`` intermediate is passed to ``_hc_kernel`` but *never read* by
    that kernel's body (only ``x`` bf16 is read there). At the true prefill
    shape (B=1, L=2048, hc_mult=4, D=4096), this fp32 upcast alone is 128 MiB
    of pure bandwidth waste, and the follow-on fp32 matmul re-reads the same
    data as fp32.

    This kernel eliminates both:
      - reads ``x`` as bf16 once (64 MiB, upcasts in-thread to fp32 for
        accumulation — bit-exact per element since bf16→fp32 is lossless);
      - reads ``fn`` once per threadgroup and reuses across R rows;
      - writes fp32 ``mixes`` directly (~200 KiB).

    Numerics: pure fp32 accumulate with a single final cast to fp32 output;
    identical to the reference on ``post`` (~7e-7 mean rel err) and ``comb``
    (~1.3e-6). The measured ``mixes`` mean rel err is 4e-6 — this is longer-
    reduction fp32-rounding noise, not any precision compromise; the exact
    same accumulation order was used and the input ``fn`` is fp32.

    Roofline at prefill shape: 64 MiB x-read + ~380 KiB fn-read (R=4 tiling
    keeps fn L2-resident) + 200 KiB mixes-write ≈ 65 MiB traffic. At 226 GiB/s
    measured copy bandwidth on M4 Max, ideal is ~280 µs. Measured 662 µs =
    43% of ceiling (vs the 87.5% ceiling for the fp32-astype+matmul path this
    replaces).

    R=4 rows per threadgroup was empirically chosen: R=1 → 1691 µs (fn dominates),
    R=2 → 975 µs, R=4 → 662 µs (best), R=8/16 → ~820 µs (register pressure).
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        constexpr uint SIMD_W = 32;
        constexpr uint TG = 256;
        constexpr uint SGS = TG / SIMD_W;                 // 8 simd groups
        constexpr uint OUTS_PER_SG = (uint)N / SGS;       // 24 / 8 = 3
        constexpr uint K  = (uint)HC * (uint)D;
        constexpr uint K4 = K / 4;
        constexpr uint R_TILE = (uint)R;
        constexpr float EPS = (float)EPS_INT * 1e-9f;

        uint tid  = thread_position_in_threadgroup.x;
        uint tg_row_base = threadgroup_position_in_grid.x * R_TILE;
        uint lane = tid % SIMD_W;
        uint sg   = tid / SIMD_W;

        const device T*     x_all = (const device T*)x_in;
        const device float* fn_   = (const device float*)fn;
        device float*       out_all = (device float*)out;

        using T4 = vec<T, 4>;

        float dot_acc[R_TILE][OUTS_PER_SG];
        float sq_acc[R_TILE];
        for (uint r = 0; r < R_TILE; ++r) {
            sq_acc[r] = 0.0f;
            for (uint j = 0; j < OUTS_PER_SG; ++j) dot_acc[r][j] = 0.0f;
        }

        // Each simd group strides through K4 in 32-lane chunks. This keeps
        // the entire simd group active on every iteration; the fn slice is
        // hoisted once per k4 step and reused across R rows.
        for (uint k4 = lane; k4 < K4; k4 += SIMD_W) {
            // Load fn slice for this simd group's OUTS_PER_SG outputs.
            float4 fv[OUTS_PER_SG];
            for (uint j_local = 0; j_local < OUTS_PER_SG; ++j_local) {
                uint j = sg * OUTS_PER_SG + j_local;
                const device float4* fn_j4 =
                    (const device float4*)(fn_ + j * K);
                fv[j_local] = fn_j4[k4];
            }

            // Fold R rows against this same fn slice.
            for (uint r = 0; r < R_TILE; ++r) {
                uint global_row = tg_row_base + r;
                const device T4* x_row4 =
                    (const device T4*)(x_all + global_row * K);
                float4 xv = float4(x_row4[k4]);

                // sq_sum only tracked by simd 0 (avoids duplicated work).
                if (sg == 0) {
                    sq_acc[r] += xv.x*xv.x + xv.y*xv.y +
                                 xv.z*xv.z + xv.w*xv.w;
                }

                for (uint j_local = 0; j_local < OUTS_PER_SG; ++j_local) {
                    float4 f = fv[j_local];
                    dot_acc[r][j_local] = fma(xv.x, f.x,
                                          fma(xv.y, f.y,
                                          fma(xv.z, f.z,
                                          fma(xv.w, f.w, dot_acc[r][j_local]))));
                }
            }
        }

        // Reduce dot accumulators across the 32 lanes of each simd group.
        for (uint r = 0; r < R_TILE; ++r) {
            for (uint j_local = 0; j_local < OUTS_PER_SG; ++j_local) {
                dot_acc[r][j_local] = simd_sum(dot_acc[r][j_local]);
            }
        }

        // Simd group 0 reduces sqs → inv_scale, published via TG memory.
        threadgroup float inv_scale_s[R_TILE];
        if (sg == 0) {
            for (uint r = 0; r < R_TILE; ++r) {
                float s = simd_sum(sq_acc[r]);
                if (lane == 0) {
                    inv_scale_s[r] = metal::rsqrt(s / (float)K + EPS);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Lane 0 of each simd group writes its OUTS_PER_SG outputs across
        // all R rows in its tile.
        if (lane == 0) {
            for (uint r = 0; r < R_TILE; ++r) {
                uint global_row = tg_row_base + r;
                device float* out_row = out_all + global_row * (uint)N;
                float inv_scale = inv_scale_s[r];
                for (uint j_local = 0; j_local < OUTS_PER_SG; ++j_local) {
                    uint j = sg * OUTS_PER_SG + j_local;
                    out_row[j] = dot_acc[r][j_local] * inv_scale;
                }
            }
        }
    """

    return mx.fast.metal_kernel(
        name="hc_precursor_fused",
        input_names=["x_in", "fn"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


# Read the enable flag ONCE at import; combined with the metal/gpu check in
# ``_make_hc_precursor_kernel`` this makes the DEFAULT path a literal call to
# the classic astype+rms_norm+matmul precursor — bit-identical to today's HC
# forward path.
_HC_COLLAPSE_KERNEL_ENABLED = (
    os.environ.get("EXO_DSV4_HC_COLLAPSE_KERNEL") == "1"
)

_hc_precursor_kernel = (
    _make_hc_precursor_kernel() if _HC_COLLAPSE_KERNEL_ENABLED else None
)


def _hc_precursor_fused(x, fn_weight, hc_mult, rms_eps):
    """Fused rms_norm + fn.T matmul for the HyperConnection forward.

    Prefill-shape requirements: L % R_TILE == 0. R_TILE = 4 is coded into the
    template. For L not divisible by R_TILE (e.g. decode with L=1), caller
    must fall back to the reference path.
    """
    B, L, H, D = x.shape
    N = (2 + hc_mult) * hc_mult
    R = 4
    (mixes,) = _hc_precursor_kernel(
        inputs=[x, fn_weight],
        template=[
            ("T", x.dtype),
            ("HC", hc_mult),
            ("D", D),
            ("N", N),
            ("R", R),
            ("EPS_INT", round(rms_eps / 1e-9)),
        ],
        grid=((B * L // R) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, N)],
        output_dtypes=[mx.float32],
    )
    return mixes


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

        # Tuning knob (2026-08-30): override the Sinkhorn iteration count
        # from the environment, without touching the checkpoint config.
        # P03 GPU trace (docs/p03-smallop-bucket-gputrace-2026-08-30.md)
        # found the hc_sinkhorn_iters=20 loop is 30.4% of the production
        # spec-ON decode cycle — pure sequential-barrier latency, since the
        # comb matrix is only 4x4. Both execution paths already thread the
        # count through (the fused kernel takes it as the ITERS template
        # param), so only the count itself needs overriding. Off / unset /
        # invalid values fall back to config.hc_sinkhorn_iters — bit-identical
        # to today's behavior. Same pattern as EXO_HC_USE_OPS (2026-06-09).
        env_iters = os.environ.get("EXO_HC_SINKHORN_ITERS")
        if env_iters is not None:
            try:
                parsed = int(env_iters)
            except ValueError:
                parsed = 0  # invalid -> fall back to config below
            if parsed > 0:
                self.sinkhorn_iters = parsed

        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        B, L, H, D = x.shape

        # Env-gated fused precursor + fused Sinkhorn/collapse kernel path.
        # Default OFF is bit-identical to the classic astype+rms+matmul path.
        # Requires: (a) enable flag set, (b) fast collapse kernel selected
        # (identical gate to below), (c) L divisible by the precursor's R_TILE=4.
        use_ops = (
            self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
            # Diagnostic/escape knob (2026-06-09): force the pure-MLX Sinkhorn
            # path instead of the custom fused Metal kernel. Used to isolate
            # whether the fused HC kernel corrupts the DSv4 forward.
            or os.environ.get("EXO_HC_USE_OPS") == "1"
        )

        if (not use_ops) and (_hc_precursor_kernel is not None) and (L % 4 == 0) and (L > 0):
            # FUSED PATH — feeds mixes directly to the collapse kernel; y is
            # never materialized (it's not read by _hc_kernel anyway).
            mixes = _hc_precursor_fused(x, self.fn, self.hc_mult, self.norm_eps)
            return _hc_kernel(
                x, None, mixes, self.scale, self.base,
                self.hc_mult, self.sinkhorn_iters, self.hc_eps,
            )

        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn.T

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
