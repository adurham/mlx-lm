import math
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure a indexing and g/beta computation based on whether gating is vectorized
    if vectorized:
        a_comment = "// a: [B, T, Hv, Dk]"
        a_setup = "auto a_ = a + (b_idx * T * Hv + hv_idx) * Dk;"
        a_advance = "a_ += Hv * Dk;"
        # Vectorized: g varies per Dk element, compute shared values before inner loop
        g_compute = (
            "float dt_val = static_cast<float>(dt_bias[hv_idx]);\n"
            "            float neg_exp_A = -exp(static_cast<float>(A_log[hv_idx]));\n"
            "            float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));"
        )
        # Per-element g computation inside inner loop
        g_per_element = (
            "float a_val = static_cast<float>(a_[s_idx]);\n"
            "              float x_g = a_val + dt_val;\n"
            "              float sp = (x_g > 20.0f) ? x_g : log(1.0f + exp(x_g));\n"
            "              float g_val = exp(neg_exp_A * sp);"
        )
    else:
        a_comment = "// a: [B, T, Hv]"
        a_setup = "auto a_ = a + b_idx * T * Hv;"
        a_advance = "a_ += Hv;"
        # Non-vectorized: g is scalar per head, compute once before inner loop
        g_compute = (
            "float a_val = static_cast<float>(a_[hv_idx]);\n"
            "            float dt_val = static_cast<float>(dt_bias[hv_idx]);\n"
            "            float x_g = a_val + dt_val;\n"
            "            float sp = (x_g > 20.0f) ? x_g : log(1.0f + exp(x_g));\n"
            "            float g_val = exp(-exp(static_cast<float>(A_log[hv_idx])) * sp);\n"
            "            float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));"
        )
        g_per_element = ""

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {a_comment}
        {a_setup}
        // b: [B, T, Hv]
        auto b_ = b + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            {g_compute}
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              {g_per_element}
              state[i] = state[i] * g_val;
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_val;

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {a_advance}
          b_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)


def _make_chunkwise_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        // Thread identity
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;   // 0-31 SIMD lane
        auto dv_local = thread_position_in_threadgroup.y;  // 0-3
        auto dv_idx = thread_position_in_grid.y;           // 0-(Dv-1)
        auto tid = dv_local * 32 + dk_idx;                 // 0-127 linear

        constexpr int NPT = Dk / 32;  // n_per_t = 6 for Dk=192
        constexpr int C = CHUNK;
        // Forward substitution replaces Neumann series — zero barriers needed

        // Shared memory
        threadgroup InT sh_k[C * Dk];        // K chunk (half — cast to float for dots)
        threadgroup float sh_cumlog[C];
        threadgroup float sh_beta[C];
        threadgroup float sh_L[C * C];        // L_strict

        // State in registers (same as sequential kernel)
        float state[NPT];
        auto i_st = state_in + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < NPT; i++)
            state[i] = static_cast<float>(i_st[dk_idx * NPT + i]);

        // Base pointers
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto a_ = a + b_idx * T * Hv;
        auto b_ = b + b_idx * T * Hv;

        int n_chunks = (T + C - 1) / C;

        for (int ch = 0; ch < n_chunks; ch++) {
            int cs = ch * C;
            int cl = min(C, T - cs);  // chunk length (may be < C for last chunk)

            // ── STEP 1: Load K into shared memory (float32 for dot products) ──
            int k_total = cl * Dk;
            for (int idx = tid; idx < k_total; idx += 128) {
                int tl = idx / Dk;
                int dk = idx % Dk;
                sh_k[tl * Dk + dk] = k_[(cs + tl) * Hk * Dk + dk];
            }

            // ── STEP 2: Compute cumlog and beta (single thread) ──
            if (tid == 0) {
                float cum = 0.0f;
                float neg_eA = -exp(static_cast<float>(A_log[hv_idx]));
                for (int t = 0; t < cl; t++) {
                    float av = static_cast<float>(a_[(cs + t) * Hv + hv_idx]);
                    float xg = av + static_cast<float>(dt_bias[hv_idx]);
                    float sp = (xg > 20.0f) ? xg : log(1.0f + exp(xg));
                    cum += log(exp(neg_eA * sp) + 1e-38f);
                    sh_cumlog[t] = cum;
                    sh_beta[t] = 1.0f / (1.0f + exp(-static_cast<float>(b_[(cs + t) * Hv + hv_idx])));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── STEP 3: Compute L_strict (lower triangle) ──
            for (int idx = tid; idx < C * C; idx += 128)
                sh_L[idx] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each of 128 threads handles some (i,j) pairs
            for (int idx = tid; idx < cl * cl; idx += 128) {
                int i = idx / cl;
                int j = idx % cl;
                if (j >= i) continue;  // strict lower triangle only
                float dot = 0.0f;
                for (int d = 0; d < Dk; d++)
                    dot += static_cast<float>(sh_k[i * Dk + d]) * static_cast<float>(sh_k[j * Dk + d]);
                float decay = exp(sh_cumlog[i] - sh_cumlog[j]);
                sh_L[i * C + j] = sh_beta[i] * dot * decay;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ── STEP 4+5: RHS and delta via forward substitution ──
            // Solve (I + L_strict) @ delta = rhs directly. No barriers needed:
            // all threads in a SIMD group compute the same delta (it only varies
            // per dv, not per dk), and L_strict is in shared memory (read-only).
            float delta[C];
            for (int t = 0; t < cl; t++) {
                float dect = exp(sh_cumlog[t]);
                // state · k[t]: partial sum + simd reduction
                float sdk = 0.0f;
                for (int i = 0; i < NPT; i++)
                    sdk += state[i] * static_cast<float>(sh_k[t * Dk + dk_idx * NPT + i]);
                sdk = simd_sum(sdk);
                float vv = static_cast<float>(v_[(cs + t) * Hv * Dv + dv_idx]);
                float rhs_t = sh_beta[t] * vv - sh_beta[t] * dect * sdk;
                // Forward sub: delta[t] = rhs[t] - sum_{j<t} L[t,j] * delta[j]
                float correction = 0.0f;
                for (int j = 0; j < t; j++)
                    correction += sh_L[t * C + j] * delta[j];
                delta[t] = rhs_t - correction;
            }

            // ── STEP 6: Output ──
            for (int t = 0; t < cl; t++) {
                float dect = exp(sh_cumlog[t]);
                auto qt = q_ + (cs + t) * Hk * Dk;
                // state · q[t]
                float sdq = 0.0f;
                for (int i = 0; i < NPT; i++)
                    sdq += state[i] * static_cast<float>(qt[dk_idx * NPT + i]);
                sdq = simd_sum(sdq);
                float y_st = dect * sdq;
                // intra-chunk: sum_j A[t,j] * delta[j]
                float y_in = 0.0f;
                for (int j = 0; j <= t; j++) {
                    float qdk = 0.0f;
                    for (int i = 0; i < NPT; i++)
                        qdk += static_cast<float>(qt[dk_idx * NPT + i]) * static_cast<float>(sh_k[j * Dk + dk_idx * NPT + i]);
                    qdk = simd_sum(qdk);
                    float decay_tj = exp(sh_cumlog[t] - sh_cumlog[j]);
                    y_in += qdk * decay_tj * delta[j];
                }
                if (thread_index_in_simdgroup == 0)
                    y_[(cs + t) * Hv * Dv + dv_idx] = static_cast<InT>(y_st + y_in);
            }

            // ── STEP 7: State update ──
            float td = exp(sh_cumlog[cl - 1]);
            for (int i = 0; i < NPT; i++)
                state[i] *= td;
            for (int j = 0; j < cl; j++) {
                float dte = exp(sh_cumlog[cl - 1] - sh_cumlog[j]);
                float wd = dte * delta[j];
                for (int i = 0; i < NPT; i++)
                    state[i] += wd * static_cast<float>(sh_k[j * Dk + dk_idx * NPT + i]);
            }

            // Barrier before next chunk overwrites shared memory
            threadgroup_barrier(mem_flags::mem_threadgroup);

        }  // end chunk loop

        // Write final state
        auto o_st = state_out + (n * Dv + dv_idx) * Dk;
        for (int i = 0; i < NPT; i++)
            o_st[dk_idx * NPT + i] = static_cast<StT>(state[i]);
    """

    return mx.fast.metal_kernel(
        name="gated_delta_chunkwise",
        input_names=["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_chunkwise_kernel = _make_chunkwise_kernel()


def gated_delta_chunkwise_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    chunk_size = 32

    return _chunkwise_kernel(
        inputs=[q, k, v, a, b, A_log, dt_bias, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("CHUNK", chunk_size),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if a.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, a, b, A_log, dt_bias, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, a, b, A_log, dt_bias, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


def gated_delta_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    chunk_size: int = 64,
) -> Tuple[mx.array, mx.array]:
    """
    Chunkwise parallel gated delta rule for prefill.

    Instead of processing tokens sequentially, splits into sub-chunks and
    uses matrix operations within each chunk. Only supports scalar gating
    (g: [B, T, Hv]).

    Shapes:
      q, k: [B, T, Hk, Dk]
      v: [B, T, Hv, Dv]
      g: [B, T, Hv]
      beta: [B, T, Hv]
      state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2], v.shape[3]

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, axis=2)
        k = mx.repeat(k, repeat_factor, axis=2)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    q_f = q.astype(mx.float32)
    k_f = k.astype(mx.float32)
    v_f = v.astype(mx.float32)
    # g = exp(-positive), so g is in (0, 1]. Use tiny epsilon to avoid log(0)
    # without clamping g itself — a larger clamp (e.g. 1e-6) corrupts gating
    # for tokens with strong decay (g ≈ 1e-14).
    log_g = mx.log(g.astype(mx.float32) + 1e-38)
    beta_f = beta.astype(mx.float32)

    if mask is not None:
        mask_h = mask[..., None].astype(mx.float32)
        log_g = log_g * mask_h
        beta_f = beta_f * mask_h

    ys = []
    for c_start in range(0, T, chunk_size):
        C = min(chunk_size, T - c_start)

        # Transpose to heads-first: [B, Hv, C, D]
        q_t = q_f[:, c_start : c_start + C].transpose(0, 2, 1, 3)
        k_t = k_f[:, c_start : c_start + C].transpose(0, 2, 1, 3)
        v_t = v_f[:, c_start : c_start + C].transpose(0, 2, 1, 3)
        beta_t = beta_f[:, c_start : c_start + C].transpose(0, 2, 1)
        log_g_t = log_g[:, c_start : c_start + C].transpose(0, 2, 1)

        # Cumulative log decay
        cumlog = mx.cumsum(log_g_t, axis=-1)

        # Decay matrix (lower triangular): decay[i,j] = prod(g[j+1..i])
        decay_mat = mx.exp(cumlog[..., :, None] - cumlog[..., None, :])
        decay_mat = mx.tril(decay_mat)

        # Pairwise key dot products
        KK = k_t @ k_t.transpose(0, 1, 3, 2)

        # L_strict: correction for intra-chunk self-interference
        L = beta_t[..., :, None] * KK * decay_mat
        L_strict = mx.tril(L, -1)

        # RHS = beta*v - beta*decay_to*(state @ k)
        decay_to = mx.exp(cumlog)
        Sk = state @ k_t.transpose(0, 1, 3, 2)
        p = (Sk * (beta_t * decay_to)[..., None, :]).transpose(0, 1, 3, 2)
        bv = beta_t[..., None] * v_t
        rhs = bv - p

        # Solve (I + L_strict) @ delta = rhs via Neumann series with doubling.
        # L_strict is nilpotent (strictly lower triangular), so the series
        # (I + L)^{-1} = sum_{n>=0} (-L)^n terminates exactly at C-1 terms.
        n_iter = math.ceil(math.log2(max(C, 2)))
        inv = mx.eye(C, dtype=mx.float32)
        power = -L_strict
        for _ in range(n_iter):
            inv = inv + inv @ power
            power = power @ power
        delta_mat = inv @ rhs

        # Output = state_contribution + intra_chunk_attention
        Q_decayed = q_t * decay_to[..., None]
        y_state = (state @ Q_decayed.transpose(0, 1, 3, 2)).transpose(
            0, 1, 3, 2
        )

        QK = q_t @ k_t.transpose(0, 1, 3, 2)
        A = mx.tril(QK * decay_mat)
        y_intra = A @ delta_mat

        ys.append((y_state + y_intra).transpose(0, 2, 1, 3))

        # State update: S = total_decay * S + sum(decay_to_end * delta outer k)
        total_decay = mx.exp(cumlog[..., -1:])
        state = state * total_decay[..., None]
        decay_to_end = mx.exp(cumlog[..., -1:] - cumlog)
        delta_weighted = delta_mat * decay_to_end[..., None]
        state = state + delta_weighted.transpose(0, 1, 3, 2) @ k_t

    y = mx.concatenate(ys, axis=1)

    if mask is not None:
        y = y * mask[..., None, None].astype(y.dtype)

    return y.astype(q.dtype), state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    T = q.shape[1]
    on_gpu = use_kernel and mx.default_device() == mx.gpu and mx.metal.is_available()

    # Fused chunkwise kernel for scalar-gated prefill (e.g. Qwen3.5)
    if T > 32 and a.ndim == 3 and on_gpu and mask is None:
        return gated_delta_chunkwise_kernel(
            q, k, v, a, b, A_log, dt_bias, state
        )

    # GPU kernel (decode, short prefill, masked, or vectorized gating)
    if on_gpu:
        return gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state, mask)

    # CPU/non-kernel fallback
    beta = mx.sigmoid(b.astype(mx.float32))
    g = compute_g(A_log, a, dt_bias)
    y, state = gated_delta_ops(q, k, v, g, beta, state, mask)
    return y, state.astype(q.dtype)
