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
    log_g = mx.log(mx.maximum(g.astype(mx.float32), 1e-6))
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

    # Chunkwise parallel for scalar-gated prefill (e.g. Qwen3.5)
    if T > 1 and a.ndim == 3:
        beta = mx.sigmoid(b.astype(mx.float32))
        g = compute_g(A_log, a, dt_bias)
        y, state = gated_delta_chunkwise(q, k, v, g, beta, state, mask)
        return y, state

    # GPU kernel (decode or vectorized-gating prefill)
    if on_gpu:
        return gated_delta_kernel(q, k, v, a, b, A_log, dt_bias, state, mask)

    # CPU/non-kernel fallback
    beta = mx.sigmoid(b.astype(mx.float32))
    g = compute_g(A_log, a, dt_bias)
    y, state = gated_delta_ops(q, k, v, g, beta, state, mask)
    return y, state.astype(q.dtype)
