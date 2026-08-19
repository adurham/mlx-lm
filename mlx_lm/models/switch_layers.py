# Copyright © 2023-2024 Apple Inc.

import math
import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from ..profiler import span


# ---------------------------------------------------------------------------
# Hybrid MoE dispatch (PROTOTYPE, env-gated, default OFF)
#
# Motivation (routing granularity, not a workaround): MLX's GatherQMM
# eval_gpu picks its kernel from an *aggregate* B/E ratio (total routed
# (token,expert) pairs / total experts). For DeepSeek-V4-Flash decode-sized
# chunks that aggregate is ~48, far above the threshold (~6) at which the
# small-run `gather_qmv_rhs` kernel is enabled, so the fast small-run kernel
# never fires -- even though real (skewed) routing leaves many individual
# experts with only a handful of rows.
#
# This prototype splits the already-expert-sorted rows into two groups by
# per-expert run length and issues two gather_qmm calls: the "short" group
# has a low aggregate B/E ratio (so MLX may select gather_qmv_rhs) and the
# "long" group keeps the existing steel path. Outputs are recombined into
# the original row order, so the result is numerically equivalent to the
# single-call path up to kernel accumulation order.
#
# Enable with EXO_MOE_HYBRID_DISPATCH=1 (default "0" == completely inert).
# Threshold via EXO_MOE_HYBRID_THRESHOLD (default 8).
#
# !!! PERFORMANCE CAVEAT (correctness-only prototype) !!!
# `_hybrid_partition` performs ONE host-side read (`.item()`) of the number
# of short rows, because MLX needs a concrete shape to slice the two
# sub-arrays. That forces a GPU sync mid-pipeline and would very likely
# erase any kernel-level win. This MUST be replaced (e.g. by a fused
# primitive, a padded fixed-capacity split, or a C++-level dispatch that
# knows the run lengths without a round-trip) before this path could ever
# be considered a real performance improvement.
# ---------------------------------------------------------------------------


def _hybrid_enabled() -> bool:
    return os.environ.get("EXO_MOE_HYBRID_DISPATCH", "0") == "1"


def _hybrid_threshold() -> int:
    return int(os.environ.get("EXO_MOE_HYBRID_THRESHOLD", "8"))


def _hybrid_partition(idx, num_experts, threshold):
    """Split expert-sorted rows into (short-run, long-run) groups.

    ``idx`` is the flat, already-expert-sorted rhs index array of shape
    ``(P,)``. Returns ``(perm, n_short)`` where ``perm`` is a permutation of
    ``range(P)`` that lists all rows belonging to short-run experts first
    (still ordered by expert id), then all rows belonging to long-run
    experts (also ordered by expert id); ``n_short`` is a Python int.

    Both halves therefore satisfy ``sorted_indices=True`` for gather_qmm.
    """
    counts = mx.zeros((num_experts,), dtype=mx.int32)
    counts = counts.at[idx].add(mx.ones(idx.shape, dtype=mx.int32))
    is_long = (counts[idx] > threshold).astype(mx.int32)
    # Sort key keeps the two groups contiguous and each group expert-sorted.
    key = is_long * (num_experts + 1) + idx.astype(mx.int32)
    perm = mx.argsort(key)
    # NOTE: host-side sync -- see PERFORMANCE CAVEAT above.
    n_short = int((1 - is_long).sum().item())
    return perm, n_short


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            with span("switch.gather_sort"):
                x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        if do_sort and _hybrid_enabled():
            x = self._hybrid_body(x, idx)
        else:
            with span("switch.up_proj"):
                x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            with span("switch.gate_proj"):
                x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
            with span("switch.activation"):
                x_act = self.activation(x_up, x_gate)
            with span("switch.down_proj"):
                x = self.down_proj(x_act, idx, sorted_indices=do_sort)

        if do_sort:
            with span("switch.scatter_unsort"):
                x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)

    def _hybrid_body(self, x, idx):
        """Two-call hybrid dispatch over expert-sorted rows (PROTOTYPE).

        ``x`` is ``(P, 1, D)`` expert-sorted, ``idx`` is ``(P,)``. Returns
        the down_proj output in the SAME row order as the inputs, so the
        caller's ``_scatter_unsort`` still applies unchanged.
        """
        num_experts = self.gate_proj.num_experts
        threshold = _hybrid_threshold()
        with span("switch.hybrid_partition"):
            perm, n_short = _hybrid_partition(idx, num_experts, threshold)
        total = idx.shape[0]
        if n_short == 0 or n_short == total:
            # Degenerate split: nothing to gain, take the single-call path.
            return self._run_group(x, idx)

        x_p = x[perm]
        idx_p = idx[perm]
        outs = []
        with span("switch.hybrid_short"):
            outs.append(self._run_group(x_p[:n_short], idx_p[:n_short]))
        with span("switch.hybrid_long"):
            outs.append(self._run_group(x_p[n_short:], idx_p[n_short:]))
        out = mx.concatenate(outs, axis=0)
        # Undo the partition permutation -> back to plain expert-sorted order.
        inv_perm = mx.argsort(perm)
        return out[inv_perm]

    def _run_group(self, x, idx):
        x_up = self.up_proj(x, idx, sorted_indices=True)
        x_gate = self.gate_proj(x, idx, sorted_indices=True)
        return self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=True)


class BatchedSwitchGLU(SwitchGLU):
    """SwitchGLU variant that fuses gate+up into a single ``gather_qmm``.

    Vanilla :class:`SwitchGLU` issues two ``gather_qmm`` dispatches (one
    for ``gate_proj``, one for ``up_proj``) before the SwiGLU activation.
    When the projections are quantised with the same group size, mode and
    bits, the two weight buffers can be concatenated along the output
    dimension and dispatched together — halving the gather/dispatch cost
    in the routed-expert path.

    Call :meth:`fuse_weights` once after the underlying
    ``QuantizedSwitchLinear`` projections have been initialised (i.e.
    after ``nn.quantize``). After that, ``__call__`` uses the fused fast
    path. Until ``fuse_weights`` runs, ``__call__`` falls back to the
    vanilla two-dispatch path so the class is safe to instantiate before
    weights are loaded.

    The fused-weight attributes (``_fused_w_gu``, ``_fused_s_gu``,
    ``_fused_b_gu``, ``_fused_n_inter``, ``_fused_k_hidden``,
    ``_fused_group_size``) are written on ``self`` so that downstream MoE
    kernels which read them (e.g. a routed-experts dispatch that wants to
    reuse the concatenated gate+up buffers) can find them in a single
    well-known place.
    """

    def fuse_weights(self) -> None:
        """Concatenate quantised gate+up weights into the fused-path buffers.

        Idempotent: re-running rebuilds the fused buffers from the current
        ``gate_proj`` / ``up_proj`` weights. Requires both projections to
        be quantised (i.e. expose ``.weight`` / ``.scales`` / ``.biases``)
        and to share the same ``group_size`` / ``bits`` / ``mode`` — a
        plain :class:`SwitchLinear` (un-quantised) does not.
        """
        gate_proj = self.gate_proj
        up_proj = self.up_proj
        for proj_name, proj in (("gate_proj", gate_proj), ("up_proj", up_proj)):
            for attr in ("scales", "biases", "group_size", "bits"):
                if not hasattr(proj, attr):
                    raise TypeError(
                        f"BatchedSwitchGLU.fuse_weights(): {proj_name} is "
                        f"missing '{attr}'. Both projections must be quantised "
                        f"(QuantizedSwitchLinear) before calling fuse_weights()."
                    )
        if gate_proj.group_size != up_proj.group_size:  # type: ignore[attr-defined]
            raise ValueError(
                "BatchedSwitchGLU.fuse_weights(): gate_proj and up_proj must "
                "share group_size."
            )
        if gate_proj.bits != up_proj.bits:  # type: ignore[attr-defined]
            raise ValueError(
                "BatchedSwitchGLU.fuse_weights(): gate_proj and up_proj must "
                "share bits."
            )

        self._fused_w_gu = mx.concatenate(
            [gate_proj.weight, up_proj.weight], axis=1
        )
        self._fused_s_gu = mx.concatenate(  # type: ignore[attr-defined]
            [gate_proj.scales, up_proj.scales], axis=1
        )
        self._fused_b_gu = mx.concatenate(  # type: ignore[attr-defined]
            [gate_proj.biases, up_proj.biases], axis=1
        )
        self._fused_n_inter = gate_proj.output_dims
        self._fused_k_hidden = gate_proj.input_dims
        self._fused_group_size = gate_proj.group_size  # type: ignore[attr-defined]
        mx.eval(self._fused_w_gu, self._fused_s_gu, self._fused_b_gu)

    def __call__(self, x, indices) -> mx.array:
        if not hasattr(self, "_fused_w_gu"):
            return super().__call__(x, indices)

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        n_inter = self._fused_n_inter

        gu = mx.gather_qmm(
            x,
            self._fused_w_gu,
            self._fused_s_gu,
            self._fused_b_gu,
            rhs_indices=idx,
            transpose=True,
            group_size=self._fused_group_size,
            bits=self.gate_proj.bits,  # type: ignore[attr-defined]
            mode=self.gate_proj.mode,  # type: ignore[attr-defined]
            sorted_indices=do_sort,
        )

        x_gate = gu[..., :n_inter]
        x_up = gu[..., n_inter:]
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
