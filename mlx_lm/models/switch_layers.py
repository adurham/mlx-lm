# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from ..profiler import span


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
