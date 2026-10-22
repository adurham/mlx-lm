# LOCAL ADDITION (not vendored). Builds V4.1 from a native-MLX (oMLX "oQ") checkpoint.
"""Build ``deepseek_v41.Model`` from a native-MLX mixed-precision checkpoint.

This is the second build path next to :mod:`exl3_build`. It reads a checkpoint
that stores **standard MLX quantized tensors** (``weight`` uint32 packed,
``scales`` and -- affine modes only -- ``biases``), with all quant metadata in
``config.json`` under ``omlx_deepseek_v41`` instead of the usual
``quantization`` key::

    "omlx_deepseek_v41": {
        "version": 1,
        "quantized_modules": {"<module path>": {bits, group_size, mode,
                                               quantize_input}, ...},
        "engram_tables": {...}, "engram_in_index": true, "preserve_mtp": true,
    }

Three consequences drive the design:

* **Modes are per-module.** The 1017 oQ2.5e pack mixes ``affine`` (g64,
  2/3/8-bit) for the body with ``mxfp4``/``mxfp8`` (g32, E8M0 scales) for the
  DSpark draft. Nothing here may assume affine: ``mode`` is threaded into
  every ``quantized_matmul``/``gather_qmm``, and biases appear only for
  affine.
* **The rank slice is packed, not float.** Tensor parallelism must slice the
  *quantized* columns: ``packed_cols = elems * bits // 32``, ``scale_cols =
  elems // group_size``. :func:`_qslice` does that and refuses a boundary that
  is not a multiple of ``lcm(32, group_size)`` -- re-using
  ``exl3_build._dense_slice`` (which slices a reconstructed *float* weight and
  re-quantizes it) would be a different, lossy operation.
* **gather_qmm cannot mix.** One stacked expert tensor carries one
  bits/group/mode; :meth:`OqCheckpoint.expert_stack` asserts the metadata
  against the packed geometry of that one tensor so a per-expert mix fails at
  load time instead of silently decoding garbage.

``quantize_input`` (an A8 / activation-quant flag oMLX implements in its own
Metal kernels) is ignored -- activations stay bf16.
:func:`check_a8_is_runtime_only` mechanically verifies the pack carries no
smoothing/input-scale tensor, i.e. that the flag really is runtime-only.

Engram tables are *not* read from this pack (its two tables are the 94 GiB
shards): rows come from the native FP8 release through the same
:class:`exl3_build.LazyEngramTable` the EXL3 path uses.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelArgs
from .model import Block, Model
from .moe import ClampedSwiGLU
from .exl3_build import (AffineProj, LazyEngramTable, ShardedHead, _DRAFT_SHARD,
                         _Grouped, _block_bounds, _bias_vl_sidecar, _set)
from ..switch_layers import _gather_sort, _scatter_unsort
from ...profiler import span

#: The pack names the text stack ``language_model.*``; the model tree and the
#: EXL3 path use ``layers.*``/``head``/``mtp.*``. Everything else (``vision``,
#: ``aligner``, ``image_*``) belongs to the vision tower this runtime drops.
TEXT_PREFIX = "language_model."

#: safetensors dtype -> (numpy reader dtype, mlx dtype). BF16 has no numpy
#: dtype, so it is read as raw uint16 and reinterpreted with ``.view``.
#: ``F8_E4M3``/``F8_E8M0`` only occur in the engram tables, which this loader
#: never materializes -- they are rejected loudly rather than half-supported.
_DTYPES = {
    "F32": (np.float32, mx.float32),
    "BF16": (np.uint16, mx.bfloat16),
    "U32": (np.uint32, mx.uint32),
    "I32": (np.int32, mx.int32),
    "U8": (np.uint8, mx.uint8),
}

#: Model-tree attribute -> checkpoint tail for the routed experts: SwitchGLU
#: calls DeepSeek's w1 ``gate_proj`` and w3 ``up_proj``.
_EXPERT_RENAME = ((".gate_proj.", ".w1."), (".up_proj.", ".w3."),
                  (".down_proj.", ".w2."))

# Which projections tensor parallelism slices, and on which axis. ``attn``
# entries follow the contiguous head split, ``shared`` entries the 128-wide
# block split -- the identical geometry ``exl3_build`` uses, so a sharded oQ
# block is numerically comparable with a sharded EXL3 block.
_TP_OUT_ATTN = frozenset({"attn.wq_b"})
_TP_IN_ATTN = frozenset({"attn.wo_b"})
_TP_OUT_SHARED = frozenset({"ffn.shared_experts.w1", "ffn.shared_experts.w3"})
_TP_IN_SHARED = frozenset({"ffn.shared_experts.w2"})

_SHARD_ATTN = os.environ.get("DSV41_TP_ATTN", "1") == "1"    # same gates as exl3
_SHARD_SHARED = os.environ.get("DSV41_TP_SHARED", "1") == "1"
_SHARD_HEAD = os.environ.get("DSV41_TP_HEAD", "1") == "1"

#: dtypes the model keeps in fp32 regardless of the checkpoint (as exl3_build).
_FP32_TAILS = ("norm.weight", "attn_sink", "hc_", "gate.bias", "q_weight",
               "k_weight")


def _force_dtype(tail: str):
    return mx.float32 if any(s in tail for s in _FP32_TAILS) else None


#: Group size implied by the mode when the pack omits ``group_size``: the
#: published oQ config writes it for affine modules only and leaves the MXFP
#: ones to the mode's fixed 32-element block.
_MODE_GROUP = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}


def _qparams(qi: dict) -> tuple[int, int, str]:
    """``(bits, group_size, mode)`` for a ``quantized_modules`` entry.

    ``group_size`` is absent for the MXFP entries of the published pack, so the
    mode's block size is implied rather than defaulted silently to affine's 64.
    """
    mode = qi.get("mode", "affine")
    group = qi.get("group_size")
    if group is None:
        if mode not in _MODE_GROUP:
            raise ValueError(
                f"quantized module in mode {mode!r} has no group_size and no "
                f"implied block size (known: {sorted(_MODE_GROUP)})")
        group = _MODE_GROUP[mode]
    return int(qi["bits"]), int(group), mode


# --------------------------------------------------------------------------
# checkpoint reader
# --------------------------------------------------------------------------
class OqCheckpoint:
    """A native-MLX (oQ) checkpoint: index, headers and packed-tensor reader.

    Opening the 393 GiB pack costs only its safetensors headers (a few hundred
    KB). Tensors are read through ``np.memmap`` and copied once into MLX, so
    the peak is 1x the tensor -- the full pack never fits in RAM.
    """

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        cfg_path = os.path.join(self.model_dir, "config.json")
        with open(cfg_path) as fh:
            self.config = json.load(fh)
        self.omlx = self.config.get("omlx_deepseek_v41")
        if not self.omlx:
            raise ValueError(
                f"{cfg_path}: no 'omlx_deepseek_v41' block -- not a native-MLX "
                f"oQ checkpoint (an EXL3 pack goes through exl3_build)")
        idx_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        with open(idx_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        self.index: dict[str, str] = {}      # canonical name -> shard file
        self._orig: dict[str, str] = {}      # canonical name -> pack key
        for key, shard in weight_map.items():
            if not key.startswith(TEXT_PREFIX):
                continue                     # vision tower / aligner
            name = key[len(TEXT_PREFIX):]
            self.index[name] = shard
            self._orig[name] = key
        self.quant: dict[str, dict] = {
            k[len(TEXT_PREFIX):]: dict(v)
            for k, v in (self.omlx.get("quantized_modules") or {}).items()
            if k.startswith(TEXT_PREFIX)}
        self.engram_tables = self.omlx.get("engram_tables") or {}
        self.engram_in_index = bool(self.omlx.get("engram_in_index"))
        self.preserve_mtp = bool(self.omlx.get("preserve_mtp"))
        self._headers: dict[str, dict] = {}
        self._maps: list[np.memmap] = []

    # -- headers -----------------------------------------------------------
    def header(self, shard: str) -> dict:
        hdr = self._headers.get(shard)
        if hdr is None:
            path = os.path.join(self.model_dir, shard)
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                raw = fh.read(n)
            if len(raw) != n:
                raise ValueError(f"{path}: truncated safetensors header")
            hdr = json.loads(raw)
            hdr.pop("__metadata__", None)
            hdr["__data_start__"] = 8 + n
            self._headers[shard] = hdr
        return hdr

    def has(self, name: str) -> bool:
        return name in self.index

    def shape(self, name: str) -> tuple:
        return tuple(self.header(self.index[name])[self._orig[name]]["shape"])

    def dtype_name(self, name: str) -> str:
        return self.header(self.index[name])[self._orig[name]]["dtype"]

    def qinfo(self, module: str) -> dict | None:
        """Quant metadata for a module path, or None when it is plain."""
        return self.quant.get(module)

    def is_quantized(self, module: str) -> bool:
        return module in self.quant

    # -- tensor access -----------------------------------------------------
    def _raw(self, name: str):
        shard = self.index[name]
        hdr = self.header(shard)
        ent = hdr[self._orig[name]]
        dt = ent["dtype"]
        if dt not in _DTYPES:
            raise NotImplementedError(
                f"{name}: dtype {dt} is not read by this loader; only the "
                f"engram tables use fp8 and they come from the native release")
        np_dt, mx_dt = _DTYPES[dt]
        lo, hi = ent["data_offsets"]
        mm = np.memmap(os.path.join(self.model_dir, shard), dtype=np.uint8,
                       mode="r")
        self._maps.append(mm)
        flat = np.frombuffer(mm, dtype=np_dt, count=(hi - lo) // np_dt().itemsize,
                             offset=hdr["__data_start__"] + lo)
        return flat.reshape(ent["shape"]), mx_dt

    def array(self, name: str) -> mx.array:
        """The named tensor as an ``mx.array`` in its stored dtype."""
        if name not in self.index:
            raise KeyError(f"{name!r} not in checkpoint index "
                           f"({len(self.index)} text tensors)")
        flat, mx_dt = self._raw(name)
        arr = mx.array(flat)
        return arr if arr.dtype == mx_dt else arr.view(mx_dt)

    def np_view(self, name: str) -> np.ndarray:
        """Raw numpy view (BF16 as uint16) -- header-only checks, no MX copy."""
        return self._raw(name)[0]

    def close(self) -> None:
        for mm in self._maps:
            try:
                mm._mmap.close()      # noqa: SLF001 - release the mapping
            except Exception:         # pragma: no cover - best effort
                pass
        self._maps.clear()
        self._headers.clear()

    def __enter__(self) -> "OqCheckpoint":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- quantized tensor access ------------------------------------------
    def _parts(self, module: str):
        qi = self.quant[module]
        mode = qi.get("mode", "affine")
        w = self.array(f"{module}.weight")
        s = self.array(f"{module}.scales")
        has_b = self.has(f"{module}.biases")
        if mode == "affine" and not has_b:
            raise ValueError(f"{module}: affine mode without biases")
        if mode != "affine" and has_b:
            raise ValueError(f"{module}: mode {mode!r} must not carry biases")
        b = self.array(f"{module}.biases") if has_b else None
        return w, s, b, qi

    def slice_quant(self, module: str, *, axis: str, lo: int, hi: int):
        """One rank's packed slice: ``((w, scales, biases), qinfo)``."""
        w, s, b, qi = self._parts(module)
        bits, group, mode = _qparams(qi)
        return _qslice(w, s, b, bits=bits, group=group, mode=mode, axis=axis,
                       lo=lo, hi=hi, what=module), qi

    def expert_stack(self, module: str, *, axis: str, lo: int, hi: int):
        """One rank's slice of a stacked ``[E, out, packed_in]`` expert tensor.

        ``gather_qmm`` dispatches one mode/bits/group for the whole stack, so
        the single metadata entry must describe all E experts: the packed
        geometry is checked against it and a per-expert mix fails here.
        """
        w, s, b, qi = self._parts(module)
        if w.ndim != 3:
            raise ValueError(
                f"{module}: expected a stacked [E, out, packed_in] weight, got "
                f"{tuple(w.shape)} -- a per-expert layout cannot be dispatched "
                f"by gather_qmm")
        bits, group, mode = _qparams(qi)
        out_elems = w.shape[-2]
        in_elems = w.shape[-1] * 32 // bits
        if s.shape[-2] != out_elems or s.shape[-1] * group != in_elems:
            raise ValueError(
                f"{module}: scales {tuple(s.shape)} do not describe "
                f"[{w.shape[0]}, {out_elems}, {in_elems}] at bits={bits} "
                f"group={group} -- the stack is not uniformly quantized")
        if b is not None and b.shape != s.shape:
            raise ValueError(f"{module}: biases {tuple(b.shape)} != scales "
                             f"{tuple(s.shape)}")
        return (_qslice(w, s, b, bits=bits, group=group, mode=mode, axis=axis,
                        lo=lo, hi=hi, what=module), (bits, group, mode))

    def slice_plain(self, name: str, *, axis: str, lo: int, hi: int) -> mx.array:
        """One rank's float slice of a plain ``[out, in]`` weight."""
        w = self.array(name)
        if axis == "out":
            return mx.contiguous(w[lo:hi])
        return mx.contiguous(w[:, lo:hi])

    def dequant(self, module: str) -> mx.array:
        """The module's dense ``[out, in]`` weight, dequantized (tests)."""
        w, s, b, qi = self._parts(module)
        bits, group, mode = _qparams(qi)
        out = mx.dequantize(w, s, b, group_size=group, bits=bits, mode=mode)
        mx.eval(out)
        return out


def _qslice(weight, scales, biases, *, bits: int, group: int, mode: str,
            axis: str, lo: int, hi: int, what: str = "?"):
    """Slice a packed MLX-quantized tensor along element axis ``axis``.

    ``weight`` is ``[..., out, in*bits//32]`` (uint32); ``scales``/``biases``
    are ``[..., out, in//group]``. The slice ``[lo, hi)`` is expressed in
    *elements* of the sliced axis and must land on ``lcm(32, group_size)`` so
    that no rank cuts a uint32 word or a quantization group in half:

    * packed columns: ``[lo*bits//32, hi*bits//32)``;
    * scale columns: ``[lo//group, hi//group)``.

    Leading (expert) dimensions are carried through untouched, so this serves
    stacked expert tensors as well as 2-D projections.
    """
    if axis not in ("out", "in"):
        raise ValueError(f"{what}: axis must be 'out' or 'in', got {axis!r}")
    if not 0 <= lo < hi:
        raise ValueError(f"{what}: bad slice [{lo}, {hi})")
    step = math.lcm(32, group)
    if lo % step or hi % step:
        raise ValueError(
            f"{what}: slice [{lo}, {hi}) is not a multiple of "
            f"lcm(32, group_size)={step} (bits={bits}, group={group}, "
            f"mode={mode}); a rank would cut a quantization group or a packed "
            f"word in half")
    in_elems = weight.shape[-1] * 32 // bits
    limit = weight.shape[-2] if axis == "out" else in_elems
    if hi > limit:
        raise ValueError(f"{what}: slice [{lo}, {hi}) runs past {limit} "
                         f"elements on axis {axis!r}")
    if scales.shape[-2] != weight.shape[-2] or scales.shape[-1] * group != in_elems:
        raise ValueError(
            f"{what}: scales {tuple(scales.shape)} do not describe "
            f"{tuple(weight.shape)} at bits={bits} group={group}")
    if axis == "out":
        w = mx.contiguous(weight[..., lo:hi, :])
        s = mx.contiguous(scales[..., lo:hi, :])
        b = None if biases is None else mx.contiguous(biases[..., lo:hi, :])
    else:
        w = mx.contiguous(weight[..., lo * bits // 32: hi * bits // 32])
        s = mx.contiguous(scales[..., lo // group: hi // group])
        b = (None if biases is None else
             mx.contiguous(biases[..., lo // group: hi // group]))
    return w, s, b


def _span(n: int, rank: int, world: int, key: str, label: str) -> tuple[int, int]:
    """Rank's element span of ``n`` features on 128-wide blocks (16-wide tiles)."""
    a, b = _block_bounds(n // 16, rank, world, label, key)
    return a * 16, b * 16


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------
class OqSwitchLinear(nn.Module):
    """One stacked expert projection held as packed MLX tensors."""

    def __init__(self, weight, scales, biases, *, bits: int, group_size: int,
                 mode: str):
        super().__init__()
        self.weight, self.scales, self.biases = weight, scales, biases
        self.bits, self.group_size, self.mode = bits, group_size, mode
        #: Feed activations in the scale dtype: MLX falls back to a ~10x slower
        #: path when the activation dtype does not match it (moe.py has the
        #: measurement for the affine case).
        self.act_dtype = (scales.dtype if scales.dtype in (mx.float16, mx.bfloat16)
                          else mx.bfloat16)

    @property
    def input_dims(self) -> int:
        return self.scales.shape[-1] * self.group_size

    @property
    def output_dims(self) -> int:
        return self.weight.shape[-2]

    def __call__(self, x, indices, sorted_indices: bool = False):
        return mx.gather_qmm(x.astype(self.act_dtype), self.weight, self.scales,
                             self.biases, rhs_indices=indices, transpose=True,
                             group_size=self.group_size, bits=self.bits,
                             mode=self.mode, sorted_indices=sorted_indices)


class OqExperts(nn.Module):
    """``SwitchGLU`` over packed MLX expert stacks, one mode per projection.

    Call convention, gather/sort behaviour and span names are copied from
    ``switch_layers.SwitchGLU`` so this is a drop-in for ``MoE.experts``; the
    only differences are ``gather_qmm`` instead of ``gather_mm`` and the
    per-projection quant metadata.
    """

    def __init__(self, gate_proj, up_proj, down_proj, activation):
        super().__init__()
        self.gate_proj, self.up_proj = gate_proj, up_proj
        self.down_proj = down_proj
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            with span("switch.gather_sort"):
                x, idx, inv_order = _gather_sort(x, indices)
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


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------
def _quant_proj(ck: OqCheckpoint, module: str, *, axis: str | None = None,
                lo: int = 0, hi: int = 0) -> AffineProj:
    """``AffineProj`` (quantized, optionally one rank's slice) for ``module``."""
    if axis is None:
        w, s, b, qi = ck._parts(module)
    else:
        (w, s, b), qi = ck.slice_quant(module, axis=axis, lo=lo, hi=hi)
    bits, group, mode = _qparams(qi)
    return AffineProj.from_quantized(w, s, b, bits=bits, group=group, mode=mode)


def _feature_count(ck: OqCheckpoint, module: str, axis: str) -> int:
    """Element count of ``module``'s output (axis="out") or input features."""
    if ck.is_quantized(module):
        bits = ck.qinfo(module)["bits"]
        w = ck.shape(f"{module}.weight")
        return w[-2] if axis == "out" else w[-1] * 32 // bits
    return ck.shape(f"{module}.weight")[0 if axis == "out" else 1]


def _ck_param_name(pre: str, path: str) -> str:
    """Model-tree parameter path -> checkpoint tensor name."""
    for a, b in _EXPERT_RENAME:
        path = path.replace(a, b)
    return pre + path


def _load_experts(ck: OqCheckpoint, pre: str, args: ModelArgs, *, rank: int,
                  world: int, limit: float):
    """Routed experts of one layer/stage: stacked, packed, rank-sliced.

    Returns ``(experts, consumed)`` -- the checkpoint tensors it took.
    """
    mods, consumed = {}, set()
    for tail, axis, n in (("w1", "out", args.moe_inter_dim),
                          ("w3", "out", args.moe_inter_dim),
                          ("w2", "in", args.moe_inter_dim)):
        name = f"{pre}{tail}"
        lo, hi = _span(n, rank, world, name, f"{tail}_{axis}")
        (w, s, b), (bits, group, mode) = ck.expert_stack(name, axis=axis,
                                                        lo=lo, hi=hi)
        mods[tail] = OqSwitchLinear(w, s, b, bits=bits, group_size=group,
                                    mode=mode)
        consumed.update({f"{name}.weight", f"{name}.scales"})
        if mode == "affine":
            consumed.add(f"{name}.biases")
    return OqExperts(mods["w1"], mods["w3"], mods["w2"],
                     ClampedSwiGLU(limit)), consumed


def _load_wo_a(ck: OqCheckpoint, blk: Block, pre: str, args: ModelArgs, *,
               rank: int, world: int) -> tuple[set, str]:
    """``attn.wo_a``: one ``[groups*o_lora, heads_per_group*head_dim]`` tensor.

    The port's grouped einsum path only needs ``wo_a.weight`` in that exact
    shape with the groups stacked contiguously (``attention.py`` reshapes it to
    ``(n_groups, o_lora_rank, -1)``), so a single ``nn.Linear`` stands in for
    the EXL3 path's per-group ``_Grouped``. A quantized ``wo_a`` -- which this
    pack does not carry -- is split back into per-group modules, because a
    plain ``quantized_matmul`` over the stacked weight would mix groups.
    """
    a = blk.attn
    module = f"{pre}attn.wo_a"
    rows_full = args.o_groups * args.o_lora_rank
    lo, hi = _span(rows_full, rank, world, module, "wo_a_groups")
    per_group = args.o_lora_rank
    if (hi - lo) % per_group:
        raise ValueError(f"{module}: slice [{lo}, {hi}) is not whole groups")
    consumed = set()
    if ck.is_quantized(module):
        (w, s, b), qi = ck.slice_quant(module, axis="out", lo=lo, hi=hi)
        bits, group, mode = _qparams(qi)
        parts = []
        for g in range((hi - lo) // per_group):
            sl = slice(g * per_group, (g + 1) * per_group)
            parts.append(AffineProj.from_quantized(
                mx.contiguous(w[..., sl, :]), mx.contiguous(s[..., sl, :]),
                None if b is None else mx.contiguous(b[..., sl, :]),
                bits=bits, group=group, mode=mode))
        a.wo_a = _Grouped(parts)
        consumed.update({f"{module}.weight", f"{module}.scales"})
        if b is not None:
            consumed.add(f"{module}.biases")
        return consumed, "quantized"
    lin = nn.Linear(a.n_heads * a.head_dim // a.n_groups, hi - lo, bias=False)
    lin.weight = ck.slice_plain(f"{module}.weight", axis="out", lo=lo,
                                hi=hi).astype(lin.weight.dtype)
    a.wo_a = lin
    return {f"{module}.weight"}, "plain"


def _load_dense(ck: OqCheckpoint, obj: nn.Module, pre: str, *, rank: int,
                world: int, tp: dict[str, str],
                skip: tuple[str, ...] = ()) -> tuple[list, set, set]:
    """Load/replace every projection and tensor of ``obj`` from prefix ``pre``.

    Walks the freshly built parameter tree: each 2-D ``.weight`` is either a
    quantized module (present in ``ck.quant``) or a plain tensor; ``tp`` maps
    the checkpoint tails this rank must slice to their axis.

    Returns ``(items, consumed, sliced)`` -- plain ``(tail, array)`` pairs for
    ``load_weights``, the checkpoint tensors consumed, and the tails already
    applied as slices (they must not be re-loaded).
    """
    items, consumed, sliced = [], set(), set()
    for path, _ in tree_flatten(obj.parameters()):
        name = _ck_param_name(pre, path)
        if not name.startswith(pre):
            raise ValueError(f"{pre}: parameter {path!r} escaped its prefix")
        tail = name[len(pre):]
        if any(tail.startswith(s) for s in skip):
            continue
        module = name[: -len(".weight")] if path.endswith(".weight") else None
        # ``tp`` is keyed by the layer-relative MODULE tail (``attn.wq_b``):
        # neither the ``layers.N.`` prefix nor the ``.weight`` suffix belongs in
        # the key, and getting that wrong silently skips every dense shard.
        mod_tail = tail[: -len(".weight")] if path.endswith(".weight") else None
        axis = tp.get(mod_tail) if mod_tail is not None else None
        if module is not None and ck.is_quantized(module):
            if axis is None:
                proj = _quant_proj(ck, module)
            else:
                n = _feature_count(ck, module, axis)
                lo, hi = _span(n, rank, world, module, f"{tail}_{axis}")
                proj = _quant_proj(ck, module, axis=axis, lo=lo, hi=hi)
            _set(obj, tail[: -len(".weight")], proj)
            consumed.update({f"{module}.weight", f"{module}.scales"})
            if ck.qinfo(module).get("mode", "affine") == "affine":
                consumed.add(f"{module}.biases")
            sliced.add(tail)
            continue
        if not ck.has(name):
            raise ValueError(f"{pre}: model parameter {path!r} has no tensor "
                             f"{name!r} in the checkpoint")
        want = _force_dtype(tail)
        if axis is None:
            arr = ck.array(name)
        else:
            if not tail.endswith(".weight") or len(ck.shape(name)) != 2:
                raise ValueError(
                    f"{name}: tensor-parallel slicing is only implemented for "
                    f"2-D weights, got {ck.shape(name)}")
            n = _feature_count(ck, name[: -len(".weight")], axis)
            lo, hi = _span(n, rank, world, name, f"{tail}_{axis}")
            arr = ck.slice_plain(name, axis=axis, lo=lo, hi=hi)
            sliced.add(tail)
        if want is not None and arr.dtype != want:
            arr = arr.astype(want)
        items.append((tail, arr))
        consumed.add(name)
    return items, consumed, sliced


def _check_accounting(ck: OqCheckpoint, obj: nn.Module, pre: str, consumed: set,
                      skip: tuple[str, ...] = ()) -> dict:
    """Every checkpoint tensor of this prefix is used, and vice versa."""
    have = {k for k in ck.index if k.startswith(pre)
            and not any(k[len(pre):].startswith(s) for s in skip)}
    missing = have - consumed
    extra = {c for c in consumed if not ck.has(c)}
    tree = {_ck_param_name(pre, p) for p, _ in tree_flatten(obj.parameters())}
    unloaded = {t for t in tree if not ck.has(t)
                and not any(t[len(pre):].startswith(s) for s in skip)}
    if missing or extra or unloaded:
        raise ValueError(
            f"{pre}: missing={sorted(missing)[:12]} "
            f"unexpected={sorted(extra)[:12]} "
            f"unloaded_params={sorted(unloaded)[:12]}")
    return {"tensors": len(have), "quantized": len(
        [k for k in ck.quant if k.startswith(pre)])}


# --------------------------------------------------------------------------
# engram tables carried inside the pack
# --------------------------------------------------------------------------
class PackedEngramTable(nn.Module):
    """Row-on-demand reader for an engram table that lives inside an oQ pack.

    The native release stores these tables as ``F8_E4M3`` with ``F8_E8M0``
    per-32 exponents, which is what ``exl3_build.LazyEngramTable`` reads. An oQ
    pack may instead carry them quantized (the abliterated 1017 pack uses
    4-bit affine, group 32, with BF16 scales *and* biases), in which case the
    rows have to be dequantized rather than reinterpreted. The pack declares
    the layout in ``omlx_deepseek_v41.engram_tables``; this class trusts that
    spec and reads rows straight off the memmap, so a 57 GiB table still costs
    only the rows a forward asks for.
    """

    def __init__(self, ck: "OqCheckpoint", layer_id: int, spec: dict):
        super().__init__()
        self._ck = ck
        self._mod = f"layers.{layer_id}.engram.embed"
        self._bits = int(spec.get("bits", 4))
        self._group = int(spec.get("group_size", 32))
        self._mode = spec.get("mode", "affine")
        rows, packed_cols = ck.shape(f"{self._mod}.weight")
        self._dim = packed_cols * 32 // self._bits
        self._pending = None

    def _take(self, name: str, uniq: np.ndarray) -> mx.array:
        view = np.ascontiguousarray(self._ck.np_view(name)[uniq])
        mx_dt = _DTYPES[self._ck.dtype_name(name)][1]
        arr = mx.array(view)
        return arr if arr.dtype == mx_dt else arr.view(mx_dt)

    def _read(self, uniq: np.ndarray) -> mx.array:
        w = self._take(f"{self._mod}.weight", uniq)
        s = self._take(f"{self._mod}.scales", uniq)
        b = (self._take(f"{self._mod}.biases", uniq)
             if self._ck.has(f"{self._mod}.biases") else None)
        rows = mx.dequantize(w, s, b, group_size=self._group, bits=self._bits,
                             mode=self._mode)
        mx.eval(rows)
        return rows

    def prefetch(self, indices) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        uniq = np.unique(idx.reshape(-1))
        self._pending = (idx.tobytes(), uniq)

    def __call__(self, indices) -> mx.array:
        idx = np.asarray(indices, dtype=np.int64)
        flat = idx.reshape(-1)
        pend, self._pending = self._pending, None
        if pend is not None and pend[0] == idx.tobytes():
            uniq = pend[1]
        else:
            uniq = np.unique(flat)
        rows = self._read(uniq)
        inv = mx.array(np.searchsorted(uniq, flat).astype(np.int32))
        return rows[inv].reshape(idx.shape + (self._dim,))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build_block(ck: OqCheckpoint, args: ModelArgs, layer_id: int, *,
                native=None, rank: int = 0, world: int = 1,
                group=None) -> tuple[Block, dict]:
    """One ``Block`` from the oQ pack. Returns ``(block, report)``.

    ``native`` is the FP8 release checkpoint the engram rows are read from
    (``exl3.loader.Exl3Checkpoint``), exactly as in ``exl3_build``. A pack that
    carries its own engram tables (``engram_in_index``) is read directly and
    needs no native release at all.
    """
    pre = f"layers.{layer_id}."
    blk = Block(layer_id, args)

    blk.ffn.experts, consumed = _load_experts(
        ck, pre + "ffn.experts.", args, rank=rank, world=world,
        limit=args.swiglu_limit)
    if world > 1:
        blk.ffn.group = group

    tp: dict[str, str] = {}
    attn_tp = world > 1 and group is not None and _SHARD_ATTN
    if attn_tp:
        a = blk.attn
        if a.n_heads % world or a.n_groups % world:
            raise ValueError(f"layer {layer_id}: {a.n_heads} heads / "
                             f"{a.n_groups} groups not divisible by {world}")
        a.n_heads //= world
        a.n_groups //= world
        a.group = group
        tp.update({k: "out" for k in _TP_OUT_ATTN})
        tp.update({k: "in" for k in _TP_IN_ATTN})
    shared_tp = world > 1 and _SHARD_SHARED
    if shared_tp:
        tp.update({k: "out" for k in _TP_OUT_SHARED})
        tp.update({k: "in" for k in _TP_IN_SHARED})
        blk.ffn.shared_sharded = True

    wo_consumed, wo_a_kind = _load_wo_a(ck, blk, pre, args, rank=rank,
                                        world=world)
    consumed |= wo_consumed

    # Engram tables: a pack that indexes its own (``engram_in_index``) is read
    # row-on-demand from the pack itself; otherwise rows come from the native
    # FP8 release, exactly as the EXL3 path does.
    skip = ("ffn.experts.", "attn.wo_a.")
    if blk.engram is not None:
        spec = (ck.engram_tables.get(TEXT_PREFIX + f"layers.{layer_id}.engram.embed")
                if getattr(ck, "engram_in_index", False) else None)
        if spec:
            blk.engram.embed = PackedEngramTable(ck, layer_id, spec)
        elif native is not None:
            blk.engram.embed = LazyEngramTable(native, layer_id)
        else:
            raise ValueError(
                f"layer {layer_id} has engram but the pack carries no table and "
                f"no native release was passed via native_dir")
        skip = skip + ("engram.embed.",)

    items, used, sliced = _load_dense(ck, blk, pre, rank=rank, world=world,
                                      tp=tp, skip=skip)
    consumed |= used
    if attn_tp:
        # attn_sink is per head: this rank keeps its own heads (as exl3 does).
        hpr = blk.attn.n_heads
        items = [(t, v[rank * hpr:(rank + 1) * hpr] if t == "attn.attn_sink"
                  else v) for t, v in items]
    vl_src = None
    if ck.has(f"{pre}ffn.gate.bias_vl"):
        vl_src = "checkpoint"
    else:
        side = _bias_vl_sidecar(ck)          # keyed by the release tensor name
        key = f"layers.{layer_id}.ffn.gate.bias_vl"
        if side is not None and key in side:
            items.append(("ffn.gate.bias_vl", mx.array(side[key], mx.float32)))
            vl_src = "sidecar"
        # absent everywhere: text runtime keeps the zero init (as exl3_build)
    blk.load_weights(items, strict=False)
    mx.eval([v for _, v in items])

    rep = _check_accounting(ck, blk, pre, consumed,
                            skip=("engram.embed.",) if blk.engram is not None
                            else ())
    rep.update({"bias_vl": vl_src, "wo_a": wo_a_kind,
                "tp": {"attn": attn_tp, "shared": shared_tp}})
    return blk, rep


def build_model(model_dir: str, *, native_dir: str | None = None,
                layers: Iterable[int] | None = None, rank: int = 0,
                world: int = 1, group=None,
                embed_dtype=mx.bfloat16) -> tuple[Model, dict]:
    """Full (or layer-subset) model from an oQ pack.

    ``layers`` builds only those blocks; the rest are omitted and must not be
    run (the same contract as ``exl3_build.build_model``).
    """
    from ..exl3.loader import Exl3Checkpoint

    ck = OqCheckpoint(model_dir)
    native = Exl3Checkpoint(native_dir) if native_dir else None
    args = ModelArgs.from_dict(ck.config)
    n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0
    want = sorted(set(range(args.n_layers) if layers is None else layers))

    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.args = args
    model.hc_mult = args.hc_mult
    model.mtp = None
    model.engram_hasher = None
    model._break_sharing = False
    model._host_engram = os.environ.get("DSV41_ENGRAM_PREFETCH", "1") == "1"
    model.embed = nn.Embedding(args.vocab_size, args.dim)
    from .layers import RMSNorm
    model.norm = RMSNorm(args.dim, args.norm_eps)
    model.layers = []
    reports = {}
    for i in range(args.n_layers):
        if i in want:
            blk, rep = build_block(ck, args, i, native=native, rank=rank,
                                   world=world, group=group)
            model.layers.append(blk)
            reports[i] = rep
    model.head = _build_head(ck, args, rank=rank, world=world, group=group)
    model.load_weights([("embed.weight", _plain(ck, "embed.weight", embed_dtype)),
                        ("norm.weight", _plain(ck, "norm.weight", mx.float32))],
                       strict=False)
    top = {k for k in ck.index if not k.startswith(("layers.", "mtp."))}
    stray = top - {"embed.weight", "norm.weight", "head.weight"}
    if stray:
        raise ValueError(f"unconsumed top-level tensors: {sorted(stray)}")
    args.n_mtp_layers = n_mtp
    srcs = {r.get("bias_vl") for r in reports.values()}
    model.vl_bias_loaded = bool(reports) and None not in srcs
    model.vl_bias_source = sorted(s for s in srcs if s) or None
    return model, {"layers": reports, "n_mtp_layers_skipped": n_mtp,
                   "vl_bias_loaded": model.vl_bias_loaded,
                   "checkpoint": checkpoint_report(ck),
                   "args": args, "ck": ck}


def _plain(ck: OqCheckpoint, name: str, dtype=None) -> mx.array:
    a = ck.array(name)
    return a.astype(dtype) if dtype is not None and a.dtype != dtype else a


def _build_head(ck: OqCheckpoint, args: ModelArgs, *, rank: int, world: int,
                group):
    """Plain bf16 head, vocab-sharded when the process group is present."""
    per = args.vocab_size
    sharded = world > 1 and _SHARD_HEAD and group is not None
    if sharded:
        if args.vocab_size % world:
            raise ValueError(f"vocab {args.vocab_size} not divisible by {world}")
        per = args.vocab_size // world
    lo = rank * per
    if ck.is_quantized("head"):
        lin = (_quant_proj(ck, "head", axis="out", lo=lo, hi=lo + per)
               if sharded else _quant_proj(ck, "head"))
    else:
        w = (ck.slice_plain("head.weight", axis="out", lo=lo, hi=lo + per)
             if sharded else ck.array("head.weight"))
        lin = nn.Linear(args.dim, per, bias=False)
        lin.weight = w.astype(lin.weight.dtype)
    if sharded:
        return ShardedHead(None, group, rank, world, args.vocab_size, proj=lin)
    return lin


#: ``mtp.*`` tails that belong to the DSpark head itself rather than a stage.
_MTP_TOP = {
    "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight",
    "main_proj.weight": "main_proj.weight",
    "markov_head.embed.weight": "markov_embed.weight",
    "markov_head.head.weight": "markov_head.weight",
    "confidence_head.proj.weight": "confidence_proj.weight",
}


def build_mtp(ck: OqCheckpoint, args: ModelArgs, *, rank: int = 0,
              world: int = 1, group=None):
    """DSpark draft head (``mtp.{0..2}``) from the oQ pack.

    The draft is **quantized in this pack** -- mxfp4 routed experts, mxfp8 for
    the attention/shared projections and ``main_proj`` -- so every projection
    goes through :meth:`AffineProj.from_quantized`; the markov and confidence
    heads are plain bf16. Routed draft experts take the same intermediate-width
    rank slice as the body; everything else is replicated so the draft runs
    identically on every rank.
    """
    from .mtp import DSparkHead

    head = DSparkHead(args)
    consumed: set = set()
    for s_i, stage in enumerate(head.stages):
        pre = f"mtp.{s_i}."
        stage.ffn.experts, exp_consumed = _load_experts(
            ck, pre + "ffn.experts.", args, rank=rank, world=world,
            limit=args.swiglu_limit)
        consumed |= exp_consumed
        if world > 1:
            stage.ffn.group = group
        items, used, _ = _load_dense(ck, stage, pre, rank=rank, world=world,
                                     tp={}, skip=("ffn.experts.",))
        consumed |= used
        stage.load_weights(items, strict=False)
        mx.eval([v for _, v in items])
    # Head-level parameters (stages own everything else).
    for key in ck.index:
        if not key.startswith("mtp.") or ".ffn.experts." in key:
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        tail = key.split(".", 2)[2]
        target = _MTP_TOP.get(tail)
        if target is None:
            continue                       # a stage parameter, handled above
        module = key[: -len(".weight")]
        if ck.is_quantized(module):
            _set(head, target[: -len(".weight")], _quant_proj(ck, module))
            consumed.update({f"{module}.weight", f"{module}.scales"})
            if ck.qinfo(module).get("mode", "affine") == "affine":
                consumed.add(f"{module}.biases")
        else:
            head.load_weights([(target, _plain(ck, key, _force_dtype(target)))],
                              strict=False)
            consumed.add(key)
    # Every stage tensor must have been consumed (attn/ffn) plus the head's.
    have = {k for k in ck.index if k.startswith("mtp.")}
    if have - consumed:
        raise ValueError(f"mtp: unconsumed {sorted(have - consumed)[:12]}")
    if world > 1 and _SHARD_HEAD and group is not None and _DRAFT_SHARD:
        v = args.vocab_size // world
        w = head.markov_head.weight[rank * v:(rank + 1) * v]
        head.markov_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        head.markov_head.weight = mx.contiguous(w)
        head.vocab_sharded = True
    return head


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def checkpoint_report(ck: OqCheckpoint) -> dict:
    """What the config's oQ block says, plus the mechanical A8 verdict."""
    return {
        "version": ck.omlx.get("version"),
        "quantized_modules": len(ck.quant),
        "engram_in_index": ck.engram_in_index,
        "engram_tables": {k: {"mode": v.get("mode"),
                              "group_size": v.get("group_size"),
                              "weight_file": v.get("weight_file")}
                          for k, v in ck.engram_tables.items()},
        "preserve_mtp": ck.preserve_mtp,
        "a8": check_a8_is_runtime_only(ck),
    }


def check_a8_is_runtime_only(ck: OqCheckpoint) -> dict:
    """Verify ``quantize_input`` is a pure runtime flag (no A8 tensors)."""
    return a8_report(ck.index, ck.quant)


def a8_report(index_keys, quant: dict[str, dict]) -> dict:
    """A8 verdict over a tensor-name set plus ``quantized_modules``.

    For every module marked ``quantize_input`` the stored key set must be
    exactly ``{weight, scales}`` (mxfp*) or ``{weight, scales, biases}``
    (affine), and the pack must carry no ``input_scale``/``smooth*`` tensor
    anywhere. A smoothing tensor would mean activations are quantized against a
    per-channel scale this loader does not apply -- a load-time stop.

    ``index_keys``/``quant`` are plain data so the check runs against either a
    local pack or the committed header fixture.
    """
    keys = list(index_keys)
    by_module: dict[str, set] = {}
    for k in keys:
        for name in quant:
            if k.startswith(name + "."):
                tail = k[len(name) + 1:]
                if "." not in tail:
                    by_module.setdefault(name, set()).add(tail)
                break
    bad_keys: dict[str, list[str]] = {}
    n_flagged = 0
    for name, qi in quant.items():
        if not qi.get("quantize_input"):
            continue
        n_flagged += 1
        want = {"weight", "scales"}
        if qi.get("mode", "affine") == "affine":
            want.add("biases")
        got = by_module.get(name, set())
        if got != want:
            bad_keys[name] = sorted(got)
    suspicious = [k for k in keys
                  if any(s in k.lower() for s in ("input_scale", "smooth"))]
    return {"quantize_input_modules": n_flagged,
            "key_set_mismatches": bad_keys,
            "suspicious_tensors": suspicious,
            "runtime_only": not bad_keys and not suspicious}
