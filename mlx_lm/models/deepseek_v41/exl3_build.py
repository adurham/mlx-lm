# LOCAL ADDITION (not vendored). Builds the V4.1 model from an EXL3 checkpoint.
"""Build ``deepseek_v41.Model`` from an exllamav3-style EXL3 checkpoint.

Every projection that the checkpoint stores as a trellis group becomes an
``EXL3Linear``; the routed experts of each layer become one stacked
``EXL3SwitchGLU`` (optionally one rank's intermediate-width slice); plain
tensors (norms, gate, sinks, hc mixes, embed) load as weights. The engram
tables are read row-on-demand from the native release (never materialized).

Accounting is strict: every parameter left in the model tree must come from
the checkpoint, and every checkpoint group of a built layer must be consumed.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..exl3.loader import Exl3Checkpoint, load_dense_linear, load_experts
from .config import ModelArgs
from .model import Block, Model

_SKIP_PREFIXES = ("vision", "aligner.", "image_", "mtp.")


class Exl3Proj(nn.Module):
    """``EXL3Linear`` that keeps the caller's activation dtype."""

    def __init__(self, lin):
        super().__init__()
        self._lin = lin

    def __call__(self, x: mx.array) -> mx.array:
        return self._lin(x).astype(x.dtype)


class Exl3GroupedProj(nn.Module):
    """Grouped block-diagonal projection: x [..., g, d_in] -> [..., g, d_out]."""

    def __init__(self, lins):
        super().__init__()
        self._lins = list(lins)

    def __call__(self, x: mx.array) -> mx.array:
        outs = [lin(x[..., g, :]) for g, lin in enumerate(self._lins)]
        return mx.stack(outs, axis=-2).astype(x.dtype)


class Exl3Experts(nn.Module):
    """``EXL3SwitchGLU`` with the port's ``SwitchGLU`` call convention."""

    def __init__(self, switch):
        super().__init__()
        self._sg = switch

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        squeeze = x.ndim == 2
        if squeeze:
            x = x[None]
        if indices.ndim == 2:
            indices = indices[None]
        y = self._sg(x.astype(mx.float16), indices)
        if y.ndim == 2:
            y = y.reshape(x.shape[0], x.shape[1], indices.shape[-1], -1)
        if squeeze:
            y = y[0]
        return y


class LazyEngramTable(nn.Module):
    """Row-on-demand reader for one fp8 engram table of the native release."""

    def __init__(self, native: Exl3Checkpoint, layer_id: int):
        super().__init__()
        self._ck = native
        self._w = f"layers.{layer_id}.engram.embed.weight"
        self._s = f"layers.{layer_id}.engram.embed.scale"
        rows, dim = native.header(self._w)["shape"]
        self._dim = dim
        self._fd: dict[str, int] = {}

    def _rows(self, name: str, idx: np.ndarray) -> np.ndarray:
        sh = self._ck._shard(name)
        ent = sh.header[name]
        width = ent["shape"][1]
        base = sh.base + ent["data_offsets"][0]
        fd = sh._open()
        buf = bytearray(len(idx) * width)
        mv = memoryview(buf)
        for i, r in enumerate(idx):
            mv[i * width:(i + 1) * width] = os.pread(fd, width, base + int(r) * width)
        return np.frombuffer(buf, np.uint8).reshape(len(idx), width)

    def __call__(self, indices: mx.array) -> mx.array:
        idx = np.array(indices, dtype=np.int64)
        flat = idx.reshape(-1)
        uniq = np.unique(flat)
        w = mx.from_fp8(mx.array(self._rows(self._w, uniq)), mx.float32)
        s = self._rows(self._s, uniq).astype(np.int32) - 127
        sf = mx.power(mx.array(2.0), mx.array(s.astype(np.float32)))
        rows = (w.reshape(len(uniq), -1, 32) * sf[..., None]).reshape(len(uniq), self._dim)
        inv = mx.array(np.searchsorted(uniq, flat).astype(np.int32))
        return rows[inv].reshape(idx.shape + (self._dim,))


def _plain(ck: Exl3Checkpoint, name: str, dtype=None) -> mx.array:
    a = mx.array(ck.np(name))
    return a.astype(dtype) if dtype is not None and a.dtype != dtype else a


def _set(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    setattr(obj, parts[-1], module)


def _groups(ck: Exl3Checkpoint, prefix: str) -> set[str]:
    return {k[: -len(".trellis")] for k in ck.index
            if k.startswith(prefix) and k.endswith(".trellis")}


def build_block(ck: Exl3Checkpoint, args: ModelArgs, layer_id: int, *,
                native: Exl3Checkpoint | None = None, rank: int = 0,
                world: int = 1, group=None) -> tuple[Block, dict]:
    """One ``Block`` with EXL3 projections, experts and plain weights loaded.

    Returns ``(block, report)``; raises if accounting is not exact."""
    pre = f"layers.{layer_id}."
    blk = Block(layer_id, args)

    # routed experts (stacked, optionally sliced for TP)
    blk.ffn.experts = Exl3Experts(load_experts(ck, layer_id, rank=rank, world=world))
    if world > 1:
        blk.ffn.group = group

    dense = {g for g in _groups(ck, pre) if ".ffn.experts." not in g}
    wo_a = sorted((g for g in dense if ".attn.wo_a.slice." in g),
                  key=lambda g: int(g.rsplit(".", 1)[1]))
    if wo_a:
        blk.attn.wo_a = Exl3GroupedProj(load_dense_linear(ck, g) for g in wo_a)
    for g in sorted(set(dense) - set(wo_a)):
        _set(blk, g[len(pre):], Exl3Proj(load_dense_linear(ck, g)))

    if blk.engram is not None:
        if native is None:
            raise ValueError(f"layer {layer_id} has engram; pass the native release")
        blk.engram.embed = LazyEngramTable(native, layer_id)

    expected = {k for k, _ in tree_flatten(blk.parameters())}
    have = {}
    for k in ck.index:
        if not k.startswith(pre) or ".ffn.experts." in k:
            continue
        tail = k[len(pre):]
        if tail.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1"):
            continue
        have[tail] = k
    missing = expected - set(have)
    # bias_vl selects experts for image spans only; text runtime keeps zeros
    optional = {t for t in missing if t.endswith("gate.bias_vl")}
    missing -= optional
    unexpected = set(have) - expected
    if missing or unexpected:
        raise ValueError(f"layer {layer_id}: missing={sorted(missing)[:20]} "
                         f"unexpected={sorted(unexpected)[:20]}")
    fp32 = ("norm.weight", "attn_sink", "hc_", "gate.bias", "q_weight", "k_weight")
    items = []
    for tail, name in have.items():
        dt = mx.float32 if any(s in tail for s in fp32) else None
        items.append((tail, _plain(ck, name, dt)))
    blk.load_weights(items, strict=False)
    mx.eval([v for _, v in items])
    return blk, {"dense_groups": len(dense), "plain": len(items),
                 "optional_absent": sorted(optional)}


def build_model(model_dir: str, *, native_dir: str | None = None,
                layers: Iterable[int] | None = None, rank: int = 0,
                world: int = 1, group=None,
                embed_dtype=mx.bfloat16) -> tuple[Model, dict]:
    """Full (or layer-subset) model. ``layers`` builds only those blocks; the
    rest are replaced by ``None`` and must not be run."""
    ck = Exl3Checkpoint(model_dir)
    native = Exl3Checkpoint(native_dir) if native_dir else None
    args = ModelArgs.from_dict(ck.config)
    n_mtp, args.n_mtp_layers = args.n_mtp_layers, 0      # body first; MTP separate
    want = sorted(set(range(args.n_layers) if layers is None else layers))

    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.args = args
    model.hc_mult = args.hc_mult
    model.mtp = None
    model.engram_hasher = None
    model._break_sharing = False
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
    model.head = Exl3Proj(load_dense_linear(ck, "head"))
    model.load_weights([("embed.weight", _plain(ck, "embed.weight", embed_dtype)),
                        ("norm.weight", _plain(ck, "norm.weight", mx.float32))],
                       strict=False)
    top = {k for k in ck.index
           if not k.startswith(("layers.",) + _SKIP_PREFIXES)}
    top_expected = {"embed.weight", "norm.weight", "head.trellis", "head.suh",
                    "head.svh", "head.mul1"}
    stray = top - top_expected
    if stray:
        raise ValueError(f"unconsumed top-level tensors: {sorted(stray)}")
    args.n_mtp_layers = n_mtp
    return model, {"layers": reports, "n_mtp_layers_skipped": n_mtp}
