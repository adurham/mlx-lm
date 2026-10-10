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

import fnmatch
import json
import os
from typing import Iterable

import numpy as np
import mlx.core as mx

from . import collective as _coll
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..exl3.loader import Exl3Checkpoint, load_dense_linear, load_experts
from .config import ModelArgs
from .model import Block, Model

_SKIP_PREFIXES = ("vision", "aligner.", "image_", "mtp.")


# Compile the small-batch EXL3 linear path: fuses the elementwise work around
# the trellis kernel (same ops, fewer launches).
_COMPILE_LIN = os.environ.get("DSV41_COMPILE_LIN", "1") == "1"


class Exl3Proj(nn.Module):
    """``EXL3Linear`` that keeps the caller's activation dtype."""

    def __init__(self, lin):
        super().__init__()
        self._lin = lin.release_source()
        self._fn = mx.compile(self._lin.__call__) if _COMPILE_LIN else self._lin

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.size // x.shape[-1]
        fn = self._fn if rows <= 16 else self._lin
        return fn(x).astype(x.dtype)


class AffineProj(nn.Module):
    """EXL3 group re-encoded as MLX affine (``bits``/``group``), fp16 activations.

    Built from the EXL3 reconstruction, so it approximates the SAME weights;
    the added error is the affine rounding only."""

    def __init__(self, layer, bits: int, group: int):
        super().__init__()
        from ..exl3.reconstruct import reconstruct_public_mlx
        w = mx.contiguous(reconstruct_public_mlx(layer).T)       # [out, in] fp16
        self._set_weight(w, bits, group)

    @classmethod
    def from_weight(cls, w: mx.array, bits: int, group: int, *,
                    expect: tuple[int, int] | None = None) -> "AffineProj":
        """``AffineProj`` from an already-reconstructed fp16 ``[out, in]`` weight.

        ``w`` must already be ``[out, in]`` -- the layout ``__init__`` builds
        after its ``.T``. This is the affine half of a TP slice: the caller
        reconstructs the dense group, slices the fp16 weight on 128-wide block
        boundaries, and hands the slice here (see ``_dense_slice``). A weight
        still in ``[in, out]`` would quantize the wrong axis and pass silently
        whenever both dims are multiples of ``group``; pass ``expect=(out, in)``
        to pin the orientation so a transposed projection cannot pass.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self._set_weight(w, bits, group, expect=expect)
        return self

    def _set_weight(self, w: mx.array, bits: int, group: int, *,
                    expect: tuple[int, int] | None = None) -> None:
        if w.ndim != 2:
            raise ValueError(f"AffineProj: expected a 2-D [out, in] weight, got {w.shape}")
        if expect is not None and tuple(w.shape) != tuple(expect):
            raise ValueError(
                f"AffineProj: weight {tuple(w.shape)} != expected [out, in] "
                f"{tuple(expect)} -- a transposed projection cannot pass")
        w = mx.contiguous(w.astype(mx.float16))
        if w.shape[-1] % group:
            raise ValueError(
                f"AffineProj: in_features {w.shape[-1]} not divisible by group {group}")
        self._q = mx.quantize(w, group_size=group, bits=bits)
        mx.eval(self._q)
        # Release the fp16 reconstruct/transpose/slice intermediates. They stay
        # in MLX's buffer cache otherwise (measured ~1.5 GiB residue after one
        # layer's affine dense roster; exl3 leaves 0), inflating the rank's
        # footprint going into the load warmup (110-114.6 GB affine vs 105.5 GB
        # exl3 on the cluster, ROUND-Q1B). Affine-only: exl3 never builds this.
        mx.clear_cache()
        self._bits, self._group = bits, group
        self.out_features = w.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        wq, sc, bi = self._q
        y = mx.quantized_matmul(x.astype(mx.float16), wq, sc, bi, transpose=True,
                                group_size=self._group, bits=self._bits)
        return y.astype(x.dtype)


class Exl3GroupedProj(nn.Module):
    """Grouped block-diagonal projection: x [..., g, d_in] -> [..., g, d_out]."""

    def __init__(self, lins):
        super().__init__()
        self._lins = [lin.release_source() for lin in lins]

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


class _Grouped(nn.Module):
    """Grouped block-diagonal projection over any per-group modules."""

    def __init__(self, mods):
        super().__init__()
        self._mods = list(mods)

    def __call__(self, x: mx.array) -> mx.array:
        outs = [m(x[..., g, :]) for g, m in enumerate(self._mods)]
        return mx.stack(outs, axis=-2).astype(x.dtype)


_ENGRAM_POOL = None

# Read shape for the engram tables (module read at import). DEFAULT "coarse"
# since 2026-10-02 (measured -10.3% 16K prefill, max|dlogit|=0; set
# DSV41_ENGRAM_READ=legacy to restore the old shape). "legacy" keeps the original per-row fan-out: one pool task
# per unique row for the weights and one for the scales -- 2*U tiny
# submit/result round trips per lookup (80 816 tasks for a 2048-row chunk at
# U = 40 408). "coarse" groups the rows into contiguous index slices and reads
# each slice with ONE task that loops its rows: 2*min(_ENGRAM_SLICES, U) tasks,
# same preads, same bytes. Root cause is the per-row granularity, not the I/O.
_ENGRAM_READ = os.environ.get("DSV41_ENGRAM_READ", "coarse")
_ENGRAM_SLICES = int(os.environ.get("DSV41_ENGRAM_SLICES", "64"))


def _engram_pool():
    global _ENGRAM_POOL
    if _ENGRAM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _ENGRAM_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="engram")
    return _ENGRAM_POOL


class Exl3FusedGroup:
    """Several EXL3 projections evaluated by ONE trellis GEMM launch.

    Members share the input shape (and usually the input array). Their
    trellises are concatenated along out-tiles; each member keeps its own
    input rotation (``suh``) through the kernel's stacked-xh ``tile_sub`` map
    and its own output rotation (``svh``, blockwise so concatenation is exact).
    Used for rows <= 16 (decode / verify); larger batches use the members'
    own ``EXL3Linear`` (prefill paths)."""

    def __init__(self, lins):
        from ..exl3 import gemv_metal as G
        from ..exl3.layer_state import _had_fn
        self._G, self._had = G, _had_fn
        self.lins = list(lins)
        rts = [lin._rt for lin in self.lins]
        self.k, self.cb = rts[0].k, rts[0].cb
        self.in_f = self.lins[0].in_features
        self.outs = [lin.out_features for lin in self.lins]
        self.trellis = mx.concatenate([rt.trellis for rt in rts], axis=1)
        self.suh = mx.stack([rt.suh.astype(mx.float16) for rt in rts])     # [n, in]
        self.svh = mx.concatenate([rt.svh.astype(mx.float16) for rt in rts])
        self.tile_sub = mx.array(np.concatenate(
            [np.full(o // 16, i, np.uint32) for i, o in enumerate(self.outs)]))
        self.bounds = np.cumsum([0] + self.outs)
        mx.eval(self.trellis, self.suh, self.svh, self.tile_sub)
        self._x_key = None
        self._ys = None
        self._run = mx.compile(self._run_stacked) if _COMPILE_LIN else self._run_stacked

    @staticmethod
    def compatible(lins) -> bool:
        rts = [lin._rt for lin in lins]
        r0 = rts[0]
        return all(rt.k == r0.k and rt.cb == r0.cb and rt.suh is not None
                   and rt.svh is not None and rt.bias is None
                   and rt.trellis.shape[0] == r0.trellis.shape[0]
                   and rt.trellis.shape[2] == r0.trellis.shape[2]
                   and lin.out_features % 128 == 0 for rt, lin in zip(rts, lins))

    def run_stacked(self, xs: mx.array) -> mx.array:
        return self._run(xs)

    def _run_stacked(self, xs: mx.array) -> mx.array:
        """xs [rows, n, in] (member i reads xs[:, i]) -> [rows, sum(outs)] fp16."""
        rows, n, d = xs.shape
        xh = self._had("pre_scaled")(xs.reshape(rows * n, d),
                                     mx.tile(self.suh, (rows, 1)).reshape(rows * n, d))
        y = self._G.inner_gem_fused_mlx(xh.reshape(rows, n, d), self.trellis,
                                        self.k, self.cb, self.tile_sub)
        return self._had("post_scaled")(y.astype(mx.float16), self.svh)

    def member_out(self, i: int, x: mx.array, lin):
        rows = x.size // x.shape[-1]
        if rows > 16:
            return lin(x)
        if self._x_key is not x:
            x2 = x.reshape(rows, 1, x.shape[-1])
            xs = mx.broadcast_to(x2, (rows, len(self.lins), x.shape[-1]))
            self._ys = self.run_stacked(xs)
            self._x_key = x
        a, b = int(self.bounds[i]), int(self.bounds[i + 1])
        return self._ys[:, a:b].reshape(x.shape[:-1] + (b - a,))


class Exl3Member(nn.Module):
    def __init__(self, group: Exl3FusedGroup, i: int):
        super().__init__()
        self._g, self._i = group, i
        self._lin = group.lins[i]
        self.out_features = group.outs[i]

    def __call__(self, x: mx.array) -> mx.array:
        return self._g.member_out(self._i, x, self._lin).astype(x.dtype)


class Exl3GroupedStack(nn.Module):
    """Grouped block-diagonal wo_a: x [..., g, d_in] -> [..., g, d_out], one launch."""

    def __init__(self, group: Exl3FusedGroup):
        super().__init__()
        self._g = group

    def __call__(self, x: mx.array) -> mx.array:
        g = self._g
        rows = x.size // (x.shape[-1] * x.shape[-2])
        if rows > 16:
            outs = [lin(x[..., i, :]) for i, lin in enumerate(g.lins)]
            return mx.stack(outs, axis=-2).astype(x.dtype)
        y = g.run_stacked(x.reshape(rows, x.shape[-2], x.shape[-1]))
        return y.reshape(x.shape[:-2] + (len(g.lins), g.outs[0])).astype(x.dtype)


_FUSE_GROUPS = os.environ.get("DSV41_FUSE_GROUPS", "1") == "1"
_GROUPS = (
    ("attn.wq_a", "attn.wkv", "attn.compressor.wkv", "attn.compressor.wgate"),
    ("attn.wq_b", "attn.indexer.wq_b"),
    ("ffn.shared_experts.w1", "ffn.shared_experts.w3"),
)


def _fuse_block(blk: nn.Module, pre: str, ck: Exl3Checkpoint) -> int:
    """Replace same-input EXL3 projections with fused-group members."""
    n = 0

    def get(path):
        obj = blk
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                return None
        return obj

    for names in _GROUPS:
        present = [(nm, get(nm)) for nm in names]
        present = [(nm, m) for nm, m in present if isinstance(m, Exl3Proj)]
        by_k: dict = {}
        for nm, m in present:
            by_k.setdefault((m._lin._rt.k, m._lin.in_features), []).append((nm, m))
        for members in by_k.values():
            if len(members) < 2 or not Exl3FusedGroup.compatible([m._lin for _, m in members]):
                continue
            grp = Exl3FusedGroup([m._lin for _, m in members])
            for i, (nm, _) in enumerate(members):
                _set(blk, nm, Exl3Member(grp, i))
            n += len(members)
    wo = getattr(blk.attn, "wo_a", None)
    if isinstance(wo, _Grouped) and all(isinstance(m, Exl3Proj) for m in wo._mods):
        lins = [m._lin for m in wo._mods]
        if Exl3FusedGroup.compatible(lins):
            blk.attn.wo_a = Exl3GroupedStack(Exl3FusedGroup(lins))
            n += len(lins)
    return n


def _block_bounds(n: int, rank: int, world: int, label: str, key: str) -> tuple[int, int]:
    """Rank's ``[a, b)`` slice of ``n`` 16-wide tiles, on 128-wide blocks.

    The single geometry shared by the trellis slice (:func:`_slice_dense`) and
    the post-reconstruction weight slice (:func:`_slice_weight`): each rank
    owns an integer number of 128-wide blocks, so the blockwise Hadamard
    rotations AND the affine groups (size 64) never straddle a slice boundary.
    """
    if n % (8 * world):
        raise ValueError(f"{key}: {label} {n} not divisible into 128-blocks x {world}")
    return rank * n // world, (rank + 1) * n // world


def _slice_dense(layer, *, axis: str, rank: int, world: int):
    """One rank's slice of a dense EXL3 group, on 128-wide Hadamard blocks
    (exact: the rotations are blockwise). axis="out" slices the output
    features, axis="in" slices the input features."""
    from ..exl3.ref.layer import EXL3Layer
    t = layer.trellis
    if axis == "out":
        a, b = _block_bounds(t.shape[1], rank, world, "out_tiles", layer.key)
        return EXL3Layer(key=f"{layer.key}#out{rank}/{world}", in_features=layer.in_features,
                         out_features=(b - a) * 16, k=layer.k, trellis=np.ascontiguousarray(t[:, a:b]),
                         suh=layer.suh, svh=layer.svh[a * 16:b * 16], mul1=layer.mul1)
    a, b = _block_bounds(t.shape[0], rank, world, "in_tiles", layer.key)
    return EXL3Layer(key=f"{layer.key}#in{rank}/{world}", in_features=(b - a) * 16,
                     out_features=layer.out_features, k=layer.k, trellis=np.ascontiguousarray(t[a:b]),
                     suh=layer.suh[a * 16:b * 16], svh=layer.svh, mul1=layer.mul1)


def _slice_weight(layer, *, axis: str, rank: int, world: int) -> mx.array:
    """One rank's fp16 ``[out, in]`` slice of a dense group, sliced AFTER the
    reconstruction instead of on the trellis (the affine path). The Hadamard
    rotations are blockwise on 128-wide blocks, so a block-boundary slice
    commutes with the reconstruction: this weight equals
    ``reconstruct_public_mlx(_slice_dense(layer, ...))``. axis="out" slices the
    output features, axis="in" the input features."""
    from ..exl3.reconstruct import reconstruct_public_mlx
    w = reconstruct_public_mlx(layer)                        # [in, out] fp16
    # Orientation guard: reconstruct_public_mlx returns [in, out]; assert it
    # against the layer's own metadata so a transposed weight cannot be sliced
    # along the wrong axis and pass silently.
    if tuple(w.shape) != (layer.in_features, layer.out_features):
        raise ValueError(
            f"{layer.key}: reconstruct gave {tuple(w.shape)}, expected [in, out] "
            f"({layer.in_features}, {layer.out_features})")
    w = mx.contiguous(w.T)                                   # [out, in]
    if axis == "out":
        a, b = _block_bounds(layer.out_features // 16, rank, world, "out_tiles", layer.key)
        return mx.contiguous(w[a * 16:b * 16])
    a, b = _block_bounds(layer.in_features // 16, rank, world, "in_tiles", layer.key)
    return mx.contiguous(w[:, a * 16:b * 16])


def _dense_slice(ck: Exl3Checkpoint, name: str, *, axis: str, rank: int, world: int):
    """One rank's TP slice of dense group ``name``, in the resolved policy
    format: the trellis slice re-wrapped as an ``EXL3Linear`` (exl3), or the
    reconstructed fp16 weight sliced on the same 128-wide block boundaries and
    affine-quantized (affineN)."""
    from ..exl3.loader import load_dense_layer
    lay = load_dense_layer(ck, name)
    kind, bits, group = _resolve_dense_mode(name, _DENSE_POLICY, DENSE_MODE)
    if kind == "affine":
        assert bits is not None and group is not None
        w = _slice_weight(lay, axis=axis, rank=rank, world=world)
        expect = ((w.shape[0], lay.in_features) if axis == "out"
                  else (lay.out_features, w.shape[1]))
        return AffineProj.from_weight(w, bits, group, expect=expect)
    from ..exl3 import EXL3Linear
    return Exl3Proj(EXL3Linear(_slice_dense(lay, axis=axis, rank=rank, world=world)))


class ShardedHead(nn.Module):
    """Vocab-sharded head: each rank computes its slice of the logits and
    writes it into a zero row of full width; one all_sum rebuilds the full row
    (adding zeros is exact, so values equal the full head's). all_sum is used
    instead of all_gather because the JACCL mesh all_gather rejects this size."""

    def __init__(self, lin, group, rank: int, world: int, vocab: int):
        super().__init__()
        self._p = Exl3Proj(lin)
        self._group = group
        self._rank, self._world = rank, world
        self._lo = rank * (vocab // world)
        self._vocab = vocab

    def local(self, h: mx.array) -> mx.array:
        """This rank's vocab slice of the logits, fp32."""
        return self._p(h).astype(mx.float32)

    def combine_argmax(self, y: mx.array) -> mx.array:
        """Exact global argmax from per-rank slices: each rank contributes its
        (max, index) pair; ties go to the lowest vocab index, same as
        ``mx.argmax`` over the full row."""
        shp = y.shape[:-1]
        flat = y.reshape(-1, y.shape[-1])
        pair = mx.stack([mx.max(flat, axis=-1),
                         mx.argmax(flat, axis=-1).astype(mx.float32) + self._lo], axis=-1)
        buf = mx.pad(pair[None], [(self._rank, self._world - self._rank - 1), (0, 0), (0, 0)])
        allp = _coll.all_sum(buf, group=self._group)          # [world, rows, 2]
        best = mx.argmax(allp[..., 0], axis=0)                          # first max = lowest idx
        idx = mx.take_along_axis(allp[..., 1], best[None], axis=0)[0]
        return idx.astype(mx.int32).reshape(shp)

    def argmax(self, h: mx.array) -> mx.array:
        return self.combine_argmax(self.local(h))

    def topk_logprobs(self, h: mx.array, k: int):
        """Exact greedy ids + log-probs (selected and top-k), one small all_sum."""
        from . import logprobs as _lp
        y = self.local(h)
        shp = y.shape[:-1]
        buf = _lp.local_buffer(y.reshape(-1, y.shape[-1]), k, self._lo)
        buf = mx.pad(buf[None], [(self._rank, self._world - self._rank - 1), (0, 0), (0, 0)])
        ids, sel, top_ids, top_lp = _lp.combine(_coll.all_sum(buf, group=self._group), k)
        kk = top_ids.shape[-1]
        return ids.reshape(shp), {"selected": sel.reshape(shp),
                                  "top_ids": top_ids.reshape(*shp, kk),
                                  "top_logprobs": top_lp.reshape(*shp, kk)}

    def __call__(self, h: mx.array) -> mx.array:
        y = self._p(h).astype(mx.float32)
        w = y.shape[-1]
        pad = [(0, 0)] * (y.ndim - 1) + [(self._lo, self._vocab - self._lo - w)]
        return _coll.all_sum(mx.pad(y, pad), group=self._group)


_SHARD_SHARED = os.environ.get("DSV41_TP_SHARED", "1") == "1"
_SHARD_ATTN = os.environ.get("DSV41_TP_ATTN", "1") == "1"
_DRAFT_SHARD = os.environ.get("DSV41_DRAFT_SHARD", "1") == "1"
_SHARD_HEAD = os.environ.get("DSV41_TP_HEAD", "1") == "1"


class LazyEngramTable(nn.Module):
    """Row-on-demand reader for one fp8 engram table of the native release.

    Rows are read with parallel preads (the GIL is released during I/O), and
    ``prefetch`` starts the reads as soon as the token ids are known, so the
    SSD latency overlaps the layers that run before this one."""

    def __init__(self, native: Exl3Checkpoint, layer_id: int):
        super().__init__()
        self._ck = native
        self._w = f"layers.{layer_id}.engram.embed.weight"
        self._s = f"layers.{layer_id}.engram.embed.scale"
        rows, dim = native.header(self._w)["shape"]
        self._dim = dim
        self._pending = None            # (key bytes, uniq, future)

    def _loc(self, name: str):
        sh = self._ck._shard(name)
        ent = sh.header[name]
        return sh._open(), sh.base + ent["data_offsets"][0], ent["shape"][1]

    def _read(self, uniq: np.ndarray):
        fw, bw, ww = self._loc(self._w)
        fs, bs, ws = self._loc(self._s)
        pool = _engram_pool()
        n = len(uniq)
        if _ENGRAM_READ != "coarse":
            # legacy shape: one pool task PER ROW per tensor (2*n tasks), each
            # result joined one by one -- the per-task Python overhead this
            # branch exists to avoid.
            fut_w = [pool.submit(os.pread, fw, ww, bw + int(r) * ww) for r in uniq]
            fut_s = [pool.submit(os.pread, fs, ws, bs + int(r) * ws) for r in uniq]
            w = np.frombuffer(b"".join(f.result() for f in fut_w), np.uint8).reshape(n, ww)
            sc = np.frombuffer(b"".join(f.result() for f in fut_s), np.uint8).reshape(n, ws)
            return w, sc
        # coarse shape: the rows are split into min(_ENGRAM_SLICES, n) contiguous
        # index slices; each slice is read by ONE task looping its rows into a
        # preallocated bytearray. Same preads, same offsets, same bytes.
        wb = bytearray(n * ww)
        sb = bytearray(n * ws)
        if n:
            k = max(1, min(_ENGRAM_SLICES, n))
            step = -(-n // k)

            def slurp(lo: int, hi: int, fd: int, base: int, width: int, mv) -> None:
                for i in range(lo, hi):
                    mv[i * width:(i + 1) * width] = os.pread(
                        fd, width, base + int(uniq[i]) * width)

            futs = [pool.submit(slurp, i * step, min((i + 1) * step, n),
                                fw, bw, ww, memoryview(wb))
                    for i in range(k)]
            futs += [pool.submit(slurp, i * step, min((i + 1) * step, n),
                                 fs, bs, ws, memoryview(sb))
                     for i in range(k)]
            for f in futs:
                f.result()
        # bytes() keeps the read-only flag the legacy join produced
        w = np.frombuffer(bytes(wb), np.uint8).reshape(n, ww)
        sc = np.frombuffer(bytes(sb), np.uint8).reshape(n, ws)
        return w, sc

    def prefetch(self, indices) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        uniq = np.unique(idx.reshape(-1))
        self._pending = (idx.tobytes(), uniq, _engram_pool().submit(self._read, uniq))

    def __call__(self, indices) -> mx.array:
        idx = np.asarray(indices, dtype=np.int64)
        flat = idx.reshape(-1)
        pend, self._pending = self._pending, None
        if pend is not None and pend[0] == idx.tobytes():
            uniq, (w_np, s_np) = pend[1], pend[2].result()
        else:
            uniq = np.unique(flat)
            w_np, s_np = self._read(uniq)
        w = mx.from_fp8(mx.array(w_np), mx.float32)
        sf = mx.power(mx.array(2.0), mx.array((s_np.astype(np.int32) - 127).astype(np.float32)))
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


DENSE_MODE = os.environ.get("DSV41_DENSE", "exl3")          # exl3 | affine8 | affine6 | affine5
# DSV41_DENSE_TP=0 forces the affine modes back to the REPLICATED (unsharded)
# dense path -- the cluster A/B needs sharded vs replicated vs exl3. Ignored by
# exl3 mode, which is always sharded.
_DENSE_TP = os.environ.get("DSV41_DENSE_TP", "1") == "1"

# DSV41_DENSE_POLICY: per-tensor dense-quant override on top of DSV41_DENSE.
# Either an inline spec ("selector=mode,selector=mode") or an absolute path to
# a JSON file (a dict {selector: mode} or a list of [selector, mode] pairs).
# Selectors are fnmatch globs matched against the FULL tensor name (e.g.
# "layers.20.attn.wq_b"); the LAST matching selector wins, and a tensor matched
# by none (or an unset/empty policy) falls back to the global DSV41_DENSE
# behavior above -- so an unset policy is byte-identical to the pre-policy
# engine. This changes ONLY the per-tensor quant FORMAT, never the sharding.
_POLICY_MODES = {                                  # policy mode -> (kind, bits, group)
    "exl3": ("exl3", None, None),
    "q6g64": ("affine", 6, 64),
    "q6g32": ("affine", 6, 32),
    "q8g64": ("affine", 8, 64),
}


def _parse_dense_policy(spec) -> list[tuple[str, str]]:
    """Parse a ``DSV41_DENSE_POLICY`` spec into ordered ``(selector, mode)`` pairs.

    ``spec`` is either an inline comma-separated ``selector=mode`` string or the
    path to a JSON file (a dict -- insertion order preserved -- or a list of
    ``[selector, mode]`` pairs). Modes are validated here, so a typo fails at
    build time instead of silently falling back mid-load.
    """
    if not spec:
        return []
    # a filesystem path to a JSON spec is the only non-inline form
    if isinstance(spec, str) and os.path.isfile(spec):
        with open(spec) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            pairs = list(data.items())
        elif isinstance(data, list):
            pairs = [(p[0], p[1]) for p in data]
        else:
            raise ValueError(
                f"DSV41_DENSE_POLICY {spec!r}: JSON must be a dict or a list "
                f"of [selector, mode] pairs, got {type(data).__name__}")
    else:
        pairs = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            sel, sep, mode = item.partition("=")
            if not sep:
                raise ValueError(
                    f"DSV41_DENSE_POLICY: bad entry {item!r}; expected selector=mode")
            pairs.append((sel.strip(), mode.strip()))
    out = []
    for sel, mode in pairs:
        if mode not in _POLICY_MODES:
            raise ValueError(
                f"DSV41_DENSE_POLICY: unknown mode {mode!r} for selector {sel!r}; "
                f"allowed: {', '.join(sorted(_POLICY_MODES))}")
        out.append((sel, mode))
    return out


def _resolve_dense_mode(name: str, policy: list[tuple[str, str]],
                        base_mode: str) -> tuple[str, int | None, int | None]:
    """Resolve tensor ``name`` to ``(kind, bits, group)`` (``kind`` exl3|affine).

    Selectors match the full name with ``fnmatch.fnmatchcase`` (``*`` crosses
    ``.``); the LAST match wins. No match -- or an empty policy -- returns
    ``base_mode``'s global behavior, so an unset policy reproduces the current
    engine exactly (byte-identical modules).
    """
    hit = None
    for sel, mode in policy:
        if fnmatch.fnmatchcase(name, sel):
            hit = mode                                     # last match wins
    if hit is not None:
        return _POLICY_MODES[hit]
    if base_mode.startswith("affine"):
        return "affine", int(base_mode[6:]), 64
    return "exl3", None, None


_DENSE_POLICY = _parse_dense_policy(os.environ.get("DSV41_DENSE_POLICY", ""))


def _dense(ck: Exl3Checkpoint, name: str):
    kind, bits, group = _resolve_dense_mode(name, _DENSE_POLICY, DENSE_MODE)
    if kind == "affine":
        assert bits is not None and group is not None
        from ..exl3.loader import load_dense_layer
        return AffineProj(load_dense_layer(ck, name), bits, group)
    return Exl3Proj(load_dense_linear(ck, name))


_BIAS_VL_CACHE: dict = {}


def bias_vl_sidecar_path(model_dir: str) -> str:
    """Where the image-routing biases live when the checkpoint dropped them.

    ``gate.bias_vl`` (one fp32 [n_experts] vector per MoE layer) selects the
    experts for image-span rows. The release ships it; an EXL3 re-quant can
    drop it (the served dealignai 2.9bpw index has 0 of the release's 43).
    ``DSV41_BIAS_VL`` overrides; default
    ``~/.exo/dsv41/gate_bias_vl_<checkpoint dir name>.npz`` (keys = the release
    tensor names, e.g. ``layers.3.ffn.gate.bias_vl``).
    """
    override = os.environ.get("DSV41_BIAS_VL")
    if override:
        return os.path.expanduser(override)
    name = os.path.basename(os.path.normpath(os.path.expanduser(model_dir)))
    return os.path.expanduser(f"~/.exo/dsv41/gate_bias_vl_{name}.npz")


def _bias_vl_sidecar(ck: Exl3Checkpoint):
    path = bias_vl_sidecar_path(ck.model_dir)
    if path not in _BIAS_VL_CACHE:
        _BIAS_VL_CACHE[path] = dict(np.load(path)) if os.path.exists(path) else None
    return _BIAS_VL_CACHE[path]


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
    # Affine modes TP-shard EXACTLY like exl3: same head-split of wo_a, same
    # 128-wide block slices of wq_b/wo_b and the shared experts -- only the
    # per-rank projection differs (AffineProj instead of EXL3Proj). The
    # DENSE_MODE gate that previously forced affine to load every dense group
    # FULL (each rank computed the full slice -> roughly break-even, not the
    # priced 2.4x) is gone; DSV41_DENSE_TP=0 restores the replicated behavior.
    tp_on = _DENSE_TP or not DENSE_MODE.startswith("affine")
    attn_tp = world > 1 and _SHARD_ATTN and tp_on and group is not None
    if attn_tp:
        # heads split contiguously: rank r owns heads [r*H/w, (r+1)*H/w) and the
        # wo_a groups over exactly those heads; wo_b takes the matching input slice
        a = blk.attn
        if a.n_heads % world or a.n_groups % world:
            raise ValueError("heads/groups not divisible by world")
        hpr, gpr = a.n_heads // world, a.n_groups // world
        a.n_heads, a.n_groups = hpr, gpr
        a.group = group
        mine = wo_a[rank * gpr:(rank + 1) * gpr]
        blk.attn.wo_a = _Grouped([_dense(ck, g) for g in mine])
    elif wo_a:
        blk.attn.wo_a = _Grouped([_dense(ck, g) for g in wo_a])
    shared_tp = world > 1 and _SHARD_SHARED and tp_on
    for g in sorted(set(dense) - set(wo_a)):
        tail = g[len(pre):]
        if attn_tp and tail in ("attn.wq_b", "attn.wo_b"):
            _set(blk, tail, _dense_slice(
                ck, g, axis="out" if tail == "attn.wq_b" else "in",
                rank=rank, world=world))
        elif shared_tp and tail.startswith("ffn.shared_experts."):
            _set(blk, tail, _dense_slice(
                ck, g, axis="in" if tail.endswith("w2") else "out",
                rank=rank, world=world))
        else:
            _set(blk, tail, _dense(ck, g))
    if shared_tp:
        blk.ffn.shared_sharded = True

    fused = _fuse_block(blk, pre, ck) if (_FUSE_GROUPS and DENSE_MODE == "exl3") else 0

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
    if attn_tp:
        hpr = blk.attn.n_heads
        items = [(t, v[rank * hpr:(rank + 1) * hpr] if t == "attn.attn_sink" else v)
                 for t, v in items]
    vl_src = None
    if "ffn.gate.bias_vl" in optional:
        side = _bias_vl_sidecar(ck)
        key = f"layers.{layer_id}.ffn.gate.bias_vl"
        if side is not None and key in side:
            items.append(("ffn.gate.bias_vl", mx.array(side[key], dtype=mx.float32)))
            optional.discard("ffn.gate.bias_vl")
            vl_src = "sidecar"
    elif "ffn.gate.bias_vl" in have:
        vl_src = "checkpoint"
    blk.load_weights(items, strict=False)
    mx.eval([v for _, v in items])
    return blk, {"dense_groups": len(dense), "plain": len(items), "fused": fused,
                 "optional_absent": sorted(optional), "bias_vl": vl_src}


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
    if world > 1 and _SHARD_HEAD and group is not None:
        from ..exl3 import EXL3Linear
        from ..exl3.loader import load_dense_layer
        hl = _slice_dense(load_dense_layer(ck, "head"), axis="out", rank=rank, world=world)
        model.head = ShardedHead(EXL3Linear(hl), group, rank, world, args.vocab_size)
    else:
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
    srcs = {r.get("bias_vl") for r in reports.values()}
    # True only when EVERY built MoE layer has its image-routing bias: an image
    # span routed with the text bias is a different model (it loops).
    model.vl_bias_loaded = bool(reports) and None not in srcs
    model.vl_bias_source = sorted(s for s in srcs if s) or None
    return model, {"layers": reports, "n_mtp_layers_skipped": n_mtp,
                   "vl_bias_loaded": model.vl_bias_loaded}


_MTP_TOP = {
    "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight",
    "markov_head.embed.weight": "markov_embed.weight",
    "markov_head.head.weight": "markov_head.weight",
    "confidence_head.proj.weight": "confidence_proj.weight",
}


def build_mtp(ck: Exl3Checkpoint, args: ModelArgs, *, rank: int = 0, world: int = 1,
              group=None):
    """DSpark draft head (``mtp.*``) with EXL3 projections.

    Routed draft experts take the same intermediate-width rank slice as the
    body (one all_sum per stage); everything else is replicated, so the draft
    runs identically on every rank. Strict accounting like ``build_block``."""
    from .mtp import DSparkHead
    head = DSparkHead(args)
    for s_i, stage in enumerate(head.stages):
        pre = f"mtp.{s_i}."
        stage.ffn.experts = Exl3Experts(load_experts(
            ck, 0, prefix=pre + "ffn.experts.", rank=rank, world=world))
        if world > 1:
            stage.ffn.group = group
        dense = {g for g in _groups(ck, pre) if ".ffn.experts." not in g}
        wo_a = sorted((g for g in dense if ".attn.wo_a.slice." in g),
                      key=lambda g: int(g.rsplit(".", 1)[1]))
        if wo_a:
            stage.attn.wo_a = _Grouped([_dense(ck, g) for g in wo_a])
        for g in sorted(set(dense) - set(wo_a)):
            tail = g[len(pre):]
            if tail == "main_proj":
                head.main_proj = _dense(ck, g)
            else:
                _set(stage, tail, _dense(ck, g))
        if _FUSE_GROUPS and DENSE_MODE == "exl3":
            _fuse_block(stage, pre, ck)

    expected = {k for k, _ in tree_flatten(head.parameters())}
    have = {}
    for k in ck.index:
        if not k.startswith("mtp.") or ".ffn.experts." in k:
            continue
        if k.rsplit(".", 1)[-1] in ("trellis", "suh", "svh", "mul1"):
            continue
        s_i, tail = k[4:].split(".", 1)
        have[_MTP_TOP.get(tail, f"stages.{s_i}.{tail}")] = k
    missing = {t for t in expected - set(have) if not t.endswith("gate.bias_vl")}
    unexpected = set(have) - expected
    if missing or unexpected:
        raise ValueError(f"mtp: missing={sorted(missing)[:20]} unexpected={sorted(unexpected)[:20]}")
    fp32 = ("norm.weight", "attn_sink", "hc_", "gate.bias")
    items = [(t, _plain(ck, n, mx.float32 if any(f in t for f in fp32) else None))
             for t, n in have.items()]
    head.load_weights(items, strict=False)
    if world > 1 and _SHARD_HEAD and group is not None and _DRAFT_SHARD:
        v = args.vocab_size // world
        w = head.markov_head.weight[rank * v:(rank + 1) * v]
        head.markov_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
        head.markov_head.weight = mx.contiguous(w)
        head.vocab_sharded = True
    mx.eval([v for _, v in items])
    return head
