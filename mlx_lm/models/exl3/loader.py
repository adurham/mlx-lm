# LOCAL ADDITION (not vendored from PonyExl3) -- see README.md "loader.py".
"""Checkpoint loading for DeepSeek-V4.1 EXL3 checkpoints.

Builds the buffers the vendored kernels consume directly from an
exllamav3-style EXL3 safetensors checkpoint:

  * :func:`load_experts`      -> ``EXL3SwitchGLU`` for one MoE layer, with an
    optional ``world``-way intermediate-width slice (the TP convention proven
    in the exo repo, phase 12: gate/up sliced on OUT-tiles of the intermediate
    axis, down sliced on the matching K(ranged over hidden)-tiles, gu_suh and
    dn_svh replicated, partial outputs all_summed by the caller).
  * :func:`load_dense_layer`  -> ``EXL3Layer`` for a dense group (attention,
    shared expert, head, MTP projections).
  * :func:`load_dense_linear` -> ``EXL3Linear`` wrapping that layer.

Format notes (checked against dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw):
  * ``.trellis`` int16, shape ``(in_tiles, out_tiles, packed)``;
    ``k = packed * 16 // 256``.
  * ``.suh`` / ``.svh`` float16 full-size sign vectors (unpacked).
  * ``.mul1`` scalar int32 flag per tensor; this checkpoint is all-mul1
    (``quantization_config.codebook == "mul1"``).

The stacked MoE path requires float16 signs; packed int16 signs are rejected
with a clear error (the dense path can handle them via the runtime's
``unpack_signs_or_pass``).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any

import numpy as np

from .exl3_linear import EXL3Linear
from .exl3_moe import EXL3SwitchGLU
from .ref.codebook import codebook_mode_from_flags
from .ref.layer import EXL3Layer

__all__ = [
    "Exl3Checkpoint",
    "load_experts",
    "load_dense_layer",
    "load_dense_linear",
    "packed_to_k",
]

_NP_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "U32": np.uint32,
}
_SHARD_CACHE = 12  # open shard readers kept (LRU-ish, by insertion order)


class _Shard:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header: dict[str, Any] = json.loads(f.read(n))
        self.base = 8 + n
        self.fd: int | None = None

    def _open(self) -> int:
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        return self.fd

    def read(self, name: str) -> np.ndarray:
        ent = self.header[name]
        o0, o1 = ent["data_offsets"]
        buf = os.pread(self._open(), o1 - o0, self.base + o0)
        dt = ent["dtype"]
        shape = ent["shape"]
        if dt == "BF16":
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        return np.frombuffer(buf, _NP_DTYPES[dt]).reshape(shape)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def packed_to_k(packed: int) -> int:
    """EXL3 packed tile size -> bit width (256*k/16 == packed)."""
    k = packed * 16 // 256
    if k * 256 // 16 != packed:
        raise ValueError(f"packed tile size {packed} is not a multiple of 16")
    return k


class Exl3Checkpoint:
    """Read-only access to an EXL3 checkpoint directory (index + shard preads)."""

    def __init__(self, model_dir: str):
        self.model_dir = os.path.expanduser(model_dir)
        index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            self.index: dict[str, str] = json.load(f)["weight_map"]
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.config: dict[str, Any] = json.load(f)
        self._shards: dict[str, _Shard] = {}

    # -- introspection -----------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.index

    def header(self, name: str) -> dict[str, Any]:
        return self._shard(name).header[name]

    def text_config(self) -> dict[str, Any]:
        return self.config.get("text_config", self.config)

    def n_experts(self, layer_id: int, prefix: str | None = None) -> int:
        pre = prefix or f"layers.{layer_id}.ffn.experts."
        ids = set()
        for k in self.index:
            if k.startswith(pre):
                head = k[len(pre):].split(".", 1)[0]
                if head.isdigit():
                    ids.add(int(head))
        if not ids:
            raise KeyError(f"no expert tensors found for layer {layer_id}")
        return max(ids) + 1

    # -- tensor reads ------------------------------------------------------
    def _shard(self, name: str) -> _Shard:
        shard_file = self.index[name]
        sh = self._shards.get(shard_file)
        if sh is None:
            if len(self._shards) >= _SHARD_CACHE:
                old = next(iter(self._shards))
                self._shards.pop(old).close()
            sh = self._shards[shard_file] = _Shard(
                os.path.join(self.model_dir, shard_file)
            )
        return sh

    def np(self, name: str) -> np.ndarray:
        """Full tensor as numpy (BF16 comes back as float32)."""
        return self._shard(name).read(name)

    def close(self) -> None:
        for sh in self._shards.values():
            sh.close()
        self._shards.clear()

    def __enter__(self) -> "Exl3Checkpoint":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _require_fp16_signs(name: str, arr: np.ndarray) -> None:
    if arr.dtype != np.float16:
        raise NotImplementedError(
            f"{name}: dtype {arr.dtype} -- the stacked MoE path requires "
            "float16 sign vectors; unpack packed int16 signs first"
        )


def _slice_intermediate(
    gu: np.ndarray,
    gu_svh: np.ndarray,
    dn: np.ndarray,
    dn_suh: np.ndarray,
    *,
    rank: int,
    world: int,
    gu_tiles: int,
    hid_tiles: int,
    n_experts: int,
    in_tiles: int,
    packed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous intermediate-width slice (phase-12 proven geometry)."""
    if not (0 <= rank < world):
        raise ValueError(f"rank {rank} out of range for world {world}")
    if gu_tiles % world or hid_tiles % world:
        raise ValueError(
            f"world={world} must divide gu_tiles={gu_tiles} and hid_tiles={hid_tiles}"
        )
    hpt = gu_tiles // world
    kpt = hid_tiles // world
    t0, t1 = rank * hpt, (rank + 1) * hpt
    k0, k1 = rank * kpt, (rank + 1) * kpt

    gu_v = gu.reshape(in_tiles, 2, n_experts, gu_tiles, packed)
    gu_s = gu_v[:, :, :, t0:t1, :].reshape(in_tiles, 2 * n_experts * hpt, packed).copy()

    svh_v = gu_svh.reshape(n_experts, 2, gu_tiles * 16)
    svh_s = svh_v[:, :, t0 * 16 : t1 * 16].reshape(n_experts, 2 * hpt * 16).copy()

    dn_s = dn[k0:k1, :, :].copy()
    dsuh_s = dn_suh[:, k0 * 16 : k1 * 16].copy()
    return gu_s, svh_s, dn_s, dsuh_s


def load_experts(
    ckpt: Exl3Checkpoint,
    layer_id: int,
    *,
    n_experts: int | None = None,
    rank: int = 0,
    world: int = 1,
    activation: str = "silu_clamp",
    mul1: bool = True,
    prefix: str | None = None,
) -> EXL3SwitchGLU:
    """Build the stacked ``EXL3SwitchGLU`` for one MoE layer of the checkpoint.

    ``n_experts`` limits the load to the first N experts (testing); ``None``
    loads all of them. ``world > 1`` returns rank ``rank``'s intermediate-width
    slice; the caller all_sums the ranks' outputs.
    """
    import mlx.core as mx

    total = ckpt.n_experts(layer_id, prefix)
    if n_experts is not None:
        if not (1 <= n_experts <= total):
            raise ValueError(f"n_experts={n_experts} outside 1..{total}")
        total_e = n_experts
    else:
        total_e = total
    pre = prefix or f"layers.{layer_id}.ffn.experts."

    h_w1 = ckpt.header(pre + "0.w1.trellis")
    h_w2 = ckpt.header(pre + "0.w2.trellis")
    in_tiles, gu_tiles, packed = h_w1["shape"]
    hid_tiles, dn_tiles, packed2 = h_w2["shape"]
    if packed2 != packed:
        raise ValueError(
            f"layer {layer_id}: w1 packed={packed} != w2 packed={packed2}"
        )
    k = packed_to_k(packed)
    H = gu_tiles * 16
    D = in_tiles * 16

    gu = np.empty((in_tiles, 2 * total_e * gu_tiles, packed), dtype=np.int16)
    gu_suh = np.empty((total_e, 2, D), dtype=np.float16)
    gu_svh = np.empty((total_e, 2 * H), dtype=np.float16)
    dn = np.empty((hid_tiles, total_e * dn_tiles, packed), dtype=np.int16)
    dn_suh = np.empty((total_e, H), dtype=np.float16)
    dn_svh = np.empty((total_e, D), dtype=np.float16)

    for e in range(total_e):
        p = f"{pre}{e}."
        w1_t = ckpt.np(p + "w1.trellis")
        w3_t = ckpt.np(p + "w3.trellis")
        w2_t = ckpt.np(p + "w2.trellis")
        if w1_t.shape[2] != packed or w3_t.shape[2] != packed:
            raise ValueError(f"expert {e}: mixed packed tile sizes within layer")
        gu[:, e * gu_tiles : (e + 1) * gu_tiles] = w1_t
        gu[:, (total_e + e) * gu_tiles : (total_e + e + 1) * gu_tiles] = w3_t
        dn[:, e * dn_tiles : (e + 1) * dn_tiles] = w2_t

        w1_suh = ckpt.np(p + "w1.suh")
        w3_suh = ckpt.np(p + "w3.suh")
        _require_fp16_signs(p + "w1.suh", w1_suh)
        _require_fp16_signs(p + "w3.suh", w3_suh)
        gu_suh[e, 0] = w1_suh
        gu_suh[e, 1] = w3_suh

        w1_svh = ckpt.np(p + "w1.svh")
        w3_svh = ckpt.np(p + "w3.svh")
        _require_fp16_signs(p + "w1.svh", w1_svh)
        _require_fp16_signs(p + "w3.svh", w3_svh)
        gu_svh[e, :H] = w1_svh
        gu_svh[e, H:] = w3_svh

        dn_suh[e] = ckpt.np(p + "w2.suh")
        dn_svh[e] = ckpt.np(p + "w2.svh")

    if world > 1:
        gu, gu_svh, dn, dn_suh = _slice_intermediate(
            gu,
            gu_svh,
            dn,
            dn_suh,
            rank=rank,
            world=world,
            gu_tiles=gu_tiles,
            hid_tiles=hid_tiles,
            n_experts=total_e,
            in_tiles=in_tiles,
            packed=packed,
        )

    module = EXL3SwitchGLU(
        gu_trellis=mx.array(gu).view(mx.uint16),
        gu_suh=mx.array(gu_suh),
        gu_svh=mx.array(gu_svh),
        dn_trellis=mx.array(dn).view(mx.uint16),
        dn_suh=mx.array(dn_suh),
        dn_svh=mx.array(dn_svh),
        k=k,
        cb=codebook_mode_from_flags(mcg=False, mul1=mul1),
        activation=activation,
    )
    mx.eval(
        module._gu_trellis,
        module._gu_suh,
        module._gu_svh,
        module._dn_trellis,
        module._dn_suh,
        module._dn_svh,
    )
    return module


def load_dense_layer(ckpt: Exl3Checkpoint, prefix: str, *, mul1: bool = True) -> EXL3Layer:
    """Build an ``EXL3Layer`` from one dense group of the checkpoint."""
    trellis = ckpt.np(prefix + ".trellis")
    suh = ckpt.np(prefix + ".suh") if ckpt.has(prefix + ".suh") else None
    svh = ckpt.np(prefix + ".svh") if ckpt.has(prefix + ".svh") else None
    k = packed_to_k(trellis.shape[2])
    in_features = len(suh) if suh is not None else trellis.shape[0] * 16
    out_features = len(svh) if svh is not None else trellis.shape[1] * 16
    return EXL3Layer(
        key=prefix,
        in_features=in_features,
        out_features=out_features,
        k=k,
        trellis=trellis,
        suh=suh,
        svh=svh,
        mul1=mul1,
    )


def load_dense_linear(
    ckpt: Exl3Checkpoint, prefix: str, *, mul1: bool = True
) -> EXL3Linear:
    """``EXL3Linear`` for one dense group of the checkpoint."""
    return EXL3Linear(load_dense_layer(ckpt, prefix, mul1=mul1))
