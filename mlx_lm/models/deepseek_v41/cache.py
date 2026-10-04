# Vendored from PipeNetwork/deepseek-v41-mlx (Apache-2.0), upstream a9567ae plus the
# node-local MTP/DSpark port edits (exo docs phase 3). See README.md here.
"""Incremental state for DeepSeek-V4.1 decode and chunked prefill.

Per layer:

* **window ring** — every layer: a circular buffer of ``window_size`` KV
  entries (position p at slot p % window), values already FP8 fake-quantized;
* **compressed KV** — kv_source layers only: one FP4-fake-quantized latent per
  complete group, append-only. Consumer layers hold a *reference* to their
  source's buffer, so a source's write is immediately visible downstream;
* **compressor partial group** — ratio>1 sources: fp32 kv/score rows of the
  open group;
* **index keys** — layers that are both kv and index sources: the FP4-fake-
  quantized index-key cache, read by every index source below them.

Model-level: the engram compressed-token-id history, and the global offset.

Storage dtype (local change). The three *quantized-grid* buffers — ``win_kv``,
``comp_kv`` and ``index_k`` — are stored in **bf16**, always, regardless of the
``dtype`` a caller passes (kept accepted for compatibility; ``engram_ids`` is
int64 and ``CompressorState`` stays fp32). This is EXACT, not lossy: every
stored value is pre-quantized onto a coarse grid (win_kv: fp8 e4m3 <=4
significant bits; comp_kv: fp4 e2m1 latents <=2 bits x e4m3 scales; index_k:
fp4 e2m1 x ue8m0 power-of-two scales; MTP draft KV: fp8). Such products carry
<=6 significant bits, exactly representable in bf16's 8 mantissa bits, so
``bf16 -> fp32`` widening reproduces the identical fp32 value every reader saw
when the buffers were fp32. bf16 halves the buffer bytes (a 1M session's
compressed/index caches are ~3.2 GiB in bf16, ~6.4 GiB fp32).

Capacity (local change). ``comp_kv`` / ``index_k`` / ``engram_ids`` start at a
small ``capacity`` and grow on demand (``ensure_capacity``) instead of being
preallocated to the cap, so an idle session does not cost the full cap up front.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .compressor import CompressorState
from .config import ModelArgs

NEG_INF = float("-inf")

#: Rows a fresh ``ModelCache`` allocates before it has to grow. Small enough that
#: an idle session costs little, large enough that ordinary turns never grow.
DEFAULT_INITIAL_CAPACITY = 65536


class CapacityError(RuntimeError):
    """A requested token position does not fit the cache's ``max_seq_len``.

    Raised by :meth:`ModelCache.ensure_capacity` when growth would exceed the
    logical maximum, or when the reallocation itself fails (OOM). The exo
    engine maps this to its own ``Dsv41UnsupportedFeature`` refusal (see
    ``dsv41/session.py``), so the request is failed cleanly instead of crashing
    the runner.
    """


def _latent_rows(tokens: int, ratio: int) -> int:
    """Complete group rows needed to cover ``tokens`` open-group positions."""
    return -(-int(tokens) // max(int(ratio), 1))


class LayerCache:
    def __init__(self, bsz: int, args: ModelArgs, layer_id: int, capacity: int,
                 dtype=mx.float32):
        self.window = args.window_size
        self.ratio = args.compress_ratio(layer_id)
        self.is_kv_source = layer_id in args.kv_source_layers
        # bf16 ALWAYS for the quantized-grid buffers (see module docstring): a
        # caller's ``dtype`` is accepted but never widens these, because bf16 is
        # lossless for their values. ``self.dtype`` is what writers cast to.
        self.dtype = mx.bfloat16
        self.bsz = int(bsz)
        del dtype  # accepted for compatibility; quantized-grid storage is bf16

        self.win_kv = mx.zeros((bsz, self.window, args.head_dim), dtype=self.dtype)

        self.comp_kv = None
        self.comp_state = None
        self.index_k = None
        if self.is_kv_source:
            n_comp = _latent_rows(capacity, self.ratio)
            self.comp_kv = mx.zeros((bsz, n_comp, args.head_dim), dtype=self.dtype)
            if self.ratio > 1:
                self.comp_state = CompressorState(bsz, self.ratio, args.head_dim)
            if layer_id in args.index_source_layers:
                self.index_k = mx.zeros((bsz, n_comp, args.index_head_dim),
                                        dtype=self.dtype)

    def grow_latent(self, new_rows: int, used_rows: int) -> None:
        """Reallocate comp_kv/index_k to ``new_rows`` group rows, copying used rows.

        Copies ONLY the rows a live read can reach (``ceil(offset / ratio)``,
        clamped to the old width) and zero-fills the rest. Called from
        ``ModelCache.ensure_capacity`` so growth happens at an eval-clean
        boundary, before the chunk's forward graph is built.
        """
        if self.comp_kv is None:
            return
        old = self.comp_kv
        if new_rows <= old.shape[1]:
            return
        used = max(0, min(int(used_rows), old.shape[1]))
        self.comp_kv = mx.zeros((old.shape[0], new_rows, old.shape[2]),
                                dtype=old.dtype)
        if used > 0:
            self.comp_kv[:, :used] = old[:, :used]
        idx = self.index_k
        if idx is not None:
            self.index_k = mx.zeros((idx.shape[0], new_rows, idx.shape[2]),
                                    dtype=idx.dtype)
            if used > 0:
                self.index_k[:, :used] = idx[:, :used]

    # ---- window ring ----

    def window_chrono(self, pos: int) -> mx.array:
        """The cached window KV in chronological order: positions
        [pos - Wp, pos) where Wp = min(pos, window). [b, Wp, head_dim]."""
        w = self.window
        wp = min(pos, w)
        if wp == 0:
            return self.win_kv[:, :0]
        first = pos - wp
        slots = (first + mx.arange(wp)) % w
        return self.win_kv[:, slots]

    def write_window(self, pos: int, kv: mx.array):
        """Write chunk KV at positions [pos, pos+n) into the ring."""
        n = kv.shape[1]
        keep = min(n, self.window)
        tail = kv[:, n - keep:]
        slots = (pos + n - keep + mx.arange(keep)) % self.window
        self.win_kv[:, slots] = tail.astype(self.dtype)

    # ---- session reuse (workstream E): exact-state snapshot / restore ----

    def ring_snapshot(self):
        """Materialized copy of the window ring.

        The ring is the one position-addressed buffer a rollback cannot
        reconstruct on its own: slots alias every ``window`` positions, so a
        discarded write at ``q`` silently corrupts a live read of ``q -
        window``. A session checkpoint therefore keeps a copy (~64 KB at
        window 128 / head_dim 512 bf16)."""
        return mx.array(self.win_kv)

    def ring_restore(self, snap) -> None:
        """Rebind the ring to a fresh copy of ``snap`` (never aliases it)."""
        self.win_kv = mx.array(snap)

    def reset_carry(self) -> None:
        """Canonicalize the open-group carry to "no carried rows" (a group
        boundary): ``kv`` rows zeroed, ``score`` rows back to NEG_INF, exactly
        as a freshly allocated cache has them. A rollback to a group boundary
        needs no carry history because the next compressor call writes every
        row it reads."""
        cs = self.comp_state
        if cs is None:
            return
        cs.kv_state = mx.zeros_like(cs.kv_state)
        cs.score_state = mx.full(cs.score_state.shape, NEG_INF, dtype=cs.score_state.dtype)


class ModelCache:
    """Incremental cache. ``max_seq_len`` is the logical cap; ``capacity`` rows
    are allocated, grown geometrically by :meth:`ensure_capacity`."""

    def __init__(self, args: ModelArgs, bsz: int = 1, max_seq_len: int | None = None,
                 dtype=mx.float32, initial_capacity: int | None = None):
        self.args = args
        self.bsz = int(bsz)
        self.max_seq_len = int(max_seq_len or min(args.max_seq_len, 4096))
        self.offset = 0
        cap = DEFAULT_INITIAL_CAPACITY if initial_capacity is None else int(initial_capacity)
        self.capacity = max(1, min(cap, self.max_seq_len))
        self.layers = [LayerCache(bsz, args, i, self.capacity, dtype)
                       for i in range(args.n_layers)]
        self.engram_ids = (np.zeros((bsz, self.capacity), dtype=np.int64)
                           if args.engram_layer_ids else None)

    # ---- capacity growth ----

    def ensure_capacity(self, required_tokens: int) -> None:
        """Ensure the cache can hold ``required_tokens`` positions right now.

        INVARIANT: any write at positions ``[p, p + n)`` must be preceded by
        ``ensure_capacity(p + n)``. Growth reallocates the latent buffers
        (``comp_kv`` / ``index_k``) and ``engram_ids`` to cover the new
        positions, copying only the rows a live read can reach and zero-filling
        the rest, and calls ``mx.clear_cache()`` afterwards. It must run at an
        EVAL-CLEAN boundary -- before the chunk's forward graph is built, never
        mid-graph -- so the reallocation happens between committed graphs.

        Raises :class:`CapacityError` when ``required_tokens > max_seq_len`` or
        when a reallocation fails (OOM), so the engine fails one request instead
        of crashing the runner.
        """
        req = int(required_tokens)
        if req <= self.capacity:
            return
        if req > self.max_seq_len:
            raise CapacityError(
                f"cache holds {self.max_seq_len} tokens; a request at "
                f"{req} exceeds it")
        new_cap = min(max(req, self.capacity * 2), self.max_seq_len)
        try:
            new_engram = None
            if self.engram_ids is not None:
                new_engram = np.zeros((self.engram_ids.shape[0], new_cap),
                                      dtype=np.int64)
                used = max(0, min(int(self.offset), self.engram_ids.shape[1]))
                if used:
                    new_engram[:, :used] = self.engram_ids[:, :used]
            for lc in self.layers:
                used = _latent_rows(min(int(self.offset), self.capacity), lc.ratio)
                lc.grow_latent(_latent_rows(new_cap, lc.ratio), used)
            if new_engram is not None:
                self.engram_ids = new_engram
        except CapacityError:
            raise
        except Exception as e:  # OOM or any allocator failure -> clean refusal
            raise CapacityError(
                f"cache growth to {new_cap} tokens failed: "
                f"{type(e).__name__}: {e}") from e
        self.capacity = new_cap
        mx.clear_cache()
