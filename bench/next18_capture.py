# Copyright © 2026 Adam Durham (hermes-gw)
"""next18 capture harness -- record real decode/verify indexer tensors on a live boot.

WHAT THIS IS
------------
A dependency-light (numpy + mlx only) capture+diff hook for the DSv4.1 indexer.
It is meant to run *inside the model server process on the cluster*, where the
real activations exist, to produce the ground-truth evidence the off-line
synthetic suite (``tests/test_dsv41_indexer_smallm_hier.py``) cannot: the ACTUAL
``q`` / ``index_k`` / ``w`` / ``lens`` and the ACTUAL returned ``[b, n, k]`` int32
index tensor at decode (n==1) and verify (n==4), for BOTH the candidate-source
layer and the consumer layers, together with an elementwise A/B diff of the two
paths (hierarchical vs fallback) on those SAME captured inputs.

It does NOT decide correctness -- it records evidence and the diff count. Read
the diff counts; a non-zero count on a real decode/verify call is the abort
signal.

USAGE (inside the model server process)
---------------------------------------
Set the env BEFORE the process imports this module, then import it once; the
module auto-installs when ``DSV41_NEXT18_CAPTURE`` is set::

    DSV41_NEXT18_CAPTURE=/tmp/next18.npz \\
    DSV41_NEXT18_CAPTURE_NS=1,4 \\
    <launch the server>

and add ``import bench.next18_capture  # noqa: F401`` (or put it on the
``sitecustomize`` / bootstrap import list) so the hook is installed. To install
explicitly instead of via env auto-install::

    from bench import next18_capture as cap
    cap.install("/tmp/next18.npz", ns=(1, 4), max_calls=128, ab=True)

The hook monkeypatches ``Indexer.__call__``. For every call whose row-count ``n``
is in ``NS`` it records:

* raw call args   -- x, qr, start_pos, offset, freqvec, index_k  (for A/B replay)
* derived inputs  -- q, index_k, w, lens, k, block, nb, ratio    (see NOTE)
* call metadata   -- layer_id, role flags (is_candidate_source, uses_candidates,
                     owns_k), n, b, offset, start_pos
* the REAL output -- the returned ``[b, n, k]`` int32 tensor (with +offset and
                     the -1 mask applied, exactly as shipped)
* A/B diff        -- if ``ab`` is set, the two paths are re-run on the SAME raw
                     args (hier forced via ``_FENCE_MIN_ROWS=-1``; fallback via
                     ``_HIER=False``) and the elementwise difference is stored:
                     ``ab_ndiff`` (count) and ``ab_pos`` (up to 32 differing
                     [b,n,k] positions).

NOTE on "derived inputs": they are RECOMPUTED in the hook with the same
expressions ``indexer.Indexer.__call__`` uses (``wq_b`` -> ``rope_tail`` ->
``fake_quant_fp4_ue8m0`` for q; ``weights_proj`` for w; the causal ``lens``). This
duplicate is capture-only and is what makes input capture work on ALL paths
(tiled, untiled, hierarchical) without reaching into private internals. The A/B
diff, by contrast, re-runs the REAL ``__call__`` and is therefore exact.

MEMORY
------
Bounded: a ring of at most ``max_calls`` (default 128) recent captured calls is
held; each flush writes the whole ring atomically (temp file + ``os.replace``) so
the target ``.npz`` never grows without bound and a crash mid-write cannot
corrupt a previous one. Set ``DSV41_NEXT18_CAPTURE_MAX=0`` to disable the cap
(NOT recommended; unbounded by design is exactly what this avoids). ``index_k``
is stored bf16 (half the bytes); set ``DSV41_NEXT18_CAPTURE_STORE_K=0`` to store
only its shape + sha256 instead of the buffer when space is tight.

ENV
---
* ``DSV41_NEXT18_CAPTURE``        output ``.npz`` path (unset => module inert).
* ``DSV41_NEXT18_CAPTURE_NS``     comma list of ``n`` to capture (default "1,4").
* ``DSV41_NEXT18_CAPTURE_MAX``    ring size (default 128).
* ``DSV41_NEXT18_CAPTURE_AB``     "1" to run the two-path diff per call (default 1).
* ``DSV41_NEXT18_CAPTURE_STORE_K``"0" to skip storing the full index_k buffer.
* ``DSV41_NEXT18_CAPTURE_FLUSH``  flush every K calls in addition to atexit
                                   (default = ring size). "1" => flush each call.

FLUSH IS OFF the request thread (R1b fix). ``flush()`` snapshots the ring on the
calling thread (cheap: a list copy) and does the expensive `np.savez_compressed`
+ `os.replace` + meta write in a **single daemon writer thread**. A single-slot
pending buffer coalesces back-to-back flushes so a slow deflate never queues up
or blocks a forward. The R1 SIGKILL was the inline `np.savez_compressed` of a
~197 MB `.npz` (full `index_k` per record) starving the runner event channel;
this makes the request thread return immediately.

RUNTIME GATE (never capture during a timing arm). Because the hook is installed
once at engine construction, capture cannot be turned on/off by the launch env
alone without a reboot. ``DSV41_NEXT18_CAPTURE_GATE`` names a sentinel path; if
that file EXISTS the hook is an immediate no-op (~one stat per indexer call, no
tensor conversion, no A/B). Timing arms run with the sentinel present; the
capture leg ``rm``s it. Unset => always active (R1 behaviour).

* ``DSV41_NEXT18_CAPTURE_GATE``   sentinel path; hook is inert while it exists.
* ``DSV41_NEXT18_CAPTURE_INTERVAL`` seconds between interval flushes (default 30;
                                   0 => no interval flush, atexit/ring only).
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np

try:  # mlx is required only when actually installing inside the server
    import mlx.core as mx
except Exception:  # pragma: no cover - import smoke on hosts without mlx
    mx = None


# --------------------------------------------------------------------------
# env
# --------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_ns(name: str, default=(1, 4)) -> tuple[int, ...]:
    v = os.environ.get(name)
    if not v:
        return tuple(default)
    out = []
    for part in v.replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return tuple(out) or tuple(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# capture state
# --------------------------------------------------------------------------
class _Capture:
    def __init__(self, path: str, ns, max_calls: int, ab: bool,
                 store_k: bool, flush_every: int,
                 gate_path: str | None = None, interval: float = 30.0):
        self.path = path
        self.ns = set(int(x) for x in ns)
        self.max_calls = int(max_calls)          # 0 => unlimited (avoid)
        self.ab = bool(ab)
        self.store_k = bool(store_k)
        self.flush_every = max(1, int(flush_every))
        self.ring: deque[dict[str, Any]] = deque(maxlen=(self.max_calls or None))
        self.seen = 0
        self.saved = 0
        self.flushed = 0
        self._since_flush = 0
        self._lock = threading.Lock()
        self._installed = False
        self._orig_call = None
        self._orig_hier = None
        self._orig_fence = None
        # R1b: off-thread flush + runtime gate.
        # ``gate_path`` is a sentinel file; while it EXISTS the hook is inert
        # (timing-arm guard). ``interval`` > 0 starts a daemon interval-flusher.
        self.gate_path = gate_path
        self.interval = float(interval)
        self._flush_lock = threading.Lock()       # serialize writer-thread flushes
        self._pending: list[dict[str, Any]] | None = None
        self._wake = threading.Event()
        self._writer: threading.Thread | None = None
        self._timer: threading.Thread | None = None
        self._stop = threading.Event()

    def _gated(self) -> bool:
        """True while the timing-arm sentinel exists (hook must be a no-op).

        One ``os.path.exists`` per indexer call: no tensor conversion, no A/B,
        no ring append -- the cost of a disabled hook is one stat.
        """
        return bool(self.gate_path) and os.path.exists(self.gate_path)

    # -- numpy conversion helpers -----------------------------------------
    @staticmethod
    def _np(a):
        """Convert an mlx array to numpy, handling bf16 (no PEP3118 format).

        A bf16 mlx array raises on ``np.array`` (2-byte items, 'B' format); we
        round-trip it losslessly through a uint16 bit-view instead (marked by
        ``*_bf16`` metadata keys) -- 2 bytes/elem, half an fp32 copy.
        """
        if a is None:
            return None
        dt = getattr(a, "dtype", None)
        if mx is not None and dt == mx.bfloat16:
            return np.array(a.view(mx.uint16))
        return np.array(a)

    # -- install / uninstall ----------------------------------------------
    def install(self):
        from mlx_lm.models.deepseek_v41 import indexer as IX
        from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H
        self._IX = IX
        self._H = H
        self._orig_call = IX.Indexer.__call__
        self._orig_hier = H.hierarchical_topk_prod
        self._orig_fence = IX._FENCE_MIN_ROWS
        cap = self
        orig = self._orig_call

        def wrapper(self_ix, x, qr, start_pos, offset, freqvec, index_k, shared):
            out = orig(self_ix, x, qr, start_pos, offset, freqvec, index_k, shared)
            try:
                cap._maybe_capture(self_ix, x, qr, start_pos, offset, freqvec,
                                   index_k, shared, out)
            except Exception as e:  # capture must NEVER break the model
                import traceback
                print(f"[next18_capture] capture error: {e}\n"
                      + traceback.format_exc(), file=sys.stderr, flush=True)
            return out

        IX.Indexer.__call__ = wrapper
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        self._IX.Indexer.__call__ = self._orig_call
        self._H.hierarchical_topk_prod = self._orig_hier
        self._IX._FENCE_MIN_ROWS = self._orig_fence
        self._installed = False

    # -- the hook ----------------------------------------------------------
    def _record(self, **kw):
        with self._lock:
            self.ring.append(kw)
            self.saved += 1
            self._since_flush += 1
            due = (self._since_flush >= self.flush_every)
        if due:
            # R1b: schedule the flush OFF the request thread. ``flush()`` only
            # snapshots the ring here and returns; the writer thread does the
            # savez_compressed. A blocking call would re-create the R1 SIGKILL.
            self.flush(block=False)

    def _maybe_capture(self, ix, x, qr, start_pos, offset, freqvec, index_k, shared, out):
        if self._gated():
            return                                     # timing-arm sentinel present
        from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail
        from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0

        bsz, n, _dim = x.shape
        with self._lock:
            self.seen += 1
        if n not in self.ns:
            return

        nb = index_k.shape[1]
        # --- recompute the score inputs exactly as __call__ does -------------
        q = ix.wq_b(qr).reshape(bsz, n, ix.n_heads, ix.head_dim)
        q = rope_tail(q, ix.rope_head_dim,
                      *cos_sin_at(freqvec, mx.arange(start_pos, start_pos + n)))
        q = fake_quant_fp4_ue8m0(q, 32)
        w = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
        lens = ((start_pos + mx.arange(n) + 1) // ix.ratio)[:, None]
        k = min(ix.index_topk, nb)
        mx.eval(q, w, lens)

        out_np = self._np(out)
        rec: dict[str, Any] = {
            "layer_id": int(ix.layer_id),
            "n": int(n), "b": int(bsz), "nb": int(nb),
            "ratio": int(ix.ratio),
            "is_candidate_source": bool(ix.is_candidate_source),
            "uses_candidates": bool(ix.uses_candidates),
            "owns_k": bool(ix.owns_k),
            "offset": int(offset), "start_pos": int(start_pos),
            "k": int(k), "block": int(ix.candidate_block_size),
            "hier_block": int(getattr(self._IX, "_HIER_BLOCK", 8)),
            "out": out_np,
            "q": self._np(q.astype(mx.float32)),
            "w": self._np(w.astype(mx.float32)),
            "lens": self._np(lens.astype(mx.int32)),
        }
        if self.store_k:
            rec["index_k"] = self._np(index_k.astype(mx.bfloat16))
            rec["index_k_is_bf16"] = True
        else:
            kb = np.array(index_k.astype(mx.float32)).tobytes()
            rec["index_k_sha256"] = hashlib.sha256(kb).hexdigest()
            rec["index_k_shape"] = np.array(index_k.shape, np.int32)
        rec["qr"] = self._np(qr.astype(mx.float32))
        rec["x"] = self._np(x.astype(mx.float32))
        rec["freqvec"] = self._np(freqvec.astype(mx.float32))

        # --- optional A/B diff on the SAME raw args --------------------------
        if self.ab:
            rec["ab_ndiff"], rec["ab_pos"] = self._ab_diff(
                ix, x, qr, start_pos, offset, freqvec, index_k, shared)

        self._record(**rec)

    def _ab_diff(self, ix, x, qr, start_pos, offset, freqvec, index_k, shared) -> tuple[int, np.ndarray]:
        """Re-run the real __call__ both ways on the SAME inputs; elementwise diff."""
        from mlx_lm.models.deepseek_v41.model import SharedState
        IX = self._IX
        orig = self._orig_call
        cand = getattr(shared, "candidates", None)
        saved_hier = IX._HIER
        saved_fence = IX._FENCE_MIN_ROWS

        def _run(hier: bool, fence: int):
            sh = SharedState()
            if cand is not None:
                sh.candidates = cand
            IX._HIER = hier
            IX._FENCE_MIN_ROWS = fence
            o = orig(ix, x, qr, start_pos, offset, freqvec, index_k, sh)
            mx.eval(o)
            return np.array(o)

        try:
            # fallback arm: feature OFF entirely (n-independent, exact path)
            a = _run(False, saved_fence)
            # hier arm: feature ON, guard forced to -1 so ALL n take HIER
            b = _run(True, -1)
        finally:
            IX._HIER = saved_hier
            IX._FENCE_MIN_ROWS = saved_fence
        nd = int((a != b).sum())
        pos = np.argwhere(a != b)[:32].astype(np.int32) if nd else np.zeros((0, 3), np.int32)
        return nd, pos

    # -- flush (atomic, ring-bounded, OFF the request thread) -------------
    def flush(self, block: bool = False, wait: float | None = None):
        """Snapshot the ring and hand it to the daemon writer thread.

        R1b: this NEVER deflates on the calling thread. The R1 SIGKILL was a
        synchronous ``np.savez_compressed`` of a ~197 MB ``.npz`` inline on the
        server's request thread (multi-minute zlib), so no runner events were
        emitted and the 45 s hang-watchdog killed the runner. Here the expensive
        work happens in ``_writer_loop``.

        ``block=True`` (used by atexit / the final capture flush) waits for the
        write to land. In the hook path it is called with ``block=False`` so the
        forward returns immediately.
        """
        with self._lock:
            if not self.ring:
                return
            recs = list(self.ring)
            self._since_flush = 0
        # Single-slot coalescing: if a flush is already pending, replace it with
        # this (newer) snapshot. The ring is cumulative, so the newest snapshot
        # is a superset -- no data is lost, and flushes never queue up.
        with self._flush_lock:
            self._pending = recs
        self._wake.set()

        if block:
            deadline = time.time() + (wait if wait is not None else 120.0)
            while time.time() < deadline:
                with self._flush_lock:
                    done = self._pending is None
                if done:
                    return
                time.sleep(0.05)

    def _writer_loop(self):
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._flush_lock:
                if self._pending is None:
                    continue
                recs, self._pending = self._pending, None
            try:
                self._write_ring(recs)
            except Exception as e:  # a capture must NEVER take down the server
                import traceback
                print(f"[next18_capture] flush error: {e}\n"
                      + traceback.format_exc(), file=sys.stderr, flush=True)

    def _timer_loop(self):
        while not self._stop.wait(self.interval):
            self.flush(block=False)

    def _start_workers(self):
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="next18-capture-writer", daemon=True)
        self._writer.start()
        if self.interval and self.interval > 0:
            self._timer = threading.Thread(target=self._timer_loop,
                                           name="next18-capture-timer", daemon=True)
            self._timer.start()

    def _write_ring(self, recs):
        arrays: dict[str, np.ndarray] = {}
        meta_rows = []
        _ARR_KEYS = ("out", "q", "w", "lens", "index_k", "qr", "x", "freqvec",
                     "ab_pos", "index_k_shape")
        _SCALAR_KEYS = ("layer_id", "n", "b", "nb", "ratio", "is_candidate_source",
                        "uses_candidates", "owns_k", "offset", "start_pos", "k",
                        "block", "hier_block", "ab_ndiff", "index_k_is_bf16")
        for i, r in enumerate(recs):
            meta_rows.append({kk: vv for kk, vv in r.items()})
            for key in _ARR_KEYS:
                if key in r and r[key] is not None:
                    arrays[f"c{i:04d}_{key}"] = np.asarray(r[key])
            for key in _SCALAR_KEYS:
                if key in r and r[key] is not None:
                    arrays[f"c{i:04d}_{key}"] = np.asarray(r[key])
        arrays["meta_seen"] = np.array(self.seen, np.int64)
        arrays["meta_saved"] = np.array(self.saved, np.int64)
        arrays["meta_nrec"] = np.array(len(recs), np.int64)
        arrays["meta_walltime"] = np.array(time.time(), np.float64)

        path = self.path
        d = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(d, exist_ok=True)
        # np.savez_compressed APPENDS '.npz' when the name lacks it, so the temp
        # name must already end in '.npz' or os.replace cannot find it. Unique
        # per write so successive flushes never share a temp file.
        tmp = f"{path}.tmp.{os.getpid()}.{self.flushed}.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        # metadata sidecar (human-readable, small): OVERWRITE so it stays bounded
        # to the ring (appending would grow without bound across flushes).
        try:
            with open(path + ".meta.jsonl", "w") as fh:
                for m in meta_rows:
                    import json
                    fh.write(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray)
                                            else v) for k, v in m.items()}) + "\n")
        except Exception:
            pass
        with self._lock:
            self.flushed += 1
        print(f"[next18_capture] flushed {len(recs)} calls -> {path} "
              f"(seen={self.seen} saved={self.saved} flush#{self.flushed})",
              file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# module-level API
# --------------------------------------------------------------------------
_CAP: _Capture | None = None


def install(path: str | None = None, *, ns=None, max_calls: int | None = None,
            ab: bool | None = None, store_k: bool | None = None,
            flush_every: int | None = None, gate: str | None = None,
            interval: float | None = None) -> _Capture:
    """Install the capture hook (idempotent). Returns the live _Capture."""
    global _CAP
    if _CAP is not None:
        return _CAP
    if mx is None:
        raise RuntimeError("mlx is required to install the capture hook")
    path = path or os.environ.get("DSV41_NEXT18_CAPTURE")
    if not path:
        raise RuntimeError("no capture path: set DSV41_NEXT18_CAPTURE or pass path")
    ns = ns if ns is not None else _env_ns("DSV41_NEXT18_CAPTURE_NS", (1, 4))
    max_calls = (max_calls if max_calls is not None
                 else _env_int("DSV41_NEXT18_CAPTURE_MAX", 128))
    ab = ab if ab is not None else _env_bool("DSV41_NEXT18_CAPTURE_AB", True)
    store_k = (store_k if store_k is not None
               else _env_bool("DSV41_NEXT18_CAPTURE_STORE_K", True))
    flush_every = (flush_every if flush_every is not None
                   else _env_int("DSV41_NEXT18_CAPTURE_FLUSH", max_calls or 1))
    gate = gate if gate is not None else (os.environ.get("DSV41_NEXT18_CAPTURE_GATE")
                                          or (path + ".gate"))
    interval = (interval if interval is not None
                else _env_float("DSV41_NEXT18_CAPTURE_INTERVAL", 30.0))
    cap = _Capture(path, ns, max_calls, ab, store_k, flush_every,
                   gate_path=gate, interval=interval)
    cap.install()
    cap._start_workers()                      # R1b: daemon writer (+ interval timer)
    _CAP = cap
    atexit.register(cap.uninstall)

    def _final_flush():                        # blocking so atexit waits for the write
        try:
            cap.flush(block=True, wait=180.0)
        except Exception:
            pass

    atexit.register(_final_flush)
    print(f"[next18_capture] installed: path={path} ns={sorted(cap.ns)} "
          f"max_calls={max_calls} ab={ab} store_k={store_k} "
          f"flush_every={flush_every} gate={gate} interval={interval} "
          f"off_thread_flush=True", file=sys.stderr, flush=True)
    return cap


def uninstall():
    global _CAP
    if _CAP is not None:
        _CAP.uninstall()
        _CAP = None


def flush():
    if _CAP is not None:
        _CAP.flush()


# auto-install when the env names a path (so a bare import is enough in a boot)
if mx is not None and os.environ.get("DSV41_NEXT18_CAPTURE"):
    try:
        install()
    except Exception as _e:  # never break the host process
        print(f"[next18_capture] auto-install failed: {_e}", file=sys.stderr, flush=True)
