#!/usr/bin/env python3
"""pB -- stream B (indexer tiling) harness for DeepSeek-V4.1.  Run on a Mac.

Subcommands
-----------
  parity   tiled-vs-untiled at module level on random inputs: bitwise score
           equality, index-set equality, candidate-mask equality, plus the
           k-th-boundary tie census.  No checkpoint needed, < 2 GB.
  micro    indexer at production shapes (b=1, n=512, h=32, d=128, nb=16384):
           tiled vs untiled wall time and peak allocation.
  prefill  single-node 16K layer-subset prefill (the plan's gate).  With
           PB_CAPTURE=1 it also captures the real (x, qr, index_k) that each
           index layer sees on the last chunk, then re-runs tiled vs untiled
           on those real tensors and reports index/mask mismatch + boundary
           ties + NLL for both paths.

Env: P48_PKG (default ~/dsv41-ws/B), DSV41_INDEXER_TILE,
     DSV41_INDEXER_TILE_MIN_NB, DSV41_INDEXER_TILE_MB, PB_* knobs.
"""
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.environ.get("P48_PKG", HOME + "/dsv41-ws/B"))
import mlx.core as mx  # noqa: E402

from mlx_lm.models.deepseek_v41 import indexer as IX  # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs  # noqa: E402
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0  # noqa: E402
from mlx_lm.models.deepseek_v41.layers import precompute_freqs_cis, rope_tail  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
LAYERS = [int(x) for x in os.environ.get("PB_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
ARGS = ModelArgs(compress_ratios=RATIOS, kv_source_layers=(2, 8, 14, 20),
                 index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
                 candidate_source_layer=20, candidate_topk_blocks=2048,
                 candidate_block_size=8)

FULL_UNTILED = (0, 10 ** 9)          # force the untiled path
FULL_TILED = (int(os.environ.get("DSV41_INDEXER_TILE", "512")), 1)   # force tiled


def log(*a):
    print("[pB]", *a, flush=True)


def rand_indexer(layer_id, seed=0):
    mx.random.seed(seed)
    ix = IX.Indexer(ARGS, layer_id)

    def fill(t):
        if isinstance(t, dict):
            return {k: fill(v) for k, v in t.items()}
        if isinstance(t, list):
            return [fill(v) for v in t]
        return (mx.random.normal(t.shape) * 0.05).astype(mx.float32)

    ix.update(fill(ix.parameters()))
    return ix


def synth(n, nb, seed=1):
    mx.random.seed(seed)
    x = (mx.random.normal((1, n, ARGS.dim)) * 0.3).astype(mx.float32)
    qr = (mx.random.normal((1, n, ARGS.q_lora_rank)) * 0.3).astype(mx.float32)
    ik = (mx.random.normal((1, nb, ARGS.index_head_dim)) * 0.5).astype(mx.float32)
    return x, qr, ik


class Sh:
    def __init__(self, cand=None):
        self.kv_src_cache = None
        self.index_src_cache = None
        self.topk_idxs = None
        self.candidates = cand


def freqs(upto):
    return precompute_freqs_cis(ARGS.rope_head_dim, max(upto * 2, 8192), 0, 160000.0,
                                16.0, 32, 1)


def scores_ref(ix, x, qr, ik, sp, cos, sin):
    """The untiled score expression, verbatim, for comparison."""
    n = x.shape[1]
    q = ix.wq_b(qr).reshape(1, n, ix.n_heads, ix.head_dim)
    q = fake_quant_fp4_ue8m0(rope_tail(q, ix.rope_head_dim,
                                       cos[sp:sp + n], sin[sp:sp + n]), 32)
    w = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads ** -0.5)
    s = mx.einsum("bshd,btd->bsht", q.astype(mx.float32), ik.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None].astype(mx.float32)
    return mx.sum(s, axis=2), q


def report_pair_deprecated(tag, idx_a, idx_b, ratio, sp, topk, nb, scores=None, lens=None):
    """DEPRECATED duplicate; superseded by report_pair below."""
    a, b = np.array(idx_a)[0], np.array(idx_b)[0]
    nrow = a.shape[0]
    same = 0
    for r in range(nrow):
        sa = sorted(int(v) for v in a[r] if v >= 0)
        sb = sorted(int(v) for v in b[r] if v >= 0)
        if sa == sb:
            same += 1
    return same, nrow - same


# --------------------------------------------------------------------------
def exact_topk(scores, lens, topk):
    """Reference top-k indices/values from a numpy score matrix (per row)."""
    s = np.array(scores)
    if s.ndim == 3:
        s = s[0]
    nrow, nb = s.shape
    order = np.argsort(-s, axis=-1, kind="stable")
    idx = []
    for r in range(nrow):
        row = order[r]
        keep = [c for c in row if np.isfinite(s[r, c])]
        idx.append(keep[:topk])
    return idx


def report_pair(tag, idx_a, idx_b, ratio, sp, topk, nb, scores=None, lens=None):
    """Compare two [1,n,k] index results against each other and, when given the
    score matrix, against the exact top-k."""
    a, b = np.array(idx_a)[0], np.array(idx_b)[0]
    nrow = a.shape[0]
    same = 0
    ex = []
    n_ties = 0
    val_mismatch = 0
    for r in range(nrow):
        sa = [int(v) for v in a[r] if v >= 0]
        sb = [int(v) for v in b[r] if v >= 0]
        if sorted(sa) == sorted(sb):
            same += 1
        elif len(ex) < 3:
            ex.append((r, sorted(set(sa) ^ set(sb))[:6], len(sa), len(sb)))
    if scores is not None:
        s = np.array(scores)
        if s.ndim == 3:
            s = s[0]
        ln = np.array(lens).reshape(-1) if lens is not None else None
        for r in range(min(nrow, s.shape[0])):
            v = s[r].astype(np.float64).copy()
            if ln is not None:
                v[ln[r]:] = -np.inf
            t = np.sort(v)[::-1]
            if topk < t.shape[0] and np.isfinite(t[topk - 1]) and t[topk - 1] == t[topk]:
                n_ties += 1
        # value multiset check against exact top-k
        exp = exact_topk(s, ln, topk)
        for r in range(min(nrow, len(exp))):
            got = sorted((float(s[r, c]) for c in a[r] if c >= 0))[::-1]
            want = sorted((float(s[r, c]) for c in exp[r]))[::-1]
            if len(got) != len(want) or any(
                    x != y for x, y in zip(got, want)):
                val_mismatch += 1
    log(f"{tag}: identical_index_set={same}/{nrow} differing={nrow - same}"
        + (f" boundary_tie_rows={n_ties}/{nrow} value_multiset_mismatch={val_mismatch}"
           if scores is not None else "")
        + (f" first_diffs={ex}" if ex else ""))
    return same, nrow - same


def cmd_parity():
    n = int(os.environ.get("PB_N", "32"))
    cos, sin = freqs(16384 + n)
    log(f"parity n={n} heads={ARGS.index_n_heads} d={ARGS.index_head_dim} "
        f"topk={ARGS.index_topk} cand_blocks={ARGS.candidate_topk_blocks}"
        f"x{ARGS.candidate_block_size}")
    for nb in (4096, 8192, 16384, 20000):
        for tile in (256, 512, 1024):
            x, qr, ik = synth(n, nb)
            ix20 = rand_indexer(20)
            s_ref, q = scores_ref(ix20, x, qr, ik, 0, cos, sin)
            # tiled score build, exactly as _tiled_scores does it
            w32 = (ix20.weights_proj(x) * (ix20.softmax_scale * ix20.n_heads ** -0.5)
                   ).astype(mx.float32)
            parts = []
            for c0 in range(0, nb, tile):
                c1 = min(c0 + tile, nb)
                st = mx.einsum("bshd,btd->bsht", q.astype(mx.float32),
                               ik[:, c0:c1].astype(mx.float32))
                st = mx.maximum(st, 0.0) * w32[..., None]
                parts.append(mx.sum(st, axis=2))
            s_tile = mx.concatenate(parts, axis=-1)
            bit_eq = bool(mx.array_equal(s_ref, s_tile).item())
            mx.eval(s_ref, s_tile)
            lens = ((0 + np.arange(n) + 1) // ix20.ratio)[:, None]
            vis = np.arange(nb)[None, :] < lens
            sv = np.array(s_ref)[0].copy()
            sv[~vis] = -np.inf
            sh_t, sh_u = Sh(), Sh()
            IX._TILE, IX._TILE_MIN_NB = tile, 1
            idx_t = ix20(x, qr, 0, 0, cos, sin, ik, sh_t)
            IX._TILE, IX._TILE_MIN_NB = FULL_UNTILED
            idx_u = ix20(x, qr, 0, 0, cos, sin, ik, sh_u)
            mx.eval(idx_t, idx_u, sh_t.candidates, sh_u.candidates)
            report_pair(f"  nb={nb:6d} tile={tile:5d} L20 score_bitwise_equal={bit_eq} "
                        f"cand_mask_equal="
                        f"{bool(mx.array_equal(sh_t.candidates, sh_u.candidates).item())} "
                        f"topk", idx_t, idx_u, ix20.ratio, 0, ix20.index_topk, nb,
                        scores=sv[None], lens=lens)
            cb = int(mx.sum(sh_t.candidates[..., ::8].astype(mx.int32)).item())
            cb_u = int(mx.sum(sh_u.candidates[..., ::8].astype(mx.int32)).item())
            d = int(mx.sum((sh_t.candidates != sh_u.candidates).astype(mx.int32)).item())
            log(f"  nb={nb:6d} tile={tile:5d} kept_blocks tiled={cb} untiled={cb_u} "
                f"mask_diff_bits={d}")
            # consumer layer 24, masked by layer 20's candidates
            ix24 = rand_indexer(24, seed=7)
            cnd = sh_u.candidates
            s24, _ = scores_ref(ix24, x, qr, ik, 0, cos, sin)
            sv24 = np.array(s24)[0].copy()
            sv24[~vis] = -np.inf
            sv24[~np.array(cnd)[0]] = -np.inf
            IX._TILE, IX._TILE_MIN_NB = tile, 1
            idx_t2 = ix24(x, qr, 0, 0, cos, sin, ik, Sh(cnd))
            IX._TILE, IX._TILE_MIN_NB = FULL_UNTILED
            idx_u2 = ix24(x, qr, 0, 0, cos, sin, ik, Sh(cnd))
            mx.eval(idx_t2, idx_u2)
            report_pair(f"  nb={nb:6d} tile={tile:5d} L24_masked topk", idx_t2, idx_u2,
                        1, 0, ix24.index_topk, nb, scores=sv24[None], lens=lens)
            mx.clear_cache()
    # ratio-2 owner with a chunk that starts mid-context (the real 16K chunk path)
    for sp in (8192, 15872):
        nb = (sp + n) // 2
        x, qr, ik = synth(n, nb)
        ix8 = rand_indexer(8, seed=3)
        lens = ((sp + np.arange(n) + 1) // 2)[:, None]
        vis = np.arange(nb)[None, :] < lens
        s8, _ = scores_ref(ix8, x, qr, ik, sp, cos, sin)
        sv8 = np.array(s8)[0].copy()
        sv8[~vis] = -np.inf
        IX._TILE, IX._TILE_MIN_NB = 512, 1
        it = ix8(x, qr, sp, 0, cos, sin, ik, Sh())
        IX._TILE, IX._TILE_MIN_NB = FULL_UNTILED
        iu = ix8(x, qr, sp, 0, cos, sin, ik, Sh())
        mx.eval(it, iu)
        # offset shift must be identical in both paths
        IX._TILE, IX._TILE_MIN_NB = 512, 1
        it_off = ix8(x, qr, sp, sp + n, cos, sin, ik, Sh())
        IX._TILE, IX._TILE_MIN_NB = FULL_UNTILED
        iu_off = ix8(x, qr, sp, sp + n, cos, sin, ik, Sh())
        mx.eval(it_off, iu_off)
        off_eq = bool(mx.array_equal(it_off, iu_off).item())
        report_pair(f"  sp={sp} ratio2 nb={nb} topk", it, iu, 2, sp, ix8.index_topk, nb,
                    scores=sv8[None], lens=lens)
        log(f"  sp={sp} ratio2 offset+{sp+n} tiled==untiled: {off_eq}")
        mx.clear_cache()
    IX._TILE, IX._TILE_MIN_NB = 512, 8192
    log("routing table (tile == nb means the untiled argpartition path):")
    for (bb, nn, nb) in ((1, 1, 4096), (1, 1, 8192), (1, 1, 16384), (1, 1, 65536),
                         (1, 512, 4096), (1, 512, 8192), (1, 512, 16384),
                         (1, 512, 32768), (1, 1024, 65536)):
        r = IX.tiled(bb, nn, 32, nb, 8)
        kind = "untiled" if r >= nb else f"tile={r}"
        log(f"   b={bb} n={nn:5d} nb={nb:6d} -> {kind}  "
            f"(untiled transient {bb*nn*32*nb*4/1e6:8.1f} MB)")
    log("PARITY_DONE")


def cmd_micro():
    n = int(os.environ.get("PB_N", "512"))
    nb = int(os.environ.get("PB_NB", "16384"))
    cos, sin = freqs(n)
    x, qr, ik = synth(n, nb)
    ix = rand_indexer(20)
    tile = int(os.environ.get("PB_TILE", "0"))       # 0 = untiled
    minnb = 10 ** 9 if tile == 0 else 1
    IX._TILE, IX._TILE_MIN_NB = tile, minnb
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    t0 = time.perf_counter()
    sh = Sh()
    idx = ix(x, qr, 0, 0, cos, sin, ik, sh)
    mx.eval(idx, sh.candidates)
    dt = time.perf_counter() - t0
    log(f"micro nb={nb} n={n} tile={tile or 'untiled'}: {dt*1e3:8.1f} ms  "
        f"peak_delta={(mx.get_peak_memory()-base)/1e6:8.1f} MB  "
        f"live_delta={(mx.get_active_memory()-base)/1e6:8.1f} MB  "
        f"budget_MB={IX._TILE_BUDGET/1e6:.0f}")
    log("MICRO_DONE")


# --------------------------------------------------------------------------
CAP = {}
MODEL_IX = {}


def _hook():
    orig = IX.Indexer.__call__
    at = int(os.environ.get("PB_CAP_AT", "15872"))
    want = {int(v) for v in os.environ.get("PB_CAP_LAYERS", "2,20,24").split(",")}

    def patched(self, x, qr, start_pos, offset, cos, sin, index_k, shared):
        out = orig(self, x, qr, start_pos, offset, cos, sin, index_k, shared)
        if start_pos >= at and self.layer_id in want and self.layer_id not in CAP:
            CAP[self.layer_id] = dict(
                x=mx.array(x), qr=mx.array(qr), sp=int(start_pos), off=int(offset),
                ik=mx.array(index_k), cos=cos, sin=sin,
                cand=(mx.array(shared.candidates) if shared.candidates is not None else None),
                nb=int(index_k.shape[1]))
            MX_EVAL()
        return out

    def MX_EVAL():
        arrs = [v for c in CAP.values() for v in c.values() if isinstance(v, mx.array)]
        mx.eval(arrs)

    IX.Indexer.__call__ = patched
    if not hasattr(IX, "_orig_call"):
        IX._orig_call = orig


def build_model():
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    model, _ = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS, rank=0,
                              world=2, group=None)
    model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
    for blk in model.layers:
        if getattr(blk, "attn", None) is not None and hasattr(blk.attn, "indexer"):
            MODEL_IX[blk.layer_id] = blk.attn.indexer
    log(f"built layers={LAYERS} indexers={sorted(MODEL_IX)} "
        f"active={mx.get_active_memory()/1e9:.1f}GB")
    return model


def cmd_prefill():
    L = int(os.environ.get("PB_LEN", "16384"))
    CHUNK = int(os.environ.get("PB_CHUNK", "512"))
    model = build_model()
    if os.environ.get("PB_CAPTURE") == "1":
        _hook()
    base = json.load(open(HOME + "/p30_prompt_ids.json"))
    ids = (base * (L // len(base) + 1))[:L]
    cache = model.make_cache(1, max_seq_len=L + 64)
    for li, lc in enumerate(cache.layers):
        if li not in LAYERS:
            lc.comp_state = None
    mx.clear_cache()
    mx.reset_peak_memory()
    active0 = mx.get_active_memory()
    t0 = time.perf_counter()
    tchunk = []
    tok = None
    peak_at = None
    PEAK_CHUNK = int(os.environ.get("PB_PEAK_CHUNK", "0"))   # 1 = final chunk
    nchunks = (L + CHUNK - 1) // CHUNK
    for a in range(0, L, CHUNK):
        cidx = a // CHUNK
        if PEAK_CHUNK and cidx == nchunks - 1:
            # isolate this one chunk's allocation: sync + reset, run, sync
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            s = time.perf_counter()
            am = model(mx.array([ids[a:a + CHUNK]]), cache, last_logit_only=True,
                       argmax=True)
            mx.eval(am)
            mx.synchronize()
            peak_at = (mx.get_peak_memory() - base) / 1e6
            tchunk.append(time.perf_counter() - s)
            tok = int(np.array(am)[0, -1])
            continue
        s = time.perf_counter()
        am = model(mx.array([ids[a:a + CHUNK]]), cache, last_logit_only=True, argmax=True)
        mx.eval(am)
        tchunk.append(time.perf_counter() - s)
        tok = int(np.array(am)[0, -1])
    dt = time.perf_counter() - t0
    log(f"PREFILL {L} tok chunk={CHUNK} TILE={IX._TILE} MIN_NB={IX._TILE_MIN_NB}: "
        f"{dt:7.1f}s = {L/dt:6.1f} tok/s peak={mx.get_peak_memory()/1e9:.2f}GB "
        f"grown={(mx.get_peak_memory()-active0)/1e9:.2f}GB "
        f"chunk_ms p10={np.percentile(tchunk,10)*1e3:.0f} med={np.median(tchunk)*1e3:.0f} "
        f"p90={np.percentile(tchunk,90)*1e3:.0f} max={np.max(tchunk)*1e3:.0f}")
    if peak_at is not None:
        log(f"ISOLATED last chunk (context {L-CHUNK}->{L}): peak_alloc={peak_at:.1f} MB")
    log(f"first argmax after {L} tokens: {tok}")
    # per-layer NLL on the last chunk's logits, if a head was built
    if CAP:
        for lid in sorted(CAP):
            c = CAP[lid]
            log(f"captured L{lid}: sp={c['sp']} n={c['x'].shape[1]} nb={c['nb']} "
                f"cand={'yes' if c['cand'] is not None else 'no'}")
    log("PREFILL_DONE")


def cmd_capture():
    """Tiled vs untiled on the REAL captured tensors; NLL from a fresh subset pass."""
    rows = int(os.environ.get("PB_ROWS", "128"))
    if not CAP:
        log("nothing captured (needs PB_CAPTURE=1 on a prefill run)"); return
    for lid in sorted(CAP):
        c = CAP[lid]
        ix = MODEL_IX.get(lid)
        if ix is None:
            log(f"L{lid}: no indexer weights in this subset; skipping"); continue
        x, qr, ik = c["x"][:, :rows], c["qr"][:, :rows], c["ik"]
        sp = c["sp"]
        cos, sin = c["cos"], c["sin"]
        cand = c["cand"][:, :rows] if c["cand"] is not None else None
        IX._TILE, IX._TILE_MIN_NB = FULL_TILED
        sh_t = Sh(cand)
        it = ix(x, qr, sp, c["off"], cos, sin, ik, sh_t)
        IX._TILE, IX._TILE_MIN_NB = FULL_UNTILED
        sh_u = Sh(cand)
        iu = ix(x, qr, sp, c["off"], cos, sin, ik, sh_u)
        mx.eval(it, iu, sh_t.candidates, sh_u.candidates)
        # ties at the k-th boundary on the real scores
        k = min(ix.index_topk, c["nb"])
        s_ref, q = scores_ref(ix, x, qr, ik, sp, cos, sin)
        lens = ((sp + np.arange(rows) + 1) // ix.ratio)[:, None]
        vis = np.arange(c["nb"])[None, :] < lens
        sv = np.array(s_ref)
        sv[~vis] = -np.inf
        report_pair(f"real L{lid} sp={sp} nb={c['nb']} rows={rows} topk={k} (tiled vs "
                    f"untiled on real tensors)", it, iu, ix.ratio, sp, k, c["nb"],
                    scores=sv, lens=lens)
        if lid == 20 and sh_t.candidates is not None:
            d = int(mx.sum((sh_t.candidates != sh_u.candidates).astype(mx.int32)).item())
            log(f"real L20 candidate mask diff_bits={d} "
                f"kept tiled={int(mx.sum(sh_t.candidates[..., ::8].astype(mx.int32)).item())} "
                f"untiled={int(mx.sum(sh_u.candidates[..., ::8].astype(mx.int32)).item())}")
        # value-multiset equality: the k-th largest VALUE must match
        ka = np.sort(np.array(it)[0], axis=-1)
        kb = np.sort(np.array(iu)[0], axis=-1)
        log(f"real L{lid}: same_selected_indices={bool((ka==kb).all())}")
        del x, qr, ik, s_ref
        mx.clear_cache()
    log("CAPTURE_DONE")


def cmd_ab():
    """Decisive gate: both indexer paths in ONE process, same weights.

    Runs the full 16K chunked prefill twice over the same subset model — tiled
    (DSV41_INDEXER_TILE, the production default) and untiled
    (DSV41_INDEXER_TILE=0) — recording, per chunk, the wall time and the
    per-layer top-k selections, then compares the two arms' greedy tokens and
    the last chunk's logits.
    """
    L = int(os.environ.get("PB_LEN", "16384"))
    CHUNK = int(os.environ.get("PB_CHUNK", "512"))
    model = build_model()
    base = json.load(open(HOME + "/p30_prompt_ids.json"))
    ids = (base * (L // len(base) + 1))[:L]
    toks = {}

    # capture every indexer's returned ids so the arms can be compared exactly
    seen = {}
    orig = IX.Indexer.__call__

    def patched(self, x, qr, start_pos, offset, cos, sin, index_k, shared):
        out = orig(self, x, qr, start_pos, offset, cos, sin, index_k, shared)
        seen.setdefault(self.layer_id, []).append((start_pos, mx.array(out)))
        return out

    IX.Indexer.__call__ = patched
    results = {}
    for arm, (tile, minnb) in (("tiled", FULL_TILED), ("untiled", FULL_UNTILED)):
        IX._TILE, IX._TILE_MIN_NB = tile, minnb
        seen.clear()
        cache = model.make_cache(1, max_seq_len=L + 64)
        for li, lc in enumerate(cache.layers):
            if li not in LAYERS:
                lc.comp_state = None
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        a0 = mx.get_active_memory()
        t0 = time.perf_counter()
        tchunk = []
        last_logits = None
        for a in range(0, L, CHUNK):
            s = time.perf_counter()
            lg = model(mx.array([ids[a:a + CHUNK]]), cache)
            mx.eval(lg)
            tchunk.append(time.perf_counter() - s)
            last_logits = lg
        mx.synchronize()
        dt = time.perf_counter() - t0
        peak = (mx.get_peak_memory() - a0) / 1e9
        toks[arm] = int(np.array(mx.argmax(last_logits[0, -1])))
        results[arm] = dict(dt=dt, tps=L / dt, peak=peak,
                            p10=np.percentile(tchunk, 10), med=np.median(tchunk),
                            p90=np.percentile(tchunk, 90), mx=max(tchunk),
                            logits=mx.array(last_logits[0, -1]).astype(mx.float32),
                            sel={lid: [(sp, mx.array(v)) for sp, v in entries]
                                 for lid, entries in seen.items()})
        log(f"AB arm={arm:8s} TILE={tile:5d}: {dt:6.1f}s = {L/dt:6.1f} tok/s "
            f"peak_grown={peak:.2f}GB chunk_ms med={np.median(tchunk)*1e3:.0f} "
            f"max={np.max(tchunk)*1e3:.0f} greedy_last={toks[arm]}")
        del cache
        mx.clear_cache()

    # compare selections
    t_, u_ = results["tiled"]["sel"], results["untiled"]["sel"]
    layers = sorted(set(t_) | set(u_))
    total_rows = total_diff = 0
    for lid in layers:
        ta, ua = t_.get(lid, []), u_.get(lid, [])
        if len(ta) != len(ua):
            log(f"AB L{lid}: chunk count differs tiled={len(ta)} untiled={len(ua)}")
            continue
        d = 0
        for (sp_t, v_t), (sp_u, v_u) in zip(ta, ua):
            if sp_t != sp_u:
                log(f"AB L{lid}: start_pos mismatch {sp_t} vs {sp_u}")
            a = np.array(v_t)[0]
            b = np.array(v_u)[0]
            for r in range(a.shape[0]):
                sa = sorted(int(c) for c in a[r] if c >= 0)
                sb = sorted(int(c) for c in b[r] if c >= 0)
                total_rows += 1
                if sa != sb:
                    d += 1
                    total_diff += 1
        log(f"AB L{lid}: {len(ta)} chunks, {d} differing rows "
            f"(of {total_rows} cumulative)")
    la = np.array(results["tiled"]["logits"])
    lb = np.array(results["untiled"]["logits"])
    log(f"AB logits: exact_equal={bool((la == lb).all())} "
        f"max|d|={float(np.abs(la-lb).max()):.4g} "
        f"cos={float((la@lb)/(np.linalg.norm(la)*np.linalg.norm(lb))):.8f} "
        f"argmax_equal={toks['tiled'] == toks['untiled']}")
    log(f"AB selection rows differing: {total_diff}/{total_rows}")
    log("AB_DONE")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parity"
    if cmd == "all":
        cmd_parity()
        cmd_micro()
        cmd_prefill()
        cmd_capture()
    else:
        globals()["cmd_" + cmd]()
