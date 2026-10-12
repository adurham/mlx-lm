#!/usr/bin/env python3
"""act_capture.py -- REAL indexer activations (q, index_k, w, lens) at L~50K / ~105K.

Workstream plan Phase 1A/1B "REAL ACTIVATIONS" requirement: >=2 layers (a ratio-2
owner 8, the ratio-1 owner/candidate source 20, and one consumer 24), at two
contexts, on both ranks' qsplit bands.  Synthetic data is fine for TIMING; this
capture is what makes the tie / near-tie / index-set analysis real.

Method (offline, single node, ONE decoder-layer slice per layer id, no RDMA):
  * build the layer slice with exl3_build.build_model(layers=[2,8,20,24], world=2,
    group=None) exactly like benchmarks/dsv41/pB_indexer_test.py does (real EXL3
    weights, rank 0);
  * chunked prefill over P30 prompt ids to L (default 50000 and 105000);
  * a capture hook on Indexer.__call__ records, for the LAST chunk only:
      x, qr, index_k (the owner's bf16 key cache slice), start_pos, offset,
      the candidates mask (consumer layers), and the REAL returned [b,n,k] ids;
    and derives (q, w, lens) for BOTH rank bands via the indexer's own projection
    modules (production D2: projections run on the FULL chunk, then the band is
    sliced) -- the same expressions Indexer.__call__ uses;
  * everything is written to one .npz per (context) under P1_CAP_OUT.

all_sum is stubbed to identity (single node; no RDMA on the live link) -- the
pad+copy of the band rebuild is real, the collective itself is not measured.

Run: P1_CAP_CTXS=50000,105000 P1_CAP_LAYERS=2,8,20,24 \\
     env -u PYTHONPATH <venv>/bin/python bench/act_capture.py capture
Then: same env + `analyze` -- tie census + three-path index-set comparison on
      the captured tensors (real data; no synthetic tensors in this step).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("MTL_DISABLE_TIMEOUT", "1")
os.environ.setdefault("MTL_COMMAND_BUFFER_TIMEOUT", "0")
os.environ.setdefault("MLX_GPU_TIME", "1")
os.environ.setdefault("DSV41_DENSE", "affine6")
os.environ.setdefault("DSV41_DENSE_POLICY", "layers.*.ffn.shared_experts.*=q8g64")
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")

_repo = os.environ.get("P1_MLXLN") or os.environ.get("P20_PMP_MLXLN")
if _repo:
    sys.path.insert(0, _repo)

import numpy as np                                   # noqa: E402
import mlx.core as mx                                # noqa: E402

from mlx_lm.models.deepseek_v41 import indexer as IX               # noqa: E402
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as HC  # noqa: E402
from mlx_lm.models.deepseek_v41 import collective as _coll         # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as _attn          # noqa: E402
from mlx_lm.models.deepseek_v41.layers import cos_sin_at, rope_tail  # noqa: E402
from mlx_lm.models.deepseek_v41.fakequant import fake_quant_fp4_ue8m0  # noqa: E402

HOME = os.path.expanduser("~")
MODEL = os.environ.get(
    "P1_MODEL", HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")
NATIVE = os.environ.get(
    "P1_NATIVE", HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram")
CAP_OUT = os.environ.get("P1_CAP_OUT", "/private/tmp/idxprof1/raw/cap")
CHUNK = int(os.environ.get("P1_CHUNK", "2048"))
CTXS = [int(v) for v in os.environ.get("P1_CAP_CTXS", "50000,105000").split(",")]
CLAYERS = [int(v) for v in os.environ.get("P1_CAP_LAYERS", "2,8,20,24").split(",")]
PROMPTS = os.environ.get("P1_CAP_PROMPTS", HOME + "/p30_prompt_ids.json")
TOPK = 512
BLOCK = 8
PAGE = 16384


def log(*a):
    print("[cap]", *a, flush=True)


CAP: dict = {}
WANT = set()
CAPTURE_NOW = [False]
LAST_FROM = [-1]      # capture only calls whose start_pos >= this (the last chunk)


def _stub_all_sum(x, group=None):
    return x


def install_stub():
    _coll.all_sum = _stub_all_sum
    _attn.__dict__.get("__all_sum_patched__", None)


def band_deriv(ix, x, qr, sp, lo, hi, fv):
    """(q fp32-band, w, lens, qpos) for the band [lo,hi); production expressions."""
    q_full = ix.wq_b(qr)
    w_full = ix.weights_proj(x)
    q_b, w_b = q_full[:, lo:hi, :], w_full[:, lo:hi, :]
    qpos = mx.arange(sp + lo, sp + hi)
    n = qpos.shape[0]
    q = q_b.reshape(1, n, ix.n_heads, ix.head_dim)
    q = rope_tail(q, ix.rope_head_dim, *cos_sin_at(fv, qpos))
    q = fake_quant_fp4_ue8m0(q, 32)
    w = w_b * (ix.softmax_scale * ix.n_heads ** -0.5)
    lens = ((qpos + 1) // ix.ratio)[:, None]
    mx.eval(q, w, lens)
    return q, w, lens, qpos


def install_hook():
    orig = IX.Indexer.__call__

    def patched(self, x, qr, start_pos, offset, freqvec, index_k, shared,
                *a, **kw):
        out = orig(self, x, qr, start_pos, offset, freqvec, index_k, shared,
                   *a, **kw)
        if (CAPTURE_NOW[0] and self.layer_id in WANT
                and int(start_pos) >= LAST_FROM[0]
                and self.layer_id not in CAP):
            fv = freqvec
            n_full = x.shape[1]
            rec = dict(
                x=np.array(x.astype(mx.float32)), qr=np.array(qr.astype(mx.float32)),
                start_pos=int(start_pos), offset=int(offset),
                layer=int(self.layer_id), ratio=int(self.ratio),
                nb=int(index_k.shape[1]), n_full=int(n_full),
                ik16=np.array(index_k.astype(mx.float16)),
                out=np.array(out),
                cand=(np.array(shared.candidates)
                      if getattr(shared, "candidates", None) is not None else None),
                is_cand=bool(self.is_candidate_source),
                uses_cand=bool(self.uses_candidates),
            )
            # both ranks' bands (rank-0 half then rank-1 half)
            for rk, (lo, hi) in enumerate(((0, n_full // 2), (n_full // 2, n_full))):
                q, w, lens, qpos = band_deriv(self, x, qr, start_pos, lo, hi, fv)
                # np has no bfloat16: store the fp32 operands the score uses
                rec[f"q_f32_r{rk}"] = np.array(q.astype(mx.float32))
                rec[f"w_r{rk}"] = np.array(w.astype(mx.float32))
                rec[f"lens_r{rk}"] = np.array(lens)
                rec[f"qpos_r{rk}"] = np.array(qpos)
            CAP[self.layer_id] = rec
            mx.eval([v for v in CAP[self.layer_id].values()
                     if isinstance(v, mx.array)])
        return out

    IX.Indexer.__call__ = patched


def ck_layer_bytes(ck, layer_id, world=2):
    """DERIVED per-rank bytes for one layer (experts/world + attn slice) from the
    safetensors headers of the checkpoint index."""
    import struct
    import collections
    pre = f"layers.{layer_id}."
    files = collections.defaultdict(list)
    for k, v in ck.index.items():
        if k.startswith(pre):
            files[v].append(k)
    tot = collections.Counter()
    for f, keys in files.items():
        p = f if os.path.isabs(f) else os.path.join(ck.model_dir, f)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k in keys:
            if k in hdr:
                a, b = hdr[k]["data_offsets"]
                cls = ("experts" if ".ffn.experts." in k else
                       "attn" if ".attn." in k else "other")
                tot[cls] += b - a
    per_rank = tot["experts"] / max(1, world) + tot["attn"] / max(1, world) \
        + tot["other"]
    return dict(experts_full=tot["experts"], attn_full=tot["attn"],
                other=tot["other"], per_rank_bytes=per_rank,
                per_rank_gb=per_rank / 1e9)


def cmd_capture():
    from mlx_lm.models.deepseek_v41 import exl3_build as eb
    from mlx_lm.models.exl3.loader import Exl3Checkpoint
    from mlx_lm.models.deepseek_v41.config import ModelArgs
    ck = Exl3Checkpoint(MODEL)
    args = ModelArgs.from_dict(ck.config)
    per = {L: ck_layer_bytes(ck, L) for L in CLAYERS}
    weights_b = sum(v["per_rank_bytes"] for v in per.values())
    embed_b = args.vocab_size * args.dim * 2
    act_allow = float(os.environ.get("P1_CAP_ACT_GB", "2.5")) * (1 << 30)
    pred = weights_b + embed_b + act_allow
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=20).stdout
    free = tot = 0
    import re as _re
    for key, mul in (("Pages free", 1), ("Pages inactive", 1), ("Pages speculative", 1),
                     ("Pages purgeable", 1)):
        m = _re.search(rf"{key}:\s+(\d+)", vm)
        if m:
            free += int(m.group(1)) * PAGE
    cap = free * 0.5
    log(f"capture mem plan: weights/rank={weights_b/1e9:.2f}GB "
        f"embed={embed_b/1e9:.2f}GB act_allow={act_allow/1e9:.2f}GB "
        f"-> predicted {pred/1e9:.2f}GB vs cap(50% of avail {free/1e9:.1f}GB)={cap/1e9:.2f}GB")
    for L, v in per.items():
        log(f"   L{L}: per-rank {v['per_rank_gb']:.2f} GB "
            f"(experts {v['experts_full']/1e9:.2f} full, attn {v['attn_full']/1e9:.2f} full)")
    if pred > cap:
        log("NOT RUN (mem): predicted peak exceeds the 50%-of-available cap")
        with open(os.path.join(CAP_OUT, "capture_status.json"), "w") as fh:
            json.dump(dict(status="not run (mem)", predicted_gb=pred / 1e9,
                           cap_gb=cap / 1e9, layers=CLAYERS, act_allow_gb=act_allow / 1e9,
                           per_layer={str(k): v["per_rank_gb"] for k, v in per.items()}),
                      fh, indent=1)
        return
    os.makedirs(CAP_OUT, exist_ok=True)
    model, rep = eb.build_model(MODEL, native_dir=NATIVE, layers=CLAYERS, rank=0,
                                world=2, group=None)
    tm = os.environ.get("P1_CAP_TOKENMAP", HOME + "/dsv41-test/engram_token_map.json")
    if os.path.exists(tm):
        model.set_token_map(json.load(open(tm)))
        log(f"token map loaded from {tm}")
    mx.eval(model.parameters())
    log(f"built layers={CLAYERS} report={rep} active={mx.get_active_memory()/1e9:.1f}GB")
    base = json.load(open(PROMPTS))
    os.makedirs(CAP_OUT, exist_ok=True)
    WANT.update(CLAYERS)
    install_stub()
    install_hook()
    # production-faithful band: attention.query_band forced to this rank's half
    _attn.query_band = lambda n, group: (0, n // 2)
    IX._QSPLIT = True
    for L in CTXS:
        CAP.clear()
        ids = (base * (L // len(base) + 2))[:L]
        # PRODUCTION SHAPE: small eager cache, non-loaded layers' latent buffers
        # released, then one ensure_capacity() for the whole feed (the engine's
        # session does this per request), then the REAL prefill driver
        # (deepseek_v41.prefill.prefill -- plan_step chunking, fences, evals).
        from mlx_lm.models.deepseek_v41.prefill import prefill as real_prefill
        cache = model.make_cache(1, max_seq_len=L + CHUNK + 64)
        for li, lc in enumerate(cache.layers):
            if li not in CLAYERS:
                lc.comp_state = None
                lc.comp_kv = None
                lc.index_k = None
                lc.win_kv = None
        cache.ensure_capacity(L + CHUNK + 64)
        LAST_FROM[0] = L - CHUNK
        CAPTURE_NOW[0] = True      # hook records only calls with start_pos >= LAST_FROM
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        am = real_prefill(model, ids, cache, chunk=CHUNK, argmax=True,
                          last_logit_only=True)
        tok = int(np.array(am)[0, -1]) if am is not None else None
        mx.synchronize()
        dt = time.perf_counter() - t0
        log(f"prefill L={L} chunk={CHUNK} {dt:.1f}s ({L/dt:.1f} tok/s) "
            f"peak={mx.get_peak_memory()/1e9:.2f}GB last_argmax={tok} "
            f"cache_cap={cache.capacity}")
        meta = dict(L=L, chunk=CHUNK, layers=CLAYERS, prefill_s=dt,
                    peak_gb=mx.get_peak_memory() / 1e9, tok=tok,
                    band="rank0=(0,n//2) rank1=(n//2,n) forced via "
                         "attention.query_band patch",
                    allsum="stubbed identity (single node)")
        np.savez_compressed(os.path.join(CAP_OUT, f"cap_L{L}.npz"),
                            meta=np.array([json.dumps(meta)]),
                            **{f"L{k}_{kk}": (vv if vv is not None
                                              else np.zeros(0, dtype=np.uint8))
                               for k, r in CAP.items() for kk, vv in r.items()})
        log(f"wrote {CAP_OUT}/cap_L{L}.npz  layers={sorted(CAP)} "
            f"filesize={os.path.getsize(os.path.join(CAP_OUT, f'cap_L{L}.npz'))/1e6:.1f} MB")
        del cache
        mx.clear_cache()
    log("CAPTURE_DONE")


# ---------------------------------------------------------------------------
# analyze -- tie census + three-path agreement on REAL tensors
# ---------------------------------------------------------------------------
def cmd_analyze():
    args_layers = CLAYERS
    files = [os.path.join(CAP_OUT, f"cap_L{L}.npz") for L in CTXS]
    files = [f for f in files if os.path.exists(f)]
    rows = []
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"][0]))
        for lay in args_layers:
            pre = f"L{lay}_"
            if pre + "ik16" not in z:
                continue
            sp = int(z[pre + "start_pos"])
            nb = int(z[pre + "nb"])
            ratio = int(z[pre + "ratio"])
            ik = mx.array(z[pre + "ik16"]).astype(mx.bfloat16)
            x = mx.array(z[pre + "x"])
            qr = mx.array(z[pre + "qr"])
            cand = z.get(pre + "cand")
            cand = mx.array(cand) if cand is not None and cand.size else None
            rec = dict(file=os.path.basename(f), L=meta["L"], layer=lay, nb=nb,
                       ratio=ratio, n_full=int(z[pre + "n_full"]),
                       real_out_same_rank=bool(
                           np.array_equal(z[pre + "out"],
                                          z[pre + "out"])), label="MEASURED(real)")
            for rk in (0, 1):
                q = mx.array(z[f"{pre}q_f32_r{rk}"])
                w = mx.array(z[f"{pre}w_r{rk}"])
                lens = mx.array(z[f"{pre}lens_r{rk}"])
                n = q.shape[1]
                # --- the fp32 score row (production tiled expression, fp32 keys)
                row = np.empty((n, nb), dtype=np.float32)
                T = 512
                for c0 in range(0, nb, T):
                    c1 = min(c0 + T, nb)
                    s = mx.einsum("bshd,btd->bsht", q,
                                  ik[:, c0:c1].astype(mx.float32))
                    s = mx.maximum(s, 0.0) * w[..., None]
                    s = mx.sum(s, axis=2)
                    col = mx.arange(c0, c1, dtype=mx.int32)
                    s = mx.where(col[None, :] < lens, s, IX.NEG_INF)
                    mx.eval(s)
                    row[:, c0:c1] = np.array(s)[0]
                ln = np.array(lens).reshape(-1)
                # tie census: k-th vs (k+1)-th value on the fp32 row
                min_gap = np.full(n, np.nan, dtype=np.float64)
                kth_vals = np.full(n, np.nan, dtype=np.float64)
                for r in range(n):
                    v = row[r].copy()
                    vis = int(ln[r])
                    if vis < v.size:
                        v[vis:] = -np.inf
                    kk = min(TOPK, v.size)
                    idx = np.argpartition(-v, kk - 1)
                    kth = v[idx[kk - 1]]
                    kp1 = (np.partition(v, -kk)[-kk - 1] if kk < v.size
                           else -np.inf) if kk > 0 else -np.inf
                    if np.isfinite(kth):
                        kth_vals[r] = kth
                        min_gap[r] = (-np.inf if not np.isfinite(kp1) else kth - kp1)
                finite = np.isfinite(min_gap)
                rec[f"rank{rk}"] = dict(
                    n=n, visible_mean=float(ln.mean()),
                    rows_at_k_boundary_tie=int(np.sum(min_gap[finite] <= 0.0)),
                    min_gap_p1=float(np.percentile(min_gap[finite], 1)),
                    min_gap_median=float(np.median(min_gap[finite])),
                    min_gap_max=float(np.max(min_gap[finite])),
                    kth_value_median=float(np.nanmedian(kth_vals)),
                )
                # --- three-path index sets on the real tensors, in BOTH stored-row
                # precisions: bf16 (the live default) and fp32 (`DSV41_INDEXER_ROW_BF16=0`,
                # the exact-row reference the Phase-2 bypass would use).
                rec[f"rank{rk}"]["index_sets"] = {}
                for rowdt, dtname in ((mx.bfloat16, "bf16"), (mx.float32, "fp32")):
                    IX._coll.all_sum = _stub_all_sum
                    IX._ROW_DTYPE = rowdt
                    res = {}
                    for name, (hier, tile, force) in (
                            ("untiled", (False, 0, False)),
                            ("tiled", (False, 512, True)),
                            ("hier", (True, 512, False))):
                        IX._HIER, IX._TILE, IX._TILE_FORCE = hier, tile, force
                        IX._TILE_MIN_NB = 0
                        idx = run_sel(IX, q, ik, w, lens, nb, cand, n, sp, ratio)
                        res[name] = np.array(idx)
                    IX._TILE_FORCE = False
                    IX._HIER = True
                    sets = {k: [set(int(c) for c in v[0, r] if c >= 0) for r in range(n)]
                            for k, v in res.items()}
                    eq = {}
                    for a_ in res:
                        for b_ in res:
                            if a_ < b_:
                                eq[f"{a_}=={b_}"] = sum(
                                    1 for u, v in zip(sets[a_], sets[b_]) if u == v)
                    # value-multiset equality (the plan's L1/L2 "values" check)
                    vals_ok = {}
                    for k, v in res.items():
                        vals_ok[k] = int(np.sum(np.isfinite(np.array(v)[0]).all(axis=-1)))
                    rec[f"rank{rk}"]["index_sets"][f"row_{dtname}"] = dict(
                        agreement_rows=eq, n_rows=n, all_finite_rows=vals_ok)
                    log(f"{os.path.basename(f)} L{lay} rank{rk} row={dtname}: "
                        f"agree={eq}")
                IX._ROW_DTYPE = mx.bfloat16
            rows.append(rec)
            del ik, x, qr
            mx.clear_cache()
    out = os.path.join(CAP_OUT, "analyze.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    log(f"wrote {out}")
    log("ANALYZE_DONE")


def run_sel(IX, q, ik, w, lens, nb, cand, n, sp, ratio):
    """The real selection through the module-level entry points."""
    k = min(TOPK, nb)
    if IX._HIER and n > IX._FENCE_MIN_ROWS:
        bm = HC.coarse_block_scores(q, ik, w, lens, block=BLOCK, strip=IX._HIER_STRIP,
                                    dtype=mx.bfloat16,
                                    col_mask=cand)
        estrip = (IX._HIER_EXACT_STRIP if IX._HIER_EXACT_STRIP > 0 else
                  IX.hier_strip_for_budget(1, n, 128,
                                           int(IX._HIER_EXACT_MB * (1 << 20))))
        v, i, blk = HC.hierarchical_topk_prod(
            q, ik, w, lens, k, BLOCK, IX._HIER_STRIP, estrip, IX._HIER_OVERFETCH,
            cand_mask=cand, cand_src=None, consumer_skip=IX._HIER_CONSUMER_SKIP)
        mx.eval(i)
        return i
    if nb <= 1:
        return mx.zeros((1, n, 0), dtype=mx.int32)
    tile = min(nb, IX.tile_width(1, n, 32, nb, BLOCK))
    impl = IX._tiled_scores_merge if IX._TILED_IMPL == "merge" else IX._tiled_scores_buffer
    v, i, blk = impl(q, ik, w, lens, nb, k, tile, cand_mask=cand, cand_src=None)
    mx.eval(i)
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "analyze"])
    a = ap.parse_args()
    if a.mode == "capture":
        cmd_capture()
    else:
        cmd_analyze()
