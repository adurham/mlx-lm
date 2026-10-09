# Copyright © 2026 Adam Durham (hermes-gw)
"""Replay the REAL-tensor capture (next18_functional.npz) through L2-full.

The capture (bench/next18_capture.py on a live boot) stores, per indexer call,
the DERIVED score inputs q/index_k/w/lens and the module config, at decode
(n=1) and verify (n=4). We replay each captured call through:
  * fallback fp32 row  (L2-full)     -> topk_from_row
  * fallback bf16 row  (shipped)     -> topk_from_row
  * hierarchical (shipped HIER, overfetch=16)
and diff each against the OTHER, so we can state the real-tensor behaviour of
the design. Only calls whose candidate mask was captured (source/plain roles)
are exactly replayable; consumer calls need shared.candidates, which the capture
does not store, so they are flagged and skipped for the exact diff.

Run: PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_replay_capture.py \
        /private/tmp/next18_functional.npz
"""
from __future__ import annotations
import json
import sys
import numpy as np
import mlx.core as mx
from mlx_lm.models.deepseek_v41 import indexer as I
from mlx_lm.models.deepseek_v41 import indexer_hierarchical as H


def bf16(a):
    return mx.array(a).view(mx.bfloat16) if a.dtype == np.uint16 else mx.array(a)


def row_scores(q, ik, w, lens, dtype):
    s = mx.einsum("bshd,btd->bsht", q, ik.astype(mx.float32))
    s = mx.maximum(s, 0.0) * w[..., None]
    row = mx.sum(s, axis=2).astype(dtype)
    vis = mx.arange(ik.shape[1])[None, :] < lens
    row = mx.where(vis[None], row, float("-inf"))
    mx.eval(row)
    return row


def main(path):
    # Our OWN capture harness (bench/next18_capture.py) wrote this npz via
    # np.savez_compressed with only ndarray values; allow_pickle=True is set
    # purely for robustness against the harness's scalar-array encoding and is
    # safe here because the file is a trusted campaign artifact, not user input.
    d = np.load(path, allow_pickle=True)
    nrec = int(d["meta_nrec"])
    print(f"# capture {path}: {nrec} records, seen={int(d['meta_seen'])}", flush=True)
    out = []
    for i in range(nrec):
        p = f"c{i:04d}_"
        n = int(d[p + "n"]); nb = int(d[p + "nb"]); k = int(d[p + "k"])
        blk = int(d[p + "hier_block"]); layer = int(d[p + "layer_id"])
        src = bool(d[p + "is_candidate_source"]); uses = bool(d[p + "uses_candidates"])
        q = mx.array(d[p + "q"]); w = mx.array(d[p + "w"])
        lens = mx.array(d[p + "lens"]).astype(mx.int32)
        ik = bf16(d[p + "index_k"])
        cap_out = d[p + "out"]
        ab_ndiff = int(d[p + "ab_ndiff"])

        if uses:
            out.append(dict(i=i, layer=layer, n=n, nb=nb, k=k, role="consumer",
                            note="needs shared.candidates (not captured)",
                            cap_ab_ndiff=ab_ndiff, replayable=False))
            print("  " + json.dumps(out[-1]), flush=True)
            continue

        # source/plain: replayable. Build the fallback row (L2-full fp32 and
        # shipped bf16) and the HIER path; compare index tensors.
        row32 = row_scores(q, ik, w, lens, mx.float32)
        rowbf = row_scores(q, ik, w, lens, mx.bfloat16)
        v32, i32 = I.topk_from_row(row32, k)
        vbf, ibf = I.topk_from_row(rowbf, k)
        hv, hi, _ = H.hierarchical_topk_prod(q, ik, w, lens, k, blk, 4096, 4096, 16)
        mx.eval(v32, i32, vbf, ibf, hv, hi)
        a = np.array(i32); b = np.array(ibf); c = np.array(hi)
        out.append(dict(i=i, layer=layer, n=n, nb=nb, k=k,
                        role=("source" if src else "plain"),
                        cap_ab_ndiff=ab_ndiff, replayable=True,
                        L2full_vs_hier_ndiff=int((a != c).sum()),
                        bf16row_vs_hier_ndiff=int((b != c).sum()),
                        slots=int(a.size)))
        print("  " + json.dumps(out[-1]), flush=True)

    rep = [r for r in out if r.get("replayable")]
    print("REPLAY_CAPTURE " + json.dumps({
        "records": nrec, "replayable": len(rep),
        "L2full_vs_hier_total_ndiff": sum(r["L2full_vs_hier_ndiff"] for r in rep),
        "bf16row_vs_hier_total_ndiff": sum(r["bf16row_vs_hier_ndiff"] for r in rep),
        "slots": sum(r["slots"] for r in rep)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/next18_functional.npz")
