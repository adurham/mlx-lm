# Stream C -- DSv4.1 prefill orchestration (fences, adaptive chunks, warmup)

Branch `ws/C`, commit `722b974`. Single-node tests on m4-1 and m4-2, each under
`lockf -k ~/dsv41-gpu.lock`, env `EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1`.

## Files

- NEW `mlx_lm/models/deepseek_v41/prefill.py` -- `prefill(model, ids, cache, ...)`
  + `warmup(model)` + `plan_step(...)`. API documented at the top of the file.
- `mlx_lm/models/deepseek_v41/model.py` -- one hook: `Model._fence_every`
  (default 0) and a per-K-layer `mx.eval(h, pre_mix)` inside `Model.__call__`
  when `n > 1`. 12 lines total; single-row forwards are never fenced.
- Harness: `scripts/pC_prefill.py` (modes smoke/cold/warm/timing/attr/fence/
  p48rows/api).

## Warmup (the 206 s first-chunk compile spike)

m4-1, 8-layer p48 subset, 1536 tokens through `prefill()`:

| | first-chunk | c1 | c2 | c3 | total | peak |
|---|---:|---:|---:|---:|---:|---:|
| cold (no warmup) | 1.73 s | 1.73 | 2.50 | 3.19 | 3.2 s (482 tok/s) | 25.42 GB |
| `warmup()` first | 0.90 s | 0.90 | 1.61 | 2.30 | 2.3 s (668 tok/s) | 23.74 GB |
| steady-state repeat | 0.70 s | -- | -- | -- | 2.0 s (751 tok/s) | -- |

Warmup cost itself: 2.2 s total (512-row 1.7 s, second 512-row 0.4 s, 128-row
0.1 s, decode 0.0 s). Absorbs the compile cost on the first user-visible chunk
(c1 1.73 -> 0.90 s) and drops the first-chunk peak by 1.7 GB. The subset's
absolute number is small; the full 40-layer TP=2 spike is the 206 s in exo
phase 19 and is made of the same branch shapes.

## Fences: cost and exactness

p48 multi-row path (P48_ROWS=512 loop + spec rollback), 24 timed steps:

| arm | m4-1 median ms/step | m4-2 median ms/step |
|---|---:|---:|
| no fence | 706.41 | 707.06 |
| every 2 layers | 707.77 (**+0.19%**) | 707.52 (**+0.06%**) |
| every 1 layer | 709.94 (**+0.50%**) | 709.21 (**+0.30%**) |

All arms bit-identical (logits compared with `mx.array_equal`). Requirement
was < 3%; measured <= 0.5%.

Chunk-set form (1536 tok, fences interleaved over 5 reps): every-2 -0.03% /
-2.14% (m4-1/m4-2), every-1 -0.25% / -2.16% -- i.e. within run-to-run noise.

Fence exactness at long context (8704 tok): spacings 0/1/2/4 and async queue
depths 0/1/2/4 all `exact_vs_nofence=True`, same peak (24.19 GB), 675-686 tok/s.

## Correctness / parity

- `[512,512,512]` chunked forward == one 1536-row forward bit-exactly on the
  p48 set? **No** -- cos 0.9962187, but argmax 100%: that is the model's own
  documented chunk-shape dependence, see next bullet.
- The driver is exactly the composition of the equivalent plain `model()` calls:
  plain 512x3 vs driver 512x3 -> `exact=True`, `max_state|d|=0`.
- Session continuation with the same boundaries ([1024] then [512]) vs one-shot
  `[512,512,512]`: `exact=True`, cos 0.9999999, all cache-state tensors |d|=0
  (window ring, comp_kv, index_k, compressor carry).
- A split at 768 ([768]+[768]) vs [512,512,512]: cos 0.9896, argmax 0%. The
  PLAIN path at the same split: cos 0.9963 vs its own one-shot -- the model has
  this dependence, not the driver (exo phase 3 / `spec.py` note). Cache diffs
  are real (L2.comp_kv 0.41, L2.index_k 0.5, carry 0). Stream E must reconstruct
  the same chunking, or compare against the plain path's own chunking.
- API contract (m4-1): 11/11 -- int32 [1,1] argmax == argmax(logits), full-logits
  [1,256,129280], taps tuple with keys matching the plain model, offset
  advance/reuse, list/[n]/[1,n] inputs, batch>1 rejected with ValueError.
- Decode after a fenced prefill: 13.0 ms/step median, unchanged (fence gated on
  n > 1).

## Adaptive chunk size at long context

8704 tokens, m4-1: sizes `[512 x16, 128 x4]` (plan_match=True, first switch at
offset 8192), 12.8 s = 678 tok/s, peak 24.29 GB. Fixed 512 chunks: 12.4 s =
699 tok/s, peak 24.33 GB. Per-token cost: 1.35 ms/tok at 512 rows, 2.47 ms/tok
at 128 rows -- the 128-chunk trades ~3% throughput for a smaller transient at
long context (the real 16K+ case lives on stream B's tiling). Adaptive-vs-fixed
cache state |d| = max 1 quantum (chunk-shape dependence again, expected).

## Unfinished / risky

- Every number here is the single-node 8-layer subset on one node; the
  full-model TP=2 16K/32K window is the parent's two-node run.
- `long_threshold=8192` and `long_chunk=128` are plan defaults, not tuned to the
  full model's memory curve -- revisit after B lands.
- The driver does not itself force chunking to match a session's earlier prefix;
  stream E owns that invariant (documented in prefill.py).
