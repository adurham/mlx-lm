# W: post-prefill first-decode-step cost — root cause and fix

Date: 2026-09-29. Stream W (post-prefill compile). Single-node layer-subset
measurements on macstudio-m4-1; production stopped, so runs were 23-80 GB.
Probes: `pW1`..`pW27` in this worktree (all GPU runs under
`lockf -k ~/dsv41-gpu.lock`, `EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1`).

## What the original brief assumed, and what is actually true

The brief said the first 1-3 decode steps after a long prefill (1.3-1.9 s at
two nodes, 76 ms steady) were a `mx.compile` **recompile** on a new
shape/dtype, and asked for a shape-stable/bucketed fix instead of more warmup.

That is not what it is. Measured, at 30 layers / 16K on one node:

| quantity | step 0 | steady | note |
|---|---|---|---|
| total | 562-590 ms | 52.6 ms | 10.7x |
| `mx.eval` | 3.2-3.4 ms | 3.0 ms | flat |
| `mx.metal.gpu_time_ns()` | 0 | 0 | no GPU term measurable |
| fresh device `newBuffer` | 145-195 | 0 | ~6 ms at 0.032 ms each |
| **host graph build** | **559 ms** | **49.6 ms** | all of it |

and from cProfile of the slow step (pW23, 30 layers / 16K):

```
549 ms total, 530 ms in sparse_attention.py:205(sparse_attn)   (31/30 calls, same as steady)
 38 ms total,  38 ms in sparse_attention.py:205(sparse_attn)   (steady step)
```

So the premium sits in sparse attention's per-layer `mx.eval(out)` fence: it is
the **first full-depth decode flush after the prefill**, a host-side sync cost,
not a compile and not per-op work.

## Ruled out by measurement (each with its own probe)

| hypothesis | probe | result |
|---|---|---|
| `mx.compile` re-trace on decode shapes | pW4, pW11 | every compile key used by decode is created during `warmup()`; pW11 logs zero cold keys in decode; shapes are context-independent (kv_all/idxs widths identical step 0 vs steady) |
| `mx.clear_cache()` evicting compiled graphs | pW2 | compiled fns and Metal JIT kernels survive `clear_cache()`; cold re-run costs 2 ms |
| Metal buffer-pool misses (context-sized buffers absent) | pW9, pW12, pW18, pW19, pW21 | 195 misses price out at 0.032 ms each (~6 ms). Priming exact miss sizes, generic pow2 sizes, with and without a pool clear: all neutral-to-worse (646-651 ms vs 583 ms) |
| Python cyclic GC | pW22 | 9373 collections during prefill (4.8 s total, amortised) but **0 during the slow step**; `gc.disable()` 549 ms, `gc.freeze()` 582 ms — no change |
| OS page faults / memory compression | pW14 | pageins/faults/compressions/swapouts all `+0` across the slow step at 79 GB active |
| the fence being the problem | pW25 | `DSV41_SPARSE_FENCE=0` (31 fewer syncs per step) still 12.8x first step |
| a pending-work backlog from prefill | pW24 | a tiny `mx.eval` fence and `mx.synchronize()` at the boundary do not absorb it (632 ms first step) |
| per-input / per-shape work | pW21 | repeating the *identical* step right after a rollback costs 49.9 ms |

It is **superlinear in the built model's live footprint**, which is why the
2-layer, 8 GB subset never showed it and the 30-layer, 80 GB subset does:

| subset | active | ctx | first step | steady | ratio |
|---|---:|---:|---:|---:|---:|
| 2 layers | 7.8 GB | 8K/16K | 3.9 ms | 3.0 ms | ~1.3x |
| 8 layers | 23.2 GB | 16K | 15.4 ms | 13.3 ms | ~1.2x |
| 20 layers | 55.6 GB | 16K | 78-100 ms | 35.7 ms | 2.2-2.8x |
| 30 layers | 80.1 GB | 16K | **550-650 ms** | 52.6 ms | **10.5-12.4x** |

The two-node production observation (105 GB/rank, 1514/1860/1277 ms then 76 ms
steady) is the same curve further out. Decay over 2-3 steps matches: the
one-time flush cost is paid once, then steady state.

## The structural fix

One insight makes this a clean fix rather than a warmup: the cost is **one-time
per prefill and moveable**, not per-step. A throwaway decode step taken at the
prefill boundary absorbs the entire premium, after which the first real decode
step is steady state (pW24 arm D: boundary 573 ms, then step 0 50.2 ms against
steady 49.6 ms = **1.01x**).

`prefill.decode_prime(model, cache)` does exactly that and is called at the end
of `prefill()` by default (`DSV41_DECODE_PRIME=0` disables it):

* runs one 1-row forward,
* restores every mutable carry it can touch bit-exactly (compressor
  `kv_state` / `score_state` / `chunk_*`; the window ring and compressed-KV /
  index-key caches are position-addressed, so the probe's slot is rewritten by
  the next real write before it can be read),
* restores `cache.offset`,
* discards the probe's outputs.

### Acceptance numbers (pW26/pW27/pW28, 30 layers)

| context | arm | first step | steady | ratio |
|---:|---|---:|---:|---:|
| 16K | baseline | 608.9 / 562.5 / 550.7 ms | 52.6-52.7 ms | 10.45-11.57x |
| 16K | primed | **53.2 / 53.2 / 53.5 ms** | 52.8-53.8 ms | **0.99 / 1.01 / 1.01x** |
| 8K | primed | 51.9-53.2 ms | 51.0-51.1 ms | 1.00-1.02x |

Boundary cost: 558-600 ms (moved, not removed — it is now billed to prefill).

### Parity (pW28, both contexts, 3 teacher-forced steps per arm)

```
C=8192  PARITY: 3-step logits exact=True | carry kv_exact=True score_exact=True |
        ring differing elems/layer≈500-511 (of 65536) probe-slot=0 slot-only=True
C=16384 PARITY: 3-step logits exact=True | carry kv_exact=True score_exact=True |
        ring differing elems/layer≈500-511 (of 65536) probe-slot=0 slot-only=True
```

The window-ring difference is the probe's own written slot (position
`offset % window`); the `slot-only=True` check confirms every differing element
is confined to that one column, and it is never read before the next real write
overwrites it. Three teacher-forced logits compare **bit-identical**
(`mx.array_equal`, not max|d| — the rows carry `-inf`).

## What is not fixed, and risks

* **The premium is moved, not removed** (~1:1): the boundary prime costs about
  what the first decode step used to cost (30 layers/16K: 558-608 ms). Total
  wall time per turn is essentially unchanged. The win is *where* it is paid:
  during prefill, before any token streams, instead of as 1.3-1.9 s of dead
  time on the first 1-3 decode steps. If a caller cares about prefill latency
  more than first-token latency, `DSV41_DECODE_PRIME=0` restores the old
  behaviour exactly.
* For SHORT delta prefills in a session the prime is a fixed per-turn overhead
  (pW29, 30 layers, 16K+offset: 128-tok delta prefill 844 ms total with the
  prime, 512-tok 2996 ms; decode after each is 0.99-1.01x steady). That is the
  price of every turn starting at full decode speed.
* **Not** measured at the real scale: the two-node 105 GB/rank case is where the
  premium is 1.3-1.9 s; the single-node curve predicts ~1.4 s at ~100 GB and the
  boundary prime should absorb it the same way, but no two-node run was
  permitted from this stream — the parent's runbook below is that check.
* The window-ring slot write is a deliberate, documented non-restoration. It is
  safe by position-addressing, and pW28 verifies every differing element is
  confined to that slot (`slot-only=True`).
* The probe id is `0`; engram hashing of that id writes engram id slots for
  position `offset` (also position-addressed).

## GPU commands for the production-down window (parent)

All single-node, one job per node, under the lock. Layer subsets only.
`W = ~/dsv41-ws2/W`, `PY = ~/repos/exo/.venv/bin/python`,
env `EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1`.

1. **Reproduce + accept at 8K/16K (≈80 GB, ~7 min)**
   ```bash
   cd ~/dsv41-ws2/W && PW_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
     PW_CTXS=8192,16384 PW_STEPS=4 lockf -k ~/dsv41-gpu.lock $PY pW27_accept.py
   ```
   Expect: `baseline ... ratio 10-12x` rows and `primed ... ratio 1.01x` rows,
   then `PARITY: ... logits exact=True` for both contexts.
2. **Two-node check (parent only, this is the real acceptance)** — run `p64`
   once with the driver default (prime on) at 8K/16K:
   `P64_DRIVER=1 P64_LENS=8192,16384`. Expect the decode line to start at
   ~76-110 ms instead of 1380-1860 ms, i.e. no 1.3-1.9 s first steps.
3. **Optional A/B control (same session)** — `DSV41_DECODE_PRIME=0` and repeat
   step 2; expect the old 1.3-1.9 s first steps back.

Unfinished: whether the same mechanism also explains the 16K two-node prefill
GPU-timeout class of crash is untested; this stream only addressed the
post-prefill decode boundary.
