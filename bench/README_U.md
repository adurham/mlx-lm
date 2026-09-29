# Stream U -- per-round accounting for the DSpark spec loop

Round 2b, stream U. Owns `spec.py` timing/orchestration only. Goal: account for
the ~11 ms per speculative round that phase 18 could not attribute (draft 11 ms
+ verify ~88 ms + ~11 ms unknown at gamma 3), and confirm the draft head needs
no per-draft-token collective beyond the exact argmax.

## What is in spec.py now

`spec.set_round_log(list)` appends one dict per round with the wall time of
every named term, the round's gamma, its acceptance and committed-token count.
With no log installed the loop is the production loop; with one installed only
`perf_counter()` calls and dict writes are added, so tokens are unchanged
(checked against the un-instrumented call in `pU7`: `P == L` true).

Terms: `draft_build` (head graph, host), `draft_gpu` (only when forced
separately), `body_build` (concat/snap/forward graph, host), `wait` (the round's
`mx.eval(am, d)`), `readback` (`np.array` of the 4 token outputs),
`rollback_ctx` (`snap`/`stashes`/`rollback`/`append_ctx` construction), `emit`
(list work), `total`.

`DSV41_SPEC_ISOLATE=1` (or `spec._ISOLATE = 1`) forces the draft tokens on their
own sync before the body graph is built. Without it the body forward's engram
hasher materialises ids with `np.array` and that force absorbs the draft's GPU
time.

`rollback()` also gained a guard: a layer whose compressor never stashed for
this chunk is skipped. On a full model every kv source stashes every forward,
so this changes nothing there; on a partial build it prevents a `None` deref.

## Harnesses

| file | what it does |
|---|---|
| `pU_cap.py` | expert-count cap helper (router width + loaded table) so a subset fits a memory budget |
| `pU0_memprobe.py`, `pU0m_probe.py`, `pU0c_probe.py`, `pU0f_full_mem.py` | memory of subsets / head / full depth, with and without caps |
| `pU2_divergence.py` | determinism + first-divergence between eval placements |
| `pU3_decompose.py` | full-depth round decomposition, async on/off, gamma sweep, verify row cost |
| `pU5_identity.py` | world=1 identity checks (instrumentation inert, eval placement inert) |
| `pU7_two_rank.py` | the main one: two ranks on ONE node (ring backend), identity + terms + census |
| `pU8_census.py` | per-call-site `all_sum` census for one draft / one verify / rollback / append |
| `pU9_draftcost.py` | prices the draft's per-token collectives, checks the variants |

All run on one Mac. Two-rank tests use two ring processes under a single
`lockf`; nothing touches another node.

## Results (see the stream report for the full tables)

- The round total closes to `+/-0.00 ms unattributed` on every instrumented arm.
- At gamma 3 a round charges **R = gamma + 1 = 4** verify rows (anchor + 3
  drafts), not R3. The phase-18 budget subtracted the R3 number.
- Draft at gamma 3: 6 `all_sum` calls = 3 (sharded draft MoE, one per stage) +
  3 (the exact cross-rank argmax, one per markov step). That is 1 argmax
  collective per draft token, and it is the *exact* merge; no other per-token
  collective exists.
- `head.draft` graph construction is 0.45 ms; the draft's GPU time is ~15 ms at
  this subset size.
- Tokens are identical across every eval placement tested (`P == L`, `P == I`),
  deterministic across repeated runs, and rank-identical at world 2.

## Running

```
rsync -a --exclude __pycache__ mlx_lm <mac>:dsv41-ws2/U/
rsync -a bench <mac>:dsv41-ws2/U/
ssh <mac> 'cd ~/dsv41-ws2/U &&
  export PU7_PKG=$HOME/dsv41-ws2/U EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 &&
  <launch pU7 under lockf>'
```

Every entry point takes its package root from `PU*_PKG`, its layer list from
`PU*_LAYERS`, and prints active/peak memory. Wrap every run in
`lockf -k ~/dsv41-gpu.lock`.
