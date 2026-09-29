# DSv4.1 greedy parity harness (stream P, round 2)

A standalone harness that answers one question with measurements: *is this
build of DSv4.1 producing the same greedy text as the recorded reference —
and would it actually notice if it were not?*

Files (all owned by stream P; add `tests/parity/` to the mlx-lm fork before
merge):

| file | what it is |
|---|---|
| `dsv41_parity.py` | the runner: modes `cos`, `record`, `check`, `negctl` |
| `parity_core.py` | MLX-free core: comparison, gates, fingerprints, ref files |
| `test_parity_core.py` | 26 unit tests for the core (no GPU, runs anywhere) |
| `run_two_node.sh` | JACCL TP=2 launcher (parent only), p64 env layout |

## Measured acceptance (m4-1, 2026-09-29, production untouched, under lockf)

    negctl --layers 2 --world 2 --cos-layers 2 --max-new 16 --stub experts   [default flip 2]
      (1) clean re-run identical                          True
      (2) argmax flip @2 caught, first_div == 2           True (n_match 2/17)
      (3) embedding-noise ladder 0.01/0.08/0.64           quiet / div1 / div0
          policy: low rung quiet=True, high rung detected=True
      (4) cos gate: clean 0.999914 -> stub(experts) 0.984593, gate FAILs
          (also measured: stub idx 0.700962, L20 clean 0.999945, L24 0.999977)
      verdict all=True, flip_exact=True, exit 0, peak 7.43 GB

    record / check, layers 2 world 2:
      record --margins: both prompts 17/17 tokens; min_margin 0.0547 / 0.0078
        (logits fp32 top1-top2 gap; near-zero gap = a float tie, not a regression)
      check: 17/17 match, first_div None, exit 0
      check --perturb argmax:5: prompt0 first_div 5, match 5/17, exit 1
      check vs a chunk=32 reference: structural guard -> {'chunk': [32, 0]}, exit 1
      check across code revisions: code_digest FAIL unless --allow-code-drift

    cos gate per layer (isolated, full width, world 1):
      L2 0.999914  L20 0.999945  L24 0.999977   -> gate PASS, sweep peak 6.43 GB
      L0..L9 0.999914..0.999987 (10-layer run, peak 7.27 GB)

    memory (single node, world 2 = half-width experts, no collective):
      layer 2 token run: 4.49 GB weights, 5.88 GB build, 7.43 GB first forward, 4.88 GB steady
      layers [2,20] would need 9.9 GB first forward -> NOT allowed under the 8 GB rule
      full 0..39 cos sweep grows ~0.13 GB/layer -> also over 8 GB (scheduled window only)

## Modes

    cos      per-layer cosine vs the p30 clamped trace (isolated, gated at
             0.9998) + end NLL/top-1 when the whole 0..39 stack is swept
    record   N prompts greedy -> reference JSON (ids, tokens, fingerprint)
    check    same prompts vs a saved reference: token-by-token compare with
             first-divergence index, cos gate, NLL gate; `--perturb argmax:k`
             proves the comparison fails when the run is perturbed
    negctl   self-contained negative control (see below)

Exit 0 = every gate passed, 1 = a gate failed, 2 = error / bad reference.
Every run also prints one machine-readable `PARITY_JSON {...}` line.

### `negctl` (the acceptance gate for this stream)

1. **clean re-run must compare equal** — two identical runs on the same build
   must be token-identical (catches nondeterminism in the harness itself).
2. **one-token flip must be caught, at the exact index** — generated token
   `--flip-step k` is replaced by `(tok+1) % vocab`; the comparison must fail
   and `first_div` must be exactly `k`.
3. **embedding noise must change the tokens** — the prompt forward's embedding
   gets one-shot noise of relative size `--noise r`; the tokens must differ.
4. **a component stub must fail the isolated cos gate** — `--stub
   {hc,shared,experts,attn,gate,lin,rope,fq,sattn,idx,rms,engram}` (the p48
   ablation stubs) is applied and the cos sweep must drop below the bar.

## Running it (GPU rules apply)

    # single node, layer subset, <=8 GB: half-width experts, no collective
    ssh macstudio-m4-1
    cd ~/dsv41-ws2/P
    lockf -k ~/dsv41-gpu.lock env EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
        ~/repos/exo/.venv/bin/python -u tests/parity/dsv41_parity.py negctl \
        --pkg-root ~/dsv41-ws2/P --max-new 16
    # default layers: token run 2, cos sweep 2,20,24 (measured peaks 7.43 / 6.43 GB)

    # full two-node model (parent only, production stopped):
    tests/parity/run_two_node.sh record --layers all --max-new 32 \
        --ref ~/dsv41-ws2/parity/ref_full.json --note "full two-node reference"
    tests/parity/run_two_node.sh check --layers all \
        --ref ~/dsv41-ws2/parity/ref_full.json
    # (or export P_REF and call the harness directly with --dist jaccl)

The harness enforces the memory rule itself: it estimates the weights it will
load from the safetensors headers (`mtp.*`/`vision.*` excluded -- the builder
skips them) and refuses to start above `--max-gb` (default 8, single node),
and aborts if the *measured* `mx.get_peak_memory()` crosses it at build or
after a token run. The cos sweep is per-layer (one block resident), so a
single-block estimate is used there. Note the FIRST forward carries a
one-time compile transient (~1.5 GB here): measure with a fresh process, not
after a warm one. `--force-oversize` overrides (never for a single-node run).

`--world`/`--rank` select the TP slice on a single node without any collective
(the p48/p66 convention: rank-0 half-width experts). Token runs default to
world 2 (half-width). The cos sweep ALWAYS builds full-width (`--cos-world 1`)
because the p30 trace is a world-1 run and a sliced, collective-less build is
not comparable to it -- that difference is worth 0.014 cos (0.9859 vs 0.9999).
With `--dist jaccl` the collective is real and the trace comparison is valid.

`--prompts p30` uses growing prefixes of `~/p30_prompt_ids.json` (default
2 prompts: 64 and 128 tokens); `--prompts chat` uses real chat-template
prompts from the checkpoint with the production template settings;
`--prompt-file <json>` takes `[[ids...],...]` or `[{name,ids},...]`.
`--chunk N` prefills through `prefill.prefill` instead of one forward (the
chunk boundaries are part of the fingerprint, since the body's output depends
on them).

## Reference files

A reference is a JSON doc (`schema: dsv41-parity/1`):

    prompts:      [{name, ids, tokens, stats}]   -- the compared thing
    fingerprint:  {layers, world, rank, max_new, chunk, model_dir, native_dir,
                   token_map_digest, prompts_digest, pkg_digest, structural}
    nll, timings, memory, note

The **structural** fields must match or the token comparison is declared
invalid (hard fail); including `chunk` because the body's output depends on
the prefill chunk boundaries (a property of the model — see
`prefill.py`'s docstring). `pkg_digest` (hash of the model code under test) is
a **soft** signal: a mismatch is reported so a reviewer knows the comparison
crossed revisions.

## Known limits

- Greedy only (temperature 0). The spec/sampling paths are streams S/W's
  territory; this harness compares the plain path so a token difference is
  always traceable to the body.
- A layer subset is not the full model: the subset keeps its own semantics
  (the residual stream is fed by the built layers only), so token parity is
  only meaningful between runs with the same `--layers` — which the
  fingerprint enforces.
- `cos` on a subset gates the built layers only; NLL/top-1 need the full
  0..39 sweep.
- Two-node runs print one `PARITY_JSON` per rank; rank 0 is the canonical one.
- The code digest covers the model package and this harness (PKG_FILES); it
  does NOT cover the MLX build itself. Record the MLX version in the note when
  it matters.

## GPU policy: NEVER run this while production is up

Measured on 2026-09-29: even a 7 GB run next to production stalls/kills it
(GPU contention, not just memory). Until the parent says otherwise, every GPU
mode below runs ONLY in a scheduled production-down window, one rank-pair at a
time, with `--require-free-gb 60`. CPU-only work (unit tests, `--help`, dry
parsing) is always fine.

### Parent runbook (production-down window, in order)

1. `python3 tests/parity/test_parity_core.py` — CPU, anywhere, ~0 s.
   Expect: `Ran 26 tests ... OK`.
2. Single-node smoke on m4-1 (production confirmed down):
   `lockf -k ~/dsv41-gpu.lock env EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \
     ~/repos/exo/.venv/bin/python -u tests/parity/dsv41_parity.py negctl \
     --pkg-root ~/dsv41-ws2/P --require-free-gb 60`
   Expect: four controls pass, `verdict: all=True`, exit 0, peak ≈7.4 GB.
3. Per-layer cos gate (full width, one block resident):
   `... cos --layers 0-9 --isolate 1 --require-free-gb 60`
   Expect: gate PASS with worst_cos ≈0.99991 at L2, peak ≤7.5 GB.
4. Full two-node record (JACCL, both Macs, production down):
   `tests/parity/run_two_node.sh record --layers all --max-new 32 \
     --ref ~/dsv41-ws2/parity/ref_full.json --note "full two-node reference"`
   Expect: rank0 builds ~105 GB, ~17 tok/s plain, and writes the reference;
   this is the baseline everything else is compared against.
5. Full two-node check: `tests/parity/run_two_node.sh check --layers all \
     --ref ~/dsv41-ws2/parity/ref_full.json`
   Expect: `tokens: true` for all prompts, `first_div: None`, `nll_gate: true`
   (p46 bar mean ≤1.010 / median ≤0.040 / top1 ≥78.0), exit 0.
6. Negative control on the FULL model (the acceptance test for the rollout
   gate): `tests/parity/run_two_node.sh check --layers all --perturb argmax:7 \
     --ref ~/dsv41-ws2/parity/ref_full.json`
   Expect: exit 1, `first_div: 7` — a gate that cannot fail is not a gate.
7. Restore production, then re-run step 3 to confirm the node came back clean.
