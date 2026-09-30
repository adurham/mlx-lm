# LOCAL ADDITION. Speculative decode (DSpark draft + chunk verify) for V4.1.
"""Round loop, rollback and helpers.

Chunk verify runs the ``gamma + 1`` verify rows in ONE body forward. It is not
bit-identical to plain greedy (the body result depends on chunk shape, a
property of the body measured in exo phase 3), which is the same tradeoff
production DSv4 makes. Rollback: window ring and compressed caches are
position-addressed and are rewritten before they are read again; only the
compressor open-group carry must be rebuilt (from the saved carry plus the
rows the compressor stashed for this chunk).
"""

from __future__ import annotations

import os

import mlx.core as mx


def snap(cache, start_pos: int):
    st = []
    for lc in cache.layers:
        cs = lc.comp_state
        if cs is None:
            st.append(None)
            continue
        m = start_pos % lc.ratio
        st.append((mx.array(cs.kv_state[:, :m]), mx.array(cs.score_state[:, :m]), m)
                  if m else (None, None, 0))
    return start_pos, st


def stashes(cache):
    return [(lc.comp_state.chunk_kv, lc.comp_state.chunk_score)
            if lc.comp_state is not None else (None, None) for lc in cache.layers]


def _rebuild(cs, ratio, chunk_start, target, saved, stash):
    s = ratio * (target // ratio)
    need = target - s
    if need == 0:
        return
    saved_kv, saved_sc, m = saved
    stash_kv, stash_sc = stash
    pk, ps = [], []
    if s < chunk_start:
        lo = max(s, chunk_start - m)
        pk.append(saved_kv[:, lo - (chunk_start - m): m])
        ps.append(saved_sc[:, lo - (chunk_start - m): m])
    lo2 = max(s, chunk_start)
    pk.append(stash_kv[:, lo2 - chunk_start: target - chunk_start])
    ps.append(stash_sc[:, lo2 - chunk_start: target - chunk_start])
    nk = mx.concatenate(pk, axis=1)
    ns = mx.concatenate(ps, axis=1)
    if nk.shape[1] != need:
        raise RuntimeError(f"carry rebuild produced {nk.shape[1]} rows, need {need}")
    cs.kv_state[:, :need] = nk
    cs.score_state[:, :need] = ns


def rollback(cache, sn, target: int, st) -> None:
    chunk_start, saved = sn
    cache.offset = target
    for lc, sv, stash in zip(cache.layers, saved, st):
        if lc.comp_state is None:
            continue
        # A layer whose compressor did not run for this chunk has no stashed
        # rows, so its carry is untouched and there is nothing to rebuild. This
        # only happens on a PARTIAL build (out-of-subset kv source never
        # stashes); on the full model every source runs every forward. (ws2/U)
        if stash[0] is None:
            continue
        _rebuild(lc.comp_state, lc.ratio, chunk_start, target, sv, stash)


# Measured on 2x M4 Max TP=2 (exo phase 16/17): verify forward ms by rows.
VERIFY_MS = {1: 58.5, 2: 74.9, 3: 87.9, 4: 97.7, 5: 111.7, 6: 120.1}


class GammaPolicy:
    """Pick the draft length each round from observed per-position acceptance.

    q_k = P(draft k accepted | drafts 1..k-1 accepted), Beta(1,1)-smoothed;
    positions never tried reuse the last estimate. Chooses the gamma that
    maximizes expected committed tokens per round time. Host-only, no syncs."""

    def __init__(self, gammas=(1, 2, 3, 4), start=3, draft_ms=(8.5, 0.9),
                 overhead_ms=4.0, verify_ms=None, warmup=4):
        self.gammas, self.g = gammas, start
        self.dms, self.oh = draft_ms, overhead_ms
        self.v = verify_ms or VERIFY_MS
        self.tried = [0] * 8
        self.acc = [0] * 8
        self.rounds, self.warmup = 0, warmup

    def update(self, gamma: int, n_acc: int) -> None:
        self.rounds += 1
        for k in range(1, gamma + 1):
            self.tried[k] += 1
            if n_acc >= k:
                self.acc[k] += 1
            else:
                break

    def _q(self, k):
        last = 0.7
        for j in range(1, k + 1):
            if self.tried[j]:
                last = (self.acc[j] + 1) / (self.tried[j] + 2)
        return last

    def next(self) -> int:
        if self.rounds < self.warmup:
            return self.g
        best, best_rate = self.g, -1.0
        for g in self.gammas:
            e, p = 1.0, 1.0
            for k in range(1, g + 1):
                p *= self._q(k)
                e += p
            t = self.dms[0] + self.dms[1] * g + self.v[g + 1] + self.oh
            if e / t > best_rate:
                best, best_rate = g, e / t
        self.g = best
        return best



# Prompts longer than this go through prefill.prefill (fenced 512-row chunks)
# instead of one forward. A single multi-thousand-row forward is one huge
# command-buffer chain per rank; measured 2026-09-29 on the full two-node
# model, a 1.9K-token prompt in one forward tripped the Metal GPU watchdog
# (31 "GPU Timeout" on rank 0) while every <100-token prompt ran clean.
_ONESHOT_MAX = int(os.environ.get("DSV41_SPEC_ONESHOT_MAX", "512"))


def _prefill_with_taps(model, prompt_ids, cache):
    """(argmax ids of the last row, {layer: taps for ALL prompt rows})."""
    ids = mx.array([list(prompt_ids)])
    if len(prompt_ids) <= _ONESHOT_MAX:
        return model(ids, cache, last_logit_only=True, return_taps=True, argmax=True)
    from . import prefill as _PF
    chunks = []
    am = _PF.prefill(model, ids[0], cache, last_logit_only=True, argmax=True,
                     taps_out=chunks)
    taps = {L: mx.concatenate([c[L] for c in chunks], axis=1) for L in chunks[0]}
    return am, taps

def generate(model, head, prompt_ids, max_new: int, *, gamma: int = 3,
             adaptive: bool = True, eos_id: int = 1, policy=None,
             temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
             seed: int | None = None, sampler=None, sp=None, cache=None,
             cache_state=None, cache_update=None, anchor=None):
    """Speculative decode with one host sync per round.

    The draft is NOT evaluated on its own: its tokens feed the verify forward
    lazily and both are read back in a single sync. Returns (tokens, stats).

    ``temperature <= 0`` (the default) is the greedy path: argmax targets,
    argmax drafts, no RNG, no logits gather -- byte-for-byte the behaviour this
    function had before sampling existed.  Any ``temperature > 0`` switches to
    proper speculative sampling in :mod:`.sampling`, which keeps the same round
    structure and cache handling: draft tokens are drawn from the filtered
    draft distribution and accepted with probability ``min(1, p_target/p_draft)``,
    rejections resample from the normalised residual ``(p_target - p_draft)+``,
    and a fully accepted round takes its bonus token from the target.  That
    path needs the full-vocab rows the sharded head does not otherwise produce,
    so it gathers them (see the sampling module) and returns the same keys plus
    ``rejects`` / ``drafted`` / ``accept_rate``.

    CONTINUING AN EXISTING SESSION (``cache=``)
    -------------------------------------------
    Pass the live ``ModelCache`` that already holds the prompt prefix
    (``session_cache.SessionCache.cache``; see ``serve.Session``) together with:

    * ``anchor`` -- the token id to continue from. It must be the argmax the
      caller's own prefill returned for the last row it fed, i.e. the token
      that sits at ``cache.offset``'s *previous* position; this call feeds the
      anchor row itself as the first verify row (exactly as the fresh path
      feeds the prompt's last-row prediction).
    * ``cache_state`` -- the draft head's window caches (``head.make_cache(1)``)
      already fed the context taps, i.e. with ``n_ctx == cache.offset``.
    * ``cache_update`` -- any object with ``append_tokens(ids)`` /
      ``mark_seen(ids)``; every row fed to the cache is reported there, so a
      ``SessionCache``'s token history tracks the cache exactly across a
      cancelled or completed turn.

    Contract: ``prompt_ids`` rows are prefilled first (empty when the caller
    has already fed them), the prefix is never re-fed, the cache is left at the
    last committed token's position, and no rewind happens on exit. Stats gain
    ``anchor``/``cache_offset`` keys on this path.

    WARM-UP (greedy path)
    ---------------------
    Two distinct problems were measured on this stack (2026-09-30, p97-p116):

    * **Load-time kernel compilation storm.** The first forward after load
      compiles ~17 custom Metal kernels per new shape; on the full model
      layers 17-26 of the first forward take 4-7 s EACH (47 s of compile) and
      the rank skew trips the Metal watchdog ("Caused GPU Timeout Error",
      hundreds of them; first failure site sparse_attn's fence eval). Fixed by
      ``prefill.warmup(model)`` before any real workload (p115: 0 timeouts on
      the exact repro that failed twice in a row; layers compile in 30-90 ms
      afterwards). Callers that can wait ~2 min at load should call it.
    * **Slow spec rounds after a long prompt.** After a prompt of >= ~430
      rows, verify rounds run ~2 s instead of ~110 ms for a while (sometimes
      all run, sometimes a few). Measured with per-module exclusive timing
      (p106): the time sits in the EXL3 expert kernels *in-model* (~35 ms/call
      vs 0.5-2.3 ms in isolation on the captured inputs) and in the peer waits;
      it is NOT the weights being evicted (p108: 0 decompressions in the slow
      rounds; locking 112 GiB changes nothing) and NOT the data (p107: real vs
      random inputs identical in isolation). Root cause still open; candidate
      is macOS GPU memory-pressure behaviour (p109/p112: "Pages wired down"
      swings 20-70 GiB within single slow rounds while MLX active memory is
      flat at ~100 GB).
    * ``DSV41_SPEC_PRIME_STEPS`` (default 8) runs plain greedy steps after the
      prefill (>= ``DSV41_SPEC_PRIME_MIN_ROWS``, default 64, rows) and
      ``DSV41_SPEC_REPRIME_STEPS`` (default 8) falls back to a burst whenever a
      verify round exceeds ``DSV41_SPEC_SLOW_MS`` (default 400 ms). Priming
      cures short/mid-context slowdowns (p98/p99: 110 ms rounds, acceptance
      ~2.8) but does NOT fully cure >=2.4K context (p116 with warm-up: still
      ~2 s rounds) and a primed run at 2456 rows produced degenerate comma
      text (p99/p100/p116). Both knobs default 0: OFF until the slow state is
      root-caused. Enable only for experiments.
    """
    import numpy as np
    import time

    if temperature and temperature > 0.0:
        from . import sampling as _sampling

        if cache is not None:
            raise NotImplementedError(
                "session reuse (cache=) is implemented for the greedy path; the "
                "sampling loop (sampling.spec_generate) does not take a live "
                "cache yet -- use serve.Session(..., greedy) or the sampling "
                "thread's follow-up")
        return _sampling.spec_generate(
            model, head, prompt_ids, max_new, gamma=gamma, adaptive=adaptive,
            eos_id=eos_id, temperature=temperature, top_p=top_p, top_k=top_k,
            seed=_sampling.DEFAULT_SEED if seed is None else seed,
            sampler=sampler, policy=policy, sp=sp)

    taps_ids = list(model.args.dspark_target_layer_ids)

    def tapcat(t):
        return mx.concatenate([t[L] for L in taps_ids], axis=-1)

    cont = cache is not None
    if not cont:
        if cache_state is not None or cache_update is not None or anchor is not None:
            raise ValueError("spec.generate: anchor/cache_state/cache_update need cache=")
        cache = model.make_cache(1, max_seq_len=len(prompt_ids) + max_new + 16)

    prompt_ids = [] if prompt_ids is None else prompt_ids
    pol = policy or GammaPolicy(start=gamma)
    hist, gams = [], []
    reprises = []
    if cont:
        n_prefilled = 0
        dsc = head.make_cache(1) if cache_state is None else cache_state
        if len(prompt_ids):
            n_prefilled = len(prompt_ids)
            am, taps = _prefill_with_taps(model, prompt_ids, cache)
            if cache_update is not None:
                cache_update.mark_seen(prompt_ids)   # the forward already fed them
            head.append_ctx(tapcat(taps), dsc)
            anchor = int(np.asarray(am).reshape(-1)[0])
        if anchor is None:
            raise ValueError("spec.generate: continuing a session needs anchor= (the "
                             "prefill's argmax for the last row fed) or prompt_ids")
    else:
        n_prefilled = len(prompt_ids)
        am, taps = _prefill_with_taps(model, prompt_ids, cache)
        dsc = head.make_cache(1)
        head.append_ctx(tapcat(taps), dsc)
        anchor = int(np.asarray(am).reshape(-1)[0])
    out = [int(anchor)]
    t0 = time.perf_counter()
    # WARM-UP: plain greedy steps cure the slow-round state (see the docstring
    # "WARM-UP" note). K is a MINIMUM run at the start; if the state appears
    # again later (it can: p103 showed it spreading to short prompts right
    # after a long one, and to the first rounds of any prefill), the spec loop
    # detects slow rounds and falls back into plain steps until rounds are
    # fast again. The plain steps are real greedy tokens, so nothing is wasted.
    prime_steps = int(os.environ.get("DSV41_SPEC_PRIME_STEPS", "0"))
    prime_min_rows = int(os.environ.get("DSV41_SPEC_PRIME_MIN_ROWS", "64"))
    slow_ms = float(os.environ.get("DSV41_SPEC_SLOW_MS", "400"))
    reprime = int(os.environ.get("DSV41_SPEC_REPRIME_STEPS", "0"))

    def plain_step():
        nxt = mx.array([out[-1]], dtype=mx.int32)
        a1, t1 = model(nxt[None], cache, last_logit_only=True,
                       return_taps=True, argmax=True)
        head.append_ctx(tapcat(t1), dsc)
        nxt = a1[:, -1].astype(mx.int32)
        mx.eval(nxt)
        if cache_update is not None:
            cache_update.mark_seen([int(out[-1])])
        out.append(int(nxt))

    if prime_steps > 0 and n_prefilled >= prime_min_rows:
        for _ in range(prime_steps):
            if len(out) >= max_new + 1 or out[-1] == eos_id:
                break
            plain_step()
    pos = cache.offset
    tloop = time.perf_counter()
    while len(out) < max_new + 1 and out[-1] != eos_id:
        g = pol.next() if adaptive else gamma
        nxt = mx.array([out[-1]], dtype=mx.int32)
        d, _ = head.draft(nxt, model.embed, model.head, dsc, width=g)
        d = d.astype(mx.int32)
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        sn = snap(cache, pos)
        tr0 = time.perf_counter()
        am, taps = model(vin, cache, return_taps=True, argmax=True)
        mx.eval(am, d)                                   # the one sync per round
        round_ms = (time.perf_counter() - tr0) * 1e3
        if round_ms > slow_ms and reprime > 0:
            # Slow-round state detected: discard this round's verify (its rows
            # are rolled back below) and run plain steps instead.
            rollback(cache, sn, pos, stashes(cache))     # undo the verify rows
            for _ in range(reprime):
                if len(out) >= max_new + 1 or out[-1] == eos_id:
                    break
                plain_step()
                pos += 1
            reprises.append(round_ms)
            continue
        tg, dd = np.array(am[0]), np.array(d[0])
        n = 0
        while n < g and tg[n] == dd[n]:
            n += 1
        pol.update(g, n)
        hist.append(n)
        gams.append(g)
        new = [int(v) for v in dd[:n]] + [int(tg[n])]
        target = pos + n + 1
        rollback(cache, sn, target, stashes(cache))
        head.append_ctx(tapcat(taps)[:, :n + 1], dsc)
        pos = target
        if cache_update is not None:
            # rows that survive AND were fed: the anchor + the accepted drafts.
            # the round's bonus token is the next anchor -- it is fed by the
            # next round (or by the caller's next turn) and must not be marked
            # yet, or the history would run ahead of the cache.
            cache_update.mark_seen([int(out[-1])] + [int(v) for v in dd[:n]])
        for t in new:
            out.append(t)
            if t == eos_id:
                break
    dt = time.perf_counter() - t0
    dt_loop = time.perf_counter() - tloop
    stats = {"tok_s": (len(out) - 1) / dt, "rounds": len(hist),
             "mean_acc": float(np.mean(hist)) if hist else 0.0,
             "ms_round": dt_loop * 1e3 / max(len(hist), 1), "gammas": gams,
             "prime_steps": max(0, len(out) - 1 - sum(1 + int(v) for v in hist)),
             "reprises": len(reprises), "slow_ms": slow_ms}
    if cont:
        stats["anchor"] = int(anchor)
        stats["cache_offset"] = int(cache.offset)
    return out, stats
