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


def generate(model, head, prompt_ids, max_new: int, *, gamma: int = 3,
             adaptive: bool = True, eos_id: int = 1, policy=None):
    """Greedy speculative decode with one host sync per round.

    The draft is NOT evaluated on its own: its tokens feed the verify forward
    lazily and both are read back in a single sync. Returns (tokens, stats)."""
    import numpy as np
    import time
    taps_ids = list(model.args.dspark_target_layer_ids)

    def tapcat(t):
        return mx.concatenate([t[L] for L in taps_ids], axis=-1)

    cache = model.make_cache(1, max_seq_len=len(prompt_ids) + max_new + 16)
    am, taps = model(mx.array([prompt_ids]), cache, last_logit_only=True,
                     return_taps=True, argmax=True)
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    nxt = am[:, -1].astype(mx.int32)
    mx.eval(nxt)
    out = [int(nxt.item())]
    pos = cache.offset
    pol = policy or GammaPolicy(start=gamma)
    hist, gams = [], []
    t0 = time.perf_counter()
    while len(out) < max_new + 1 and out[-1] != eos_id:
        g = pol.next() if adaptive else gamma
        d, _ = head.draft(nxt, model.embed, model.head, dsc, width=g)
        d = d.astype(mx.int32)
        vin = mx.concatenate([nxt.reshape(1, 1), d], axis=1)
        sn = snap(cache, pos)
        am, taps = model(vin, cache, return_taps=True, argmax=True)
        mx.eval(am, d)                                   # the one sync per round
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
        nxt = mx.array([new[-1]], dtype=mx.int32)
        for t in new:
            out.append(t)
            if t == eos_id:
                break
    dt = time.perf_counter() - t0
    return out, {"tok_s": (len(out) - 1) / dt, "rounds": len(hist),
                 "mean_acc": float(np.mean(hist)) if hist else 0.0,
                 "ms_round": dt * 1e3 / max(len(hist), 1),
                 "gammas": gams}
