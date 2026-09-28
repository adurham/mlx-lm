# LOCAL ADDITION. Sampling (temperature / top-p / top-k) for the DSv4.1 spec loop.
"""Seeded speculative sampling for the DSpark decode loop.

The greedy loop in :mod:`spec` commits ``argmax`` targets and ``argmax`` drafts;
this module adds temperature / top-p / top-k sampling to the *same* loop with
**proper speculative sampling**: the draft token is accepted with probability
``min(1, p_target / p_draft)``, a rejection resamples from the normalised
residual ``(p_target - p_draft)+``, and a fully accepted round takes the bonus
token from the target distribution itself.  That construction emits tokens
distributed *exactly* as the target model's filtered distribution while keeping
the ``gamma + 1`` verify rows in one body forward; the greedy path (temperature
<= 0) is untouched and does not consume RNG.

Three things this module has to solve that plain single-stream sampling does
not:

1. **Full-vocab probabilities under a vocab-sharded head.**  The body head is
   ``exl3_build.ShardedHead`` (TP=2: each rank owns half of the vocab row) and
   the draft's markov head is sliced the same way.  ``argmax`` needs only a
   ``(max, index)`` pair per rank, but acceptance and the residual need the
   *whole* normalised row.  So the gather (pad + ``all_sum``, the same geometry
   ``ShardedHead.__call__`` uses) happens only on the sampling path:
   * the body rows come back full because the verify forward is called with
     ``argmax=False`` (``ShardedHead.__call__`` already rebuilds the full row);
   * the draft's per-step rows are gathered by :class:`DraftProbe`.

2. **The draft's token *and* its distribution.**  ``DSparkHead.draft`` picks its
   tokens with ``argmax``/``combine_argmax`` internally, so the sampling path
   passes a :class:`DraftProbe` in place of the body head.  The probe keeps
   ``DSparkHead.draft``'s call pattern intact (it answers ``local`` and
   ``combine_argmax``) but draws each draft token from the *filtered* draft
   distribution and records that distribution per position, which is exactly
   what the acceptance ratio needs.  No change to ``mtp.py``.

3. **One host sync per round.**  All per-step draws stay lazy inside the verify
   graph; the round does a single ``mx.eval`` (verify logits + draft tokens +
   the recorded draft distributions + the round's uniforms) and only then runs
   the accept/reject arithmetic on the host in fp32 numpy.

RNG: one stream per ``generate`` call, split once per round into a fixed-size
uniform pool (``2 * gamma + 2`` values: one per draft step, ``gamma`` acceptance
draws, one residual draw, one bonus draw) so the draw sequence never depends on
which branch a round takes.  ``mlx.core``'s RNG is key-based and deterministic
across processes and ranks, which is what keeps a TP=2 run's tokens identical on
both ranks; the fixture tests can run the same code with numpy arrays
(``backend="numpy"``), which is also the gateway-without-MLX path.

Mask order matches ``mlx_lm.sample_utils`` exactly (top-p then top-k, on
*untempered* log-probabilities; the temperature is applied to the final draw),
so a sampled stream here is the same distribution as the rest of the stack.
"""

from __future__ import annotations

import math

import numpy as np

try:  # MLX is optional: the fixture tests below/CI may run without it.
    import mlx.core as mx
except Exception:  # pragma: no cover - import guard only
    mx = None

DEFAULT_SEED = 20260928
_NEG_INF = -math.inf


def _xp(x):
    """The array module (mlx.core or numpy) that owns ``x``."""
    if mx is not None and isinstance(x, mx.array):
        return mx
    if isinstance(x, np.ndarray):
        return np
    raise TypeError(f"expected an mlx or numpy array, got {type(x).__name__}")


def _full_like(x, value):
    xp = _xp(x)
    return xp.full(x.shape, value, dtype=x.dtype)


def _put_along_axis(a, idx, value):
    """Functional-ish scatter of a scalar ``value`` at ``idx`` along the last axis."""
    xp = _xp(a)
    fill = xp.full(idx.shape, value, dtype=a.dtype)
    if xp is np:
        out = np.array(a, copy=True)
        np.put_along_axis(out, idx, fill, -1)
        return out
    return mx.put_along_axis(a, idx, fill, -1)


def softmax(x):
    xp = _xp(x)
    m = xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(x - m)
    return e / xp.sum(e, axis=-1, keepdims=True)


def log_softmax(x):
    xp = _xp(x)
    m = xp.max(x, axis=-1, keepdims=True)
    y = x - m
    return y - xp.log(xp.sum(xp.exp(y), axis=-1, keepdims=True))


# --------------------------------------------------------------------------- #
# The sampling transform (mlx-lm compatible)
# --------------------------------------------------------------------------- #
def filtered_logprobs(logits, top_p: float = 1.0, top_k: int = 0):
    """Masked log-probabilities, exactly as ``mlx_lm.sample_utils`` would.

    ``top_p`` then ``top_k``, both on the *untempered* distribution (that is the
    order and the space ``make_sampler`` uses: it filters log-probabilities and
    only then divides by the temperature inside ``categorical_sampling``).
    """
    xp = _xp(logits)
    lp = log_softmax(logits)
    if 0.0 < float(top_p) < 1.0 and lp.shape[-1] > 1:
        probs = xp.exp(lp)
        order = xp.argsort(lp, axis=-1)                       # ascending
        sp = xp.take_along_axis(probs, order, axis=-1)
        cum = xp.cumsum(sp, axis=-1)
        keep_sorted = cum > (1.0 - float(top_p))
        # scatter the "keep" mask back into vocabulary order
        keep = _put_along_axis(xp.zeros(keep_sorted.shape, dtype=xp.bool_), order, keep_sorted)
        lp = xp.where(keep, lp, _full_like(lp, _NEG_INF))
    top_k = int(top_k)
    v = lp.shape[-1]
    if 0 < top_k < v:
        idx = xp.argpartition(-lp, kth=top_k - 1, axis=-1)[..., top_k:]
        lp = _put_along_axis(lp, idx, _NEG_INF)
    return lp


def filtered_probs(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    """The target distribution: filtered, then sharpened by ``temperature``.

    ``temperature`` is applied to the final draw only (mlx-lm semantics), so
    ``filtered_probs(x, t)`` is the distribution ``make_sampler(t, top_p, top_k)``
    samples from.
    """
    return softmax(filtered_logprobs(logits, top_p=top_p, top_k=top_k) / float(temperature))


def residual_probs(p, q):
    """``normalize((p - q)+)``, with ``p`` itself as the degenerate fallback."""
    xp = _xp(p)
    r = xp.maximum(p - q, 0)
    s = xp.sum(r, axis=-1, keepdims=True)
    safe = xp.where(s > 0, s, 1.0)
    out = r / safe
    return xp.where(s > 0, out, p)


def acceptance_prob(p_x: float, q_x: float) -> float:
    """``min(1, p(x) / q(x))``; a token the drafter could not have emitted is
    accepted outright (``q`` is the draft's own distribution, so ``q_x > 0``)."""
    if q_x <= 0.0:
        return 1.0
    return min(1.0, float(p_x) / float(q_x))


def sample_index(probs, u):
    """Inverse-CDF draw: index of the first cumulative mass strictly above ``u``.

    ``argmax`` over the boolean ``cumsum > u`` rather than a count, so that
    plateaus from zero-probability (filtered) tokens cannot overshoot into a
    masked token.  Works on mlx and numpy arrays.
    """
    xp = _xp(probs)
    c = xp.cumsum(probs, axis=-1)
    c = c / xp.maximum(c[..., -1:], xp.array(1e-30, dtype=c.dtype))
    gt = (c > u).astype(xp.int8)
    return xp.argmax(gt, axis=-1).astype(xp.int32)


def _as_int(x) -> int:
    a = np.asarray(x)
    return int(a.reshape(-1)[0])


class _Device:
    """Array-module facade for the loop's device-side ops."""

    def __init__(self, backend: str):
        if backend == "mlx" and mx is None:
            raise RuntimeError("backend='mlx' requested but mlx.core is not importable")
        self.backend = backend
        self.xp = mx if backend == "mlx" else np
        self.mlx = backend == "mlx"

    def array(self, x, dtype=None):
        return self.xp.array(x, dtype=dtype)

    def concat(self, xs, axis=0):
        return self.xp.concatenate(xs, axis=axis)

    def stack(self, xs, axis=0):
        return self.xp.stack(xs, axis=axis)

    def eval(self, *xs):
        if self.mlx:
            mx.eval(*[x for x in xs if isinstance(x, mx.array)])
        else:
            for x in xs:
                if isinstance(x, np.ndarray):
                    _ = x.shape

    def host(self, x):
        return np.asarray(x)

    @property
    def int32(self):
        return self.xp.int32


# --------------------------------------------------------------------------- #
# RNG
# --------------------------------------------------------------------------- #
class Sampler:
    """Sampling parameters plus one seeded RNG stream per generation.

    ``backend="numpy"`` exists so the fixture tests (and any host without MLX)
    can drive the same code path with numpy arrays; production uses the mlx
    default.  The uniform pool is fixed-size per round, so a round's draws do
    not depend on which branch the round takes.
    """

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
                 seed: int = DEFAULT_SEED, backend: str | None = None):
        temperature = float(temperature)
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0 (greedy is temperature <= 0), got {temperature}")
        top_p = float(top_p)
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        top_k = int(top_k)
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.seed = int(seed)
        self.backend = backend or ("mlx" if mx is not None else "numpy")
        self.dev = _Device(self.backend)
        self.round = 0
        self.pool = None
        self._key = mx.random.key(self.seed) if self.backend == "mlx" else None

    # -- per-round uniforms ---------------------------------------------------
    def begin_round(self, n_uniforms: int):
        n = int(n_uniforms)
        if self.backend == "mlx":
            keys = mx.random.split(self._key, 2)
            self._key = keys[1]
            self.pool = mx.random.uniform(shape=(n,), key=keys[0])
        else:
            rng = np.random.default_rng([self.seed, self.round])
            self.pool = np.asarray(rng.random(n), dtype=np.float32)
        self.round += 1
        return self.pool

    def uniform(self, i: int):
        return self.pool[i]

    def host_pool(self) -> np.ndarray:
        return np.asarray(self.pool)

    def probs(self, logits):
        return filtered_probs(logits, self.temperature, self.top_p, self.top_k)


# --------------------------------------------------------------------------- #
# Draft side
# --------------------------------------------------------------------------- #
class DraftProbe:
    """Sampling stand-in for the body head inside ``DSparkHead.draft``.

    ``draft()`` calls ``head.local(norm(x))`` once for the block logits and
    ``head.combine_argmax(step_logits)`` for every markov step.  Handed this
    object instead, the second call is taken over: the probe gathers the full
    vocab row (pad + ``all_sum`` on the sharded-head geometry, only when the
    sampling path is active), applies the sampling transform, draws the draft
    token from it and records that per-step distribution for the acceptance
    ratio.
    """

    def __init__(self, model, sampler: Sampler, width: int):
        self.sampler = sampler
        self.dev = sampler.dev
        body = getattr(model, "head", None)
        vocab = int(model.args.vocab_size)
        self.world = int(getattr(body, "_world", 1) or 1)
        self.lo = int(getattr(body, "_lo", 0) or 0)
        self.vocab = int(getattr(body, "_vocab", vocab) or vocab)
        self.group = getattr(body, "_group", None)
        self._local = getattr(body, "local", None)
        self._call = body if callable(body) else None
        self.width = int(width)
        self.qs = []
        self._k = 0

    def begin_round(self, width: int):
        """Start a round's uniform pool: ``[0, 2*width+2)``.

        Slots ``[0, width)`` are the per-step draft draws, ``[width, 2*width)``
        the acceptance tests, then the residual and bonus draws.  The pool size
        is fixed by ``width`` so a round's draw sequence never depends on which
        branch it takes; the two halves never overlap (see ``spec_generate``).
        """
        self.width = int(width)
        self.qs = []
        self._k = 0

    # -- what ``DSparkHead.draft`` calls --------------------------------------
    def local(self, h):
        if self._local is not None:
            return self._local(h)
        if self._call is not None:
            return self._call(h)
        raise TypeError("draft probe needs a body head with .local() or __call__")

    def combine_argmax(self, y):
        full = self.gather(y.astype(self.dev.xp.float32))
        q = self.sampler.probs(full)
        self.qs.append(q)
        idx = sample_index(q, self.sampler.uniform(self._k))
        self._k += 1
        return idx.astype(self.dev.xp.int32).reshape(full.shape[:-1])

    # -- geometry -------------------------------------------------------------
    def gather(self, y):
        """Rank-local row -> full-vocab row (identity when not sharded)."""
        w = y.shape[-1]
        if self.world <= 1 or self.group is None:
            if w != self.vocab:
                raise ValueError(f"local head width {w} != vocab {self.vocab} (world={self.world})")
            return y
        if w * self.world != self.vocab:
            raise ValueError(f"local head width {w} x world {self.world} != vocab {self.vocab}")
        pad = [(0, 0)] * (y.ndim - 1) + [(self.lo, self.vocab - self.lo - w)]
        return mx.distributed.all_sum(mx.pad(y, pad), group=self.group)


def draft_geometry(model, head):
    """Check the draft head can be sampled from, force the probe path on.

    ``DSparkHead.draft`` takes its sampling route only when
    ``head.vocab_sharded`` is set and the object passed in answers
    ``combine_argmax``; the probe answers both, so the flag is set here after
    checking the markov head's width is consistent with the body head's
    sharding.  Returns ``(vocab, world)``.
    """
    body = getattr(model, "head", None)
    vocab = int(model.args.vocab_size)
    world = int(getattr(body, "_world", 1) or 1)
    mh = getattr(head, "markov_head", None)
    weight = getattr(mh, "weight", None)
    w = None if weight is None else int(weight.shape[0])
    if world > 1:
        # the draft's base logits come from the *body* head's rank slice, so the
        # markov head must be sharded the same way or the two cannot be added
        want = vocab // world
        if w is not None and w != want:
            raise RuntimeError(
                f"draft markov width {w} != vocab//world {want}: sampling needs the "
                "draft head sharded like the body head (DSV41_TP_HEAD=1, DSV41_DRAFT_SHARD=1)")
    elif w is not None and w != vocab:
        raise RuntimeError(f"draft markov width {w} != vocab {vocab}")
    head.vocab_sharded = True
    return vocab, world


# --------------------------------------------------------------------------- #
# Verify step (host math)
# --------------------------------------------------------------------------- #
def verify_round(p_rows, q_rows, draft, u_acc, u_res, u_bonus):
    """Accept/reject one drafted block against the target's filtered rows.

    ``p_rows``   ``[gamma + 1, V]`` target probabilities (row ``k`` verifies draft ``k``).
    ``q_rows``   ``[gamma, V]``     the distributions the draft drew from.
    ``draft``    ``[gamma]``        drafted token ids.
    ``u_acc``    ``[gamma]``        one uniform per acceptance test.
    ``u_res``    scalar             uniform for the residual resample.
    ``u_bonus``  scalar             uniform for the bonus token.

    Returns ``(accepted, next_token, n_accepted, rejected)`` where
    ``accepted`` is the committed prefix of the block (length ``n_accepted``)
    and ``next_token`` is the residual resample (a rejection) or the target's
    bonus draw (all accepted) that is committed after it.
    """
    g = len(p_rows) - 1
    accepted = []
    for k in range(g):
        pk, qk = p_rows[k], q_rows[k]
        dk = int(draft[k])
        a = acceptance_prob(pk[dk], qk[dk])
        if float(u_acc[k]) < a:
            accepted.append(dk)
            continue
        r = residual_probs(pk, qk)
        t = _as_int(sample_index(r, float(u_res)))
        return accepted, t, len(accepted), True
    t = _as_int(sample_index(p_rows[g], float(u_bonus)))
    return accepted, t, g, False


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def _spec_module():
    from . import spec as _spec

    return _spec


def _assert_backend(name: str, x, dev: _Device) -> None:
    """Fail loudly when an array and the sampler's backend disagree.

    Easy to do by accident: ``Sampler()`` defaults to MLX whenever mlx.core is
    importable, so a numpy fixture fed to the default sampler looks fine until
    the first device-side concat fails with a confusing type error.
    """
    xp = _xp(x)                                   # raises TypeError for others
    if (xp is mx) != dev.mlx:
        raise TypeError(
            f"{name} is a {xp.__name__} array but the sampler backend is "
            f"{dev.backend!r}; build the Sampler with backend='numpy' or with "
            f"the backend matching the arrays' module")


def spec_generate(model, head, prompt_ids, max_new: int, *, gamma: int = 3,
                  adaptive: bool = True, eos_id: int = 1, temperature: float = 1.0,
                  top_p: float = 1.0, top_k: int = 0, seed: int = DEFAULT_SEED,
                  sampler: Sampler | None = None, policy=None, sp=None):
    """Sampling speculative decode with one host sync per round.

    Mirrors ``spec.generate``'s scaffolding (snap / rollback / tap feed, same
    ``stats`` keys) but every committed token comes from the target's filtered
    distribution: accepted draft tokens, a residual resample on rejection, or
    the target's own bonus draw when the whole block is accepted.  ``sp``
    defaults to the real :mod:`spec` module; the fixture tests pass a stub with
    the three cache primitives.
    """
    import time

    sp = sp if sp is not None else _spec_module()
    sampler = sampler or Sampler(temperature, top_p, top_k, seed)
    dev = sampler.dev
    draft_geometry(model, head)

    taps_ids = list(model.args.dspark_target_layer_ids)

    def tapcat(t):
        return dev.concat([t[L] for L in taps_ids], axis=-1)

    cache = model.make_cache(1, max_seq_len=len(prompt_ids) + max_new + 16)
    logits, taps = model(dev.array([list(prompt_ids)], dtype=dev.int32), cache,
                         last_logit_only=True, return_taps=True)
    _assert_backend("model logits", logits, dev)
    dsc = head.make_cache(1)
    head.append_ctx(tapcat(taps), dsc)
    # first generated token: drawn from the target's distribution at the prompt
    sampler.begin_round(1)
    p0 = sampler.probs(dev.host(logits)[0])                     # [1, V] host fp32
    tok = _as_int(sample_index(p0[0], float(sampler.host_pool()[0])))
    out = [int(tok)]
    pos = cache.offset
    probe = DraftProbe(model, sampler, gamma)
    pol = policy
    if pol is None and adaptive:
        pol = sp.GammaPolicy(start=gamma)
    hist, gams = [], []
    n_rejects, n_drafted = 0, 0
    t0 = time.perf_counter()
    while len(out) < max_new + 1 and out[-1] != eos_id:
        g = pol.next() if pol is not None else gamma
        sampler.begin_round(2 * g + 2)
        probe.begin_round(g)
        nxt = dev.array([[int(out[-1])]], dtype=dev.int32)
        d, _ = head.draft(nxt[:, 0], model.embed, probe, dsc, width=g)
        vin = dev.concat([nxt, d], axis=1)
        sn = sp.snap(cache, pos)
        lg, taps = model(vin, cache, return_taps=True)          # fp32 logits [1, g+1, V]
        dev.eval(lg, d, sampler.pool, *probe.qs)                        # mx.eval: one sync
        p_rows = sampler.probs(dev.host(lg)[0])                         # [g+1, V] host fp32
        q_rows = [dev.host(q)[0] for q in probe.qs]                     # gamma x [V]
        drafted = dev.host(d)[0]
        u = sampler.host_pool()
        # slot layout: [0, g) draft draws, [g, 2g) acceptance tests, 2g residual,
        # 2g+1 bonus.  The acceptance draws MUST be disjoint from the draft
        # draws: sharing a uniform makes "which token was drawn" and "was it
        # accepted" dependent, which biases the emitted distribution away from p.
        acc, bonus, n, rejected = verify_round(p_rows, q_rows, drafted, u[g:2 * g], u[2 * g], u[2 * g + 1])
        n_rejects += int(rejected)
        n_drafted += g
        if pol is not None:
            pol.update(g, n)
        hist.append(n)
        gams.append(g)
        new = [int(v) for v in acc] + [int(bonus)]
        target = pos + n + 1
        sp.rollback(cache, sn, target, sp.stashes(cache))
        head.append_ctx(tapcat(taps)[:, : n + 1], dsc)
        pos = target
        for t in new:
            out.append(int(t))
            if int(t) == eos_id:
                break
    dt = time.perf_counter() - t0
    return out, {"tok_s": (len(out) - 1) / dt, "rounds": len(hist),
                 "mean_acc": float(np.mean(hist)) if hist else 0.0,
                 "ms_round": dt * 1e3 / max(len(hist), 1),
                 "gammas": gams, "rejects": n_rejects, "drafted": n_drafted,
                 "accept_rate": (1.0 - n_rejects / max(len(hist), 1)) if hist else 0.0,
                 "temperature": sampler.temperature, "top_p": sampler.top_p,
                 "top_k": sampler.top_k, "seed": sampler.seed,
                 "backend": sampler.backend}
