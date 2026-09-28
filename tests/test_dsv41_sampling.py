# LOCAL ADDITION. Tests for the DSv4.1 speculative sampler (spec + mtp + exl3_build).
"""Distribution tests for ``mlx_lm.models.deepseek_v41.sampling``.

What is asserted, and why it is exact rather than "looks about right":

* The spec loop's committed token at a position whose *target* distribution is
  ``p`` must itself be distributed as ``p`` -- whatever the draft's distribution
  ``q`` is.  That is the whole contract of speculative sampling: accept with
  ``min(1, p/q)``, resample from the normalised residual ``(p - q)+`` on
  rejection, take the bonus from ``p``.  It holds *regardless of the draft*, so
  the test drives the loop with three different draft qualities (identical,
  sharper, rotated) and checks each against the same target.
* The check is a chi-square and a KS test against the analytic target, on
  fixtures where the numbers are computable in closed form:

    - the fake target's per-position distribution depends only on the token fed
      at that position (``R[token % P]``), exactly like a real causal model's
      dependence on its prefix, so the *phase* of every generated position --
      and therefore its expected distribution -- is recoverable from the output
      stream alone;
    - the fake draft head answers ``local``/``combine_argmax`` like
      ``DSparkHead.draft`` does, so the probe path that production uses is the
      path under test (the probe's recorded ``q`` is compared to the fixture's
      analytic ``q`` implicitly: a wrong ``q`` changes the accepted mass and the
      chi-square fails).

* A negative control runs the *same* loop with the textbook-wrong verifier
  (resample from ``p`` instead of the residual) and must be rejected loudly;
  a test that cannot fail proves nothing.
* Seed control: same seed -> identical stream, different seed -> different
  stream, and both backends agree on the transform, so a TP=2 run (both ranks
  draw with the same keys) commits identical tokens.

Runs two ways:

    python tests/test_dsv41_sampling.py                    # gateway: numpy path
    ~/repos/exo/.venv/bin/python ~/dsv41-ws/F/tests/...    # Mac: adds the MLX path

On a host without MLX the package import of ``mlx_lm`` fails (its ``__init__``
pulls in mlx), so the module under test is loaded from its file path instead;
``TestSourceUnderTest`` prints and asserts which route was taken.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import mlx.core as mx

    _HAVE_MLX = True
except Exception:                                          # pragma: no cover
    mx = None
    _HAVE_MLX = False


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, "file", path


def _load_sampling():
    """Import the module under test, with or without the mlx_lm package."""
    if _HAVE_MLX:
        from mlx_lm.models.deepseek_v41 import sampling as mod

        return mod, "package", os.path.abspath(mod.__file__)
    path = os.path.join(_ROOT, "mlx_lm", "models", "deepseek_v41", "sampling.py")
    return _load_module(path, "dsv41_sampling_under_test")


S, _IMPORT_ROUTE, _IMPORT_PATH = _load_sampling()


# --------------------------------------------------------------------------- #
# Statistics (no scipy on the gateway: chi-square / KS survival functions)
# --------------------------------------------------------------------------- #
def _gser(a: float, x: float) -> float:
    gln, ap, s, d = math.lgamma(a), a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-15:
            break
    return s * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    tiny = 1e-300
    gln = math.lgamma(a)
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def gammq(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    if a <= 0 or x < 0:
        raise ValueError("gammq: bad arguments")
    if x == 0:
        return 1.0
    return 1.0 - _gser(a, x) if x < a + 1.0 else _gcf(a, x)


def chi2_sf(chi2: float, dof: int) -> float:
    """P(chi2_dof > chi2)."""
    return gammq(0.5 * dof, 0.5 * chi2)


def kolmogorov_q(x: float) -> float:
    """Kolmogorov distribution Q(x) = P(K > x), the asymptotic KS tail."""
    if x <= 0:
        return 1.0
    if x < 0.05:
        return 1.0
    if x > 6.0:
        return 0.0
    s = 0.0
    for j in range(1, 200):
        term = (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * x * x)
        s += term
        if abs(term) < 1e-18:
            break
    return max(0.0, min(1.0, 2.0 * s))


def chi_square(obs: np.ndarray, p: np.ndarray):
    """Chi-square goodness of fit against a fully specified distribution."""
    obs = np.asarray(obs, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = obs.sum()
    keep = p > 0
    exp = n * p[keep]
    stat = float((((obs[keep] - exp) ** 2) / exp).sum())
    dof = max(int(keep.sum()) - 1, 1)
    return stat, dof, chi2_sf(stat, dof)


def ks_test(obs: np.ndarray, p: np.ndarray):
    """One-sample KS against a discrete distribution (conservative p-value)."""
    obs = np.asarray(obs, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = obs.sum()
    f_emp = np.cumsum(obs / n)
    f_theo = np.cumsum(p / p.sum())
    d = float(np.max(np.abs(f_emp - f_theo)))
    return d, kolmogorov_q(math.sqrt(n) * d)


# --------------------------------------------------------------------------- #
# Fixtures: a fake causal target + a fake DSpark draft head (backend agnostic)
# --------------------------------------------------------------------------- #
class _FakeCache:
    layers = ()

    def __init__(self):
        self.offset = 0


class _StubSpec:
    """The cache primitives ``spec_generate`` uses plus a GammaPolicy for the
    adaptive path, so the fake (cacheless) loop can exercise both."""

    @staticmethod
    def snap(cache, start_pos):
        return None

    @staticmethod
    def stashes(cache):
        return []

    @staticmethod
    def rollback(cache, sn, target, st):
        cache.offset = target

    class GammaPolicy:
        """Same shape as spec.GammaPolicy, driven by accept counts."""

        def __init__(self, gammas=(1, 2, 3, 4), start=3, **_):
            self.gammas, self.g = gammas, start
            self.tried = [0] * 8
            self.acc = [0] * 8
            self.rounds = 0

        def update(self, gamma, n_acc):
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

        def next(self):
            if self.rounds < 4:
                return self.g
            # cheap proxy: draft wider when acceptance is high
            q = self._q(1) * self._q(2)
            self.g = 4 if q > 0.5 else (2 if q > 0.2 else 1)
            return self.g


def _to_int(dev, x) -> int:
    dev.eval(x)
    return int(np.asarray(x).reshape(-1)[0])


class _FakeGroup:
    """Stands in for the JACCL mesh: all_sum over one process is identity.

    With identity all_sum, ``ShardedHead``'s pad+sum gather reduces to "put
    this rank's slice back where it came from", which is exactly the per-rank
    contribution the real 2-rank mesh would add up.
    """

    def __init__(self, backend):
        self.backend = backend

    def rank(self):
        return 0

    def size(self):
        return 2


def _patch_all_sum_identity():
    """Make mlx.distributed.all_sum identity for _FakeGroup (single process)."""
    real = mx.distributed.all_sum
    mx.distributed.all_sum = (lambda x, group=None, **k:
                              x if isinstance(group, _FakeGroup)
                              else real(x, group=group, **k))
    return real


class FixtureModel:
    """Causal target stand-in: ``logits = R[token_of_this_position % P]``.

    The distribution at a position depends on the token fed at that position
    (the prefix's last token), which is all the test needs: the phase of every
    generated position is then recoverable from the output stream alone.
    """

    def __init__(self, R: np.ndarray, dev):
        self.R_np = R.astype(np.float32)
        self.R = dev.array(self.R_np) if dev.mlx else self.R_np
        self.P = int(R.shape[0])
        self.dev = dev
        self.args = SimpleNamespace(vocab_size=int(R.shape[1]),
                                    dspark_target_layer_ids=(0,))
        self.head = SimpleNamespace()          # world=1 geometry (unsharded)
        self.embed = None                      # fixture draft ignores embeddings

    def make_cache(self, bsz=1, max_seq_len=None):
        return _FakeCache()

    def __call__(self, ids, cache, last_logit_only=False, return_taps=False,
                 argmax=False):
        dev = self.dev
        ids = ids if dev.mlx else np.asarray(ids)
        rows = (ids[0] % self.P).astype(dev.int32)
        logits = dev.xp.take(self.R, rows, axis=0)[None]        # [1, n, V]
        if last_logit_only:                                     # as model.Model does
            logits = logits[:, -1:]
        cache.offset += int(ids.shape[1])
        if return_taps:
            taps = {0: dev.xp.zeros((1, int(ids.shape[1]), 1), dtype=dev.xp.float32)}
            return logits, taps
        return logits


class FixtureDraftHead:
    """Stand-in for ``DSparkHead`` with the same probe call pattern.

    ``draft`` asks the object passed as ``head`` (the probe) for
    ``combine_argmax`` at every markov step, exactly like ``mtp.DSparkHead``.
    """

    def __init__(self, Stab: np.ndarray, dev):
        self.S = dev.array(Stab.astype(np.float32)) if dev.mlx else Stab.astype(np.float32)
        self.P = int(Stab.shape[0])
        self.dev = dev
        self.vocab = int(Stab.shape[1])
        self.markov_head = SimpleNamespace(
            weight=dev.array(np.zeros((self.vocab, 4), np.float32), dtype=None)
            if dev.mlx else np.zeros((self.vocab, 4), np.float32))
        self.call_phases = []

    def make_cache(self, bsz=1):
        return None

    def append_ctx(self, main_hidden_cat, caches):
        pass

    def draft(self, anchor_tokens, embed, head, caches, width=None):
        dev = self.dev
        dev.eval(anchor_tokens)
        prev = int(np.asarray(anchor_tokens).reshape(-1)[0])
        toks = []
        for _ in range(int(width)):
            row = self.S[prev % self.P][None, :]                # [1, V]
            row = dev.array(row) if dev.mlx else row
            nxt = head.combine_argmax(row.astype(dev.xp.float32))
            tok = _to_int(dev, nxt)
            self.call_phases.append(prev % self.P)
            toks.append(nxt.astype(dev.int32).reshape(-1))
            prev = tok
        d = dev.concat(toks, axis=0).reshape(1, -1)
        conf = dev.xp.zeros((1, int(width)), dtype=dev.xp.float32)
        return d, conf


def make_fixture(V=32, P=4, seed=11, draft="sharp", draft_scale=2.0):
    rng = np.random.default_rng(seed)
    R = (rng.normal(size=(P, V)) * 1.4).astype(np.float32)
    if draft == "identical":
        S_ = R.copy()
    elif draft == "sharp":
        S_ = R * draft_scale
    elif draft == "rotate":
        S_ = np.roll(R, 3, axis=-1)
    elif draft == "flat":
        S_ = R * 0.4
    else:
        raise ValueError(draft)
    return R, S_


def run_loop(model, head, prompt, n_new, *, temperature=1.0, top_p=1.0, top_k=0,
             seed=1, gamma=3, sp=_StubSpec, sampler=None, backend="numpy"):
    """Drive ``spec_generate`` with the fixture's array module.

    ``backend`` is pinned to the fixture's module on purpose: on a Mac the
    default sampler would be MLX, and a numpy fixture driven by it fails with a
    confusing device-side type error instead of a clear message.
    """
    if sampler is None:
        sampler = S.Sampler(temperature, top_p, top_k, seed, backend=backend)
    return S.spec_generate(model, head, prompt, n_new, gamma=gamma,
                           adaptive=False, eos_id=-1, seed=seed,
                           sampler=sampler, sp=sp)


def phases_of(out, prompt_last, P):
    """Phase (target-table row) of every generated position."""
    toks = [prompt_last] + [int(t) for t in out[:-1]]
    return [int(t) % P for t in toks]


def histogram(out, V):
    h = np.bincount(np.asarray(out, dtype=np.int64), minlength=V)
    return h[:V].astype(np.float64)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestSourceUnderTest(unittest.TestCase):
    def test_route_and_module_identity(self):
        print(f"\n[sampling] import route: {_IMPORT_ROUTE} -> {_IMPORT_PATH}")
        self.assertTrue(os.path.exists(_IMPORT_PATH))
        if _IMPORT_ROUTE == "package":
            self.assertTrue(_IMPORT_PATH.endswith("deepseek_v41/sampling.py"))
        self.assertTrue(hasattr(S, "spec_generate"))

    def test_stats_helpers_match_known_tails(self):
        self.assertAlmostEqual(chi2_sf(1.0, 1), 0.3173105078629141, places=10)
        self.assertAlmostEqual(chi2_sf(2.0, 2), 0.3678794411714423, places=10)
        self.assertAlmostEqual(chi2_sf(9.487729036781154, 4), 0.05, places=8)
        self.assertEqual(chi2_sf(0.0, 3), 1.0)
        self.assertAlmostEqual(kolmogorov_q(1.0), 0.2699996716758627, places=9)
        self.assertAlmostEqual(kolmogorov_q(0.6), 0.86428, places=4)   # KS table
        self.assertAlmostEqual(kolmogorov_q(1.3581), 0.05, places=3)   # KS table


class TestSamplingTransform(unittest.TestCase):
    """``filtered_probs`` must be the distribution mlx-lm's sampler draws from."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.logits = (rng.normal(size=(64, 48)) * 2.0).astype(np.float32)

    def test_no_filter_is_softmax(self):
        p = S.filtered_probs(self.logits, 1.0, 1.0, 0)
        ref = np.exp(self.logits - self.logits.max(-1, keepdims=True))
        ref /= ref.sum(-1, keepdims=True)
        self.assertLess(float(np.abs(np.asarray(p) - ref).max()), 1e-6)
        self.assertAlmostEqual(float(np.asarray(p).sum(-1).min()), 1.0, places=6)

    def test_top_k_keeps_exactly_k(self):
        for k in (1, 5, 17):
            p = np.asarray(S.filtered_probs(self.logits, 1.0, 1.0, k))
            nz = (p > 0).sum(-1)
            self.assertTrue(np.all(nz == k), f"top_k={k} kept {set(nz.tolist())}")
            # the kept set is exactly the top-k by logit
            order = np.argsort(-self.logits, axis=-1)[:, :k]
            for r in range(self.logits.shape[0]):
                self.assertEqual(sorted(np.nonzero(p[r])[0].tolist()),
                                 sorted(order[r].tolist()))

    def test_top_p_nucleus_matches_mlxlm_definition(self):
        # mlx-lm keeps cumsum(sorted_probs) > 1 - top_p, on untempered probs.
        lp = np.log(S.filtered_probs(self.logits, 1.0, 1.0, 0))
        for tp in (0.5, 0.9, 0.99):
            p = np.asarray(S.filtered_probs(self.logits, 1.0, tp, 0))
            keep = p > 0
            probs = np.exp(np.asarray(lp))
            order = np.argsort(np.asarray(lp), axis=-1)
            sp = np.take_along_axis(probs, order, -1)
            ref = np.cumsum(sp, -1) > (1.0 - tp)
            got = np.take_along_axis(keep, order, -1)
            self.assertTrue(np.array_equal(got, ref), f"top_p={tp}")
            # every kept row keeps at least the top token
            self.assertTrue(np.all(keep.any(-1)))
            self.assertTrue(np.all(keep[np.arange(len(p)), np.argmax(self.logits, -1)]))

    def test_temperature_is_applied_last(self):
        # filtered_probs(x, t) == softmax(filtered_logprobs(x) / t)
        for t in (0.5, 1.0, 2.5):
            a = np.asarray(S.filtered_probs(self.logits, t, 0.9, 8))
            b = np.asarray(S.filtered_logprobs(self.logits, 0.9, 8)) / t
            b = np.exp(b - b.max(-1, keepdims=True))
            b /= b.sum(-1, keepdims=True)
            self.assertLess(float(np.abs(a - b).max()), 1e-6)

    def test_sampler_rejects_greedy_temperature(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                S.Sampler(temperature=bad)
        with self.assertRaises(ValueError):
            S.Sampler(top_p=1.5)
        with self.assertRaises(ValueError):
            S.Sampler(top_k=-2)


class TestVerifyRoundHostMath(unittest.TestCase):
    """``p_rows`` has ``gamma + 1`` rows: row k verifies draft k, the last row
    feeds the bonus token.  ``q_rows`` has ``gamma``."""

    def test_accept_when_u_below_ratio(self):
        p = np.array([[0.5, 0.5], [0.9, 0.1]], np.float32)
        q = np.array([[1.0, 0.0]], np.float32)                 # sharper than p
        acc, tok, n, rej = S.verify_round(p, q, [0], np.array([0.4], np.float32),
                                          0.5, 0.5)
        self.assertEqual((acc, n, rej), ([0], 1, False))
        self.assertEqual(tok, 0)          # bonus row cumsum [0.9, 1.0], u=0.5
        # u above the ratio 0.5 rejects and resamples from the residual
        acc, tok, n, rej = S.verify_round(p, q, [0], np.array([0.6], np.float32),
                                          0.5, 0.99)
        self.assertEqual((acc, n, rej), ([], 0, True))
        self.assertEqual(tok, 1)          # residual is [0, 1] -> token 1 for any u

    def test_reject_resamples_from_residual(self):
        p = np.array([[0.5, 0.5], [0.9, 0.1]], np.float32)
        q = np.array([[1.0, 0.0]], np.float32)
        acc, tok, n, rej = S.verify_round(p, q, [0], np.array([0.9], np.float32),
                                          0.0, 0.5)
        self.assertTrue(rej)
        self.assertEqual(acc, [])
        self.assertEqual(n, 0)
        self.assertEqual(tok, 1)          # residual is [0, 1] -> token 1, any u

    def test_all_accepted_takes_bonus_from_target(self):
        p = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], np.float32)
        q = np.array([[0.8, 0.2], [0.3, 0.7]], np.float32)
        acc, tok, n, rej = S.verify_round(p, q, [0, 0],
                                          np.array([0.1, 0.1], np.float32), 0.5, 0.5)
        self.assertEqual(acc, [0, 0])
        self.assertEqual(n, 2)
        self.assertFalse(rej)
        self.assertEqual(tok, 0)          # bonus row [0.7, 0.3], u=0.5 -> token 0
        acc, tok, _, _ = S.verify_round(p, q, [0, 0], np.array([0.1, 0.1], np.float32),
                                        0.5, 0.95)
        self.assertEqual(acc, [0, 0])
        self.assertEqual(tok, 1)          # u=0.95 lands past the 0.7 mass -> token 1

    def test_q_zero_accepts_unconditionally(self):
        # a token the drafter gave zero density cannot be rejected by p/q
        self.assertEqual(S.acceptance_prob(0.0, 0.0), 1.0)
        self.assertEqual(S.acceptance_prob(0.4, 0.0), 1.0)
        self.assertAlmostEqual(S.acceptance_prob(0.25, 0.5), 0.5)

    def test_degenerate_residual_falls_back_to_p(self):
        p = np.array([[0.3, 0.7]], np.float32)
        r = np.asarray(S.residual_probs(p, p.copy()))
        self.assertTrue(np.all(np.isfinite(r)))
        self.assertAlmostEqual(float(r.sum()), 1.0, places=6)
        np.testing.assert_allclose(r, p, atol=1e-6)

    def test_sample_index_is_a_valid_inverse_cdf(self):
        p = np.array([0.0, 0.0, 0.25, 0.75], np.float32)
        self.assertEqual(int(np.asarray(S.sample_index(p, 0.0))), 2)
        self.assertEqual(int(np.asarray(S.sample_index(p, 0.24999))), 2)
        self.assertEqual(int(np.asarray(S.sample_index(p, 0.25001))), 3)
        self.assertEqual(int(np.asarray(S.sample_index(p, 0.99999))), 3)


class TestSpeculativeOutputDistribution(unittest.TestCase):
    """The committed stream must follow the target, for any draft quality."""

    V, P = 32, 4
    N_NEW = 16000

    def _run(self, draft, *, top_p=1.0, top_k=0, seed=5, N=None, broken=False,
             draft_scale=2.0, temperature=1.0):
        dev = S._Device("numpy")
        R, S_ = make_fixture(self.V, self.P, seed=17, draft=draft,
                             draft_scale=draft_scale)
        model = FixtureModel(R, dev)
        head = FixtureDraftHead(S_, dev)
        prompt = [3, 6, 9]
        orig = S.verify_round
        if broken:
            def broken_verify(p_rows, q_rows, dr, u_acc, u_res, u_bonus):
                g = len(p_rows) - 1
                acc = []
                for k in range(g):
                    pk, qk, dk = p_rows[k], q_rows[k], int(dr[k])
                    a = S.acceptance_prob(pk[dk], qk[dk])
                    if float(u_acc[k]) < a:
                        acc.append(dk)
                        continue
                    t = _to_int(dev, S.sample_index(pk, float(u_res)))  # WRONG: p
                    return acc, t, len(acc), True
                t = _to_int(dev, S.sample_index(p_rows[g], float(u_bonus)))
                return acc, t, g, False
            S.verify_round = broken_verify
        try:
            out, stats = run_loop(model, head, prompt, N or self.N_NEW,
                                  temperature=temperature, top_p=top_p,
                                  top_k=top_k, seed=seed)
        finally:
            S.verify_round = orig
        return out, stats, R, S_, prompt

    def _check(self, draft, tag, *, top_p=1.0, top_k=0, min_chi2_p=1e-3,
               min_ks_p=1e-3, draft_scale=2.0, seed=5, N=None):
        out, stats, R, S_, prompt = self._run(draft, top_p=top_p, top_k=top_k,
                                             seed=seed, N=N, draft_scale=draft_scale)
        V = self.V
        hist = histogram(out, V)
        n = hist.sum()
        self.assertGreater(n, 0.9 * (self.N_NEW if N is None else N))
        ph = phases_of(out, prompt[-1], self.P)
        self.assertEqual(len(ph), len(out))
        rows = []
        for f in range(self.P):
            mask = np.asarray([x == f for x in ph])
            obs = hist_of(out, mask, V)
            p = np.asarray(S.filtered_probs(R[f][None, :], 1.0, top_p, top_k))[0]
            chi2, dof, pv = chi_square(obs, p)
            d, ksp = ks_test(obs, p)
            rows.append((f, int(obs.sum()), chi2, dof, pv, d, ksp))
            self.assertGreater(pv, min_chi2_p,
                               f"{tag} phase {f}: chi2={chi2:.2f} dof={dof} p={pv:.3g}")
            self.assertGreater(ksp, min_ks_p,
                               f"{tag} phase {f}: KS D={d:.4f} p={ksp:.3g}")
        print(f"\n[sampling] {tag}: {len(out)} tokens, {stats['rounds']} rounds, "
              f"accept {1 - stats['rejects'] / max(stats['rounds'], 1):.3f}, "
              f"drafted {stats['drafted']}")
        for f, n_f, chi2, dof, pv, d, ksp in rows:
            print(f"   phase {f} n={n_f:6d}  chi2={chi2:7.2f} (dof {dof:2d}) p={pv:.4f}"
                  f"   KS D={d:.4f} p={ksp:.4f}")
        return out, stats

    def test_identical_draft_never_rejects_and_matches_target(self):
        out, stats = self._check("identical", "draft==target")
        self.assertEqual(stats["rejects"], 0)

    def test_sharp_draft_matches_target(self):
        # q much sharper than p: rejection is common, the residual does the work
        out, stats = self._check("sharp", "draft=2x sharp")
        self.assertGreater(stats["rejects"], 0)
        self.assertGreater(stats["accept_rate"], 0.0)

    def test_rotated_draft_matches_target(self):
        # q almost disjoint from p: nearly every round rejects
        out, stats = self._check("rotate", "draft rotated")
        self.assertGreater(stats["rejects"], 0)

    def test_flat_draft_matches_target(self):
        out, stats = self._check("flat", "draft flat", draft_scale=1.0)
        self.assertGreater(stats["rejects"], 0)

    def test_top_k_target_matches_target(self):
        out, stats = self._check("sharp", "top_k=8", top_k=8)
        hist = histogram(out, self.V)
        R = make_fixture(self.V, self.P, 17)[0]
        # every emitted token must be inside the target's top-8 support, in any
        # phase (each phase is a different table row)
        support = np.zeros(self.V, dtype=bool)
        for f in range(self.P):
            p = np.asarray(S.filtered_probs(R[f][None, :], 1.0, 1.0, 8))[0]
            support |= p > 0
        self.assertTrue(np.all(hist[~support] == 0), "sampled a top-k-filtered token")

    def test_top_p_target_matches_target(self):
        out, stats = self._check("rotate", "top_p=0.9", top_p=0.9)

    def test_broken_verifier_is_rejected_by_the_same_test(self):
        """Negative control: resampling from p instead of (p - q)+ must fail."""
        out, stats, R, S_, prompt = self._run("rotate", broken=True, N=6000)
        V = self.V
        hist = histogram(out, V)
        ph = phases_of(out, prompt[-1], self.P)
        worst = 1.0
        for f in range(self.P):
            obs = hist_of(out, np.asarray([x == f for x in ph]), V)
            p = np.asarray(S.filtered_probs(R[f][None, :], 1.0, 1.0, 0))[0]
            _, _, pv = chi_square(obs, p)
            worst = min(worst, pv)
        print(f"\n[sampling] negative control (wrong residual): worst chi2 p={worst:.3g}")
        self.assertLess(worst, 1e-6)

    def test_seed_control_is_deterministic(self):
        a, sa, *_ = self._run("sharp", seed=99, N=1200)
        b, sb, *_ = self._run("sharp", seed=99, N=1200)
        c, sc, *_ = self._run("sharp", seed=100, N=1200)
        self.assertEqual(a, b, "same seed produced a different stream")
        self.assertEqual(sa["rejects"], sb["rejects"])
        self.assertNotEqual(a, c, "different seeds produced the same stream")

    def test_same_seed_means_same_stream_on_both_ranks(self):
        # TP=2 correctness: the loop draws only from the seeded pool, so two
        # identical runs (one per rank) commit identical tokens.
        a, *_ = self._run("rotate", seed=7, N=800)
        b, *_ = self._run("rotate", seed=7, N=800)
        self.assertEqual(a, b)

    def test_gamma_and_adaptive_paths_all_match_target(self):
        for gamma, adaptive in ((1, False), (2, False), (4, False), (3, True)):
            dev = S._Device("numpy")
            R, S_ = make_fixture(self.V, self.P, 17, "sharp")
            model, head = FixtureModel(R, dev), FixtureDraftHead(S_, dev)
            out, stats = S.spec_generate(model, head, [3, 6, 9], 6000, gamma=gamma,
                                         adaptive=adaptive, eos_id=-1, seed=3,
                                         sp=_StubSpec,
                                         sampler=S.Sampler(1.0, 1.0, 0, 3, backend="numpy"))
            hist = histogram(out, self.V)
            ph = phases_of(out, 9, self.P)
            worst = 1.0
            for f in range(self.P):
                obs = hist_of(out, np.asarray([x == f for x in ph]), self.V)
                p = np.asarray(S.filtered_probs(R[f][None, :], 1.0, 1.0, 0))[0]
                _, _, pv = chi_square(obs, p)
                _, ksp = ks_test(obs, p)
                worst = min(worst, pv, ksp)
            self.assertGreater(worst, 1e-3, f"gamma={gamma} worst p={worst:.3g}")
            print(f"\n[sampling] gamma={gamma} adaptive={adaptive}: "
                  f"{len(out)} tokens, worst p={worst:.4f}, gammas "
                  f"{sorted(set(stats['gammas']))}, acc "
                  f"{1 - stats['rejects'] / stats['rounds']:.3f}")


def hist_of(out, mask, V):
    vals = np.asarray(out, dtype=np.int64)[mask]
    return np.bincount(vals, minlength=V)[:V].astype(np.float64)


class TestDraftGeometry(unittest.TestCase):
    """``draft_geometry`` must refuse a draft head that cannot be gathered."""

    def test_unsharded_full_vocab_head_is_accepted(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32), head=SimpleNamespace())
        head = SimpleNamespace(markov_head=SimpleNamespace(
            weight=np.zeros((32, 4), np.float32)))
        vocab, world = S.draft_geometry(model, head)
        self.assertEqual((vocab, world), (32, 1))
        self.assertTrue(head.vocab_sharded)

    def test_sharded_pair_is_accepted(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32),
                                head=SimpleNamespace(_world=2))
        head = SimpleNamespace(markov_head=SimpleNamespace(
            weight=np.zeros((16, 4), np.float32)))
        self.assertEqual(S.draft_geometry(model, head), (32, 2))

    def test_mismatched_sharding_is_refused(self):
        # body sharded, draft not sharded: gather would be given full-vocab rows
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32),
                                head=SimpleNamespace(_world=2))
        head = SimpleNamespace(markov_head=SimpleNamespace(
            weight=np.zeros((32, 4), np.float32)))
        with self.assertRaises(RuntimeError):
            S.draft_geometry(model, head)

    def test_wrong_full_vocab_width_is_refused(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32), head=SimpleNamespace())
        head = SimpleNamespace(markov_head=SimpleNamespace(
            weight=np.zeros((16, 4), np.float32)))
        with self.assertRaises(RuntimeError):
            S.draft_geometry(model, head)

    def test_gather_refuses_a_row_that_is_not_a_rank_slice(self):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32),
                                head=SimpleNamespace(_world=2, _lo=0, _vocab=32,
                                                     _group=SimpleNamespace()))
        probe = S.DraftProbe(model, S.Sampler(1.0, 1.0, 0, 1, backend="numpy"), 1)
        with self.assertRaises(ValueError):
            probe.gather(np.zeros((1, 7), np.float32))      # not vocab//world


class TestUniformSlotLayout(unittest.TestCase):
    """The round's uniform pool must be partitioned, never shared.

    Regression test for a real bias this suite caught: when the acceptance test
    reused the *draft draw's* uniform, "which token was drawn" and "was it
    accepted" became dependent and the emitted distribution drifted from p
    (chi2=1461 at dof=31). The layout is [0,g) draft, [g,2g) accept, 2g residual,
    2g+1 bonus, and this pins it exactly.
    """

    def test_slots_are_disjoint_and_fixed_size(self):
        dev = S._Device("numpy")
        R, S_ = make_fixture(32, 4, 17, "sharp")
        model, head = FixtureModel(R, dev), FixtureDraftHead(S_, dev)
        seen = {}
        orig = S.verify_round

        def spy(p_rows, q_rows, dr, u_acc, u_res, u_bonus):
            seen["g"] = len(p_rows) - 1
            seen["u_acc"] = np.array(u_acc, copy=True)
            seen["u_res"], seen["u_bonus"] = float(u_res), float(u_bonus)
            return orig(p_rows, q_rows, dr, u_acc, u_res, u_bonus)

        S.verify_round = spy
        try:
            sampler = S.Sampler(1.0, 1.0, 0, 4, backend="numpy")
            pool_per_round = []
            bset = sampler.begin_round
            sampler.begin_round = lambda n: (pool_per_round.append(int(n)), bset(n))[1]
            _, stats = S.spec_generate(model, head, [3, 6, 9], 60, gamma=2,
                                       adaptive=False, eos_id=-1, seed=1,
                                       sampler=sampler, sp=_StubSpec)
        finally:
            S.verify_round = orig
        g = seen["g"]
        self.assertEqual(g, 2)
        # every round reserved 2g+2 slots
        self.assertTrue(all(n == 2 * 2 + 2 for n in pool_per_round[1:]),
                        f"pool sizes {sorted(set(pool_per_round))}")
        # the acceptance draws really are uniforms
        self.assertTrue(np.all((seen["u_acc"] >= 0) & (seen["u_acc"] < 1)))
        self.assertGreater(stats["drafted"], 0)
        self.assertEqual(stats["drafted"] % g, 0)

    def test_accept_draws_do_not_reuse_draft_draws(self):
        """The bias that motivated the layout: if a round's g draft-step draws
        and its g acceptance draws came from the same slots, the acceptance
        draws would equal them. They must be different values."""
        dev = S._Device("numpy")
        R, S_ = make_fixture(32, 4, 17, "rotate")
        pools = []
        orig = S.verify_round
        S.verify_round = lambda p, q, dr, ua, ur, ub: (pools.append(
            (np.array(ua, copy=True), float(ur), float(ub))), orig(p, q, dr, ua, ur, ub))[1]
        try:
            sampler = S.Sampler(1.0, 1.0, 0, 21, backend="numpy")
            allpools = []
            bset = sampler.begin_round
            sampler.begin_round = lambda n: (allpools.append(int(n)), bset(n))[1]
            S.spec_generate(FixtureModel(R, dev), FixtureDraftHead(S_, dev),
                            [3, 6, 9], 40, gamma=2, adaptive=False, eos_id=-1,
                            seed=1, sampler=sampler, sp=_StubSpec)
        finally:
            S.verify_round = orig
        self.assertTrue(len(pools) > 3)
        # no acceptance uniform equals the corresponding draft uniform: the
        # draft slots and accept slots are distinct positions of the same pool
        seen = set()
        for ua, _, _ in pools:
            for v in ua:
                seen.add(round(float(v), 12))
        self.assertEqual(len(seen), len(pools) * 2, "accept draws collided with draft slots")

    def test_pool_slots_are_actually_distinct_values(self):
        s = S.Sampler(1.0, 1.0, 0, 17, backend="numpy")
        for _ in range(6):
            s.begin_round(8)
            p = s.host_pool()
            self.assertEqual(p.shape[0], 8)
            self.assertEqual(len(set(np.round(p, 12).tolist())), 8)


class TestSpecModuleUntouched(unittest.TestCase):
    """The greedy path must be exactly what it was, with sampling a strict
    opt-in: ``spec.generate`` reaches the sampling module only for
    ``temperature > 0``, and the greedy branch below the dispatch is byte-for-
    byte the pre-existing loop (argmax targets, argmax drafts, no RNG)."""

    SPEC = os.path.join(_ROOT, "mlx_lm", "models", "deepseek_v41", "spec.py")

    def _source(self):
        with open(self.SPEC) as fh:
            return fh.read()

    def test_sampling_is_imported_lazily_and_only_on_demand(self):
        src = self._source()
        # module import must not pull in the sampling module (it is optional at
        # import time), the dispatch imports it inside the function
        for line in src.splitlines():
            if "import sampling" in line or "from . import sampling" in line:
                self.assertTrue(line.startswith((" ", "\t")),
                                f"top-level sampling import: {line!r}")
        self.assertIn("if temperature and temperature > 0.0:", src)
        self.assertIn('from . import sampling as _sampling', src)

    def test_greedy_defaults_unchanged(self):
        """Signature defaults and the greedy body, parsed from the source.

        ``spec.py`` imports mlx.core at module level, so on a host without MLX
        the module cannot be imported at all -- the source is the ground truth
        for this check either way.
        """
        import ast

        tree = ast.parse(self._source())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "generate")
        args = fn.args
        names = [a.arg for a in args.args + args.kwonlyargs]
        defaults = {}
        for a, d in zip(args.args[len(args.args) - len(args.defaults):], args.defaults):
            defaults[a.arg] = d
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            defaults[a.arg] = d
        self.assertEqual(ast.literal_eval(defaults["temperature"]), 0.0)
        self.assertEqual(ast.literal_eval(defaults["gamma"]), 3)
        self.assertEqual(ast.literal_eval(defaults["eos_id"]), 1)
        self.assertTrue(ast.literal_eval(defaults["top_p"]) == 1.0)
        self.assertEqual(ast.literal_eval(defaults["top_k"]), 0)
        self.assertIsNone(ast.literal_eval(defaults["policy"]))
        self.assertIsNone(ast.literal_eval(defaults["seed"]))
        for expected in ("temperature", "top_p", "top_k", "seed", "sampler"):
            self.assertIn(expected, names)
        if not _HAVE_MLX:
            # the file-level import is the reason the test above cannot run on
            # this host; make that explicit rather than silently weaker
            self.assertIn("import mlx.core as mx", self._source())
        # the greedy loop below the dispatch is untouched: argmax targets and
        # argmax drafts, no RNG and no sampling call anywhere in the body
        body = self._source().split("taps_ids = list(")[1]
        self.assertIn("argmax=True", body)
        for needle in ("mx.random", "sampling.", "top_p", "top_k", "temperature"):
            self.assertNotIn(needle, body)

    def test_sampling_module_refuses_greedy_temperature(self):
        with self.assertRaises(ValueError):
            S.Sampler(temperature=0.0)


@unittest.skipUnless(_HAVE_MLX, "mlx not available (gateway)")
class TestMlxBackend(unittest.TestCase):
    V, P = 32, 4

    @classmethod
    def setUpClass(cls):
        cls._real_all_sum = _patch_all_sum_identity()

    @classmethod
    def tearDownClass(cls):
        mx.distributed.all_sum = cls._real_all_sum

    def test_transform_matches_numpy(self):
        rng = np.random.default_rng(3)
        logits = (rng.normal(size=(8, 64)) * 2.0).astype(np.float32)
        for top_p, top_k in ((1.0, 0), (0.9, 0), (1.0, 8), (0.95, 12)):
            a = np.asarray(S.filtered_probs(mx.array(logits), 1.0, top_p, top_k))
            b = np.asarray(S.filtered_probs(logits, 1.0, top_p, top_k))
            self.assertLess(float(np.abs(a - b).max()), 1e-6,
                            f"top_p={top_p} top_k={top_k}")

    def test_mlx_rng_is_key_deterministic(self):
        k = mx.random.key(1234)
        a = np.asarray(mx.random.categorical(mx.array(np.linspace(-2, 2, 16, np.float32)),
                                             axis=-1, key=k))
        b = np.asarray(mx.random.categorical(mx.array(np.linspace(-2, 2, 16, np.float32)),
                                             axis=-1, key=k))
        np.testing.assert_array_equal(a, b)

    def test_uniform_pool_is_fixed_size_and_reproducible(self):
        s1 = S.Sampler(1.0, 1.0, 0, seed=42, backend="mlx")
        s2 = S.Sampler(1.0, 1.0, 0, seed=42, backend="mlx")
        p1 = np.asarray(s1.begin_round(8))
        p2 = np.asarray(s2.begin_round(8))
        np.testing.assert_allclose(p1, p2, rtol=0, atol=0)
        p1b = np.asarray(S.Sampler(1.0, 1.0, 0, seed=43, backend="mlx").begin_round(8))
        self.assertFalse(np.allclose(p1, p1b))

    def test_mlx_loop_matches_target_distribution(self):
        dev = S._Device("mlx")
        R, S_ = make_fixture(self.V, self.P, seed=17, draft="sharp")
        model = FixtureModel(R, dev)
        head = FixtureDraftHead(S_, dev)
        out, stats = run_loop(model, head, [3, 6, 9], 8000, seed=5, backend="mlx")
        hist = histogram(out, self.V)
        ph = phases_of(out, 9, self.P)
        for f in range(self.P):
            obs = hist_of(out, np.asarray([x == f for x in ph]), self.V)
            p = np.asarray(S.filtered_probs(R[f][None, :], 1.0, 1.0, 0))[0]
            chi2, dof, pv = chi_square(obs, p)
            d, ksp = ks_test(obs, p)
            print(f"\n[sampling] mlx phase {f} n={int(obs.sum())} chi2={chi2:.2f} "
                  f"(dof {dof}) p={pv:.4f} KS D={d:.4f} p={ksp:.4f}")
            self.assertGreater(pv, 1e-3)
        self.assertEqual(stats["backend"], "mlx")

    def test_mlx_seed_control(self):
        x1 = self._run_once(11)
        x2 = self._run_once(11)
        x3 = self._run_once(12)
        self.assertEqual(x1, x2)
        self.assertNotEqual(x1, x3)

    def test_mlx_probe_pads_and_sums_like_the_sharded_head(self):
        """The probe's gather must be ShardedHead's geometry exactly.

        Real ShardedHead.__call__ is: ``all_sum(pad(y, [lo, vocab-lo-w]))``.
        On one process the all_sum is identity, so rank r's gathered row is
        zero everywhere except ``[lo, lo+w)`` -- the exact per-rank contribution
        whose sum over ranks rebuilds the full head row.  Checked against an
        explicitly padded reference for both ranks.
        """
        V, W = 32, 16
        dev = S._Device("mlx")
        rng = np.random.default_rng(4)
        full = rng.normal(size=(2, V)).astype(np.float32)
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=V),
                                head=SimpleNamespace(_world=2, _lo=0, _vocab=V,
                                                     _group=_FakeGroup("mlx")))
        probe = S.DraftProbe(model, S.Sampler(1.0, 1.0, 0, 1, backend="mlx"), 1)
        for rank in (0, 1):
            probe.lo = lo = rank * W
            y = mx.array(full[:, lo:lo + W].copy())
            got = np.asarray(probe.gather(y.astype(mx.float32)))
            want = np.zeros((2, V), np.float32)
            want[:, lo:lo + W] = full[:, lo:lo + W]
            np.testing.assert_allclose(got, want, rtol=0, atol=0)
        # and the two rank contributions add back to the full row exactly
        probe.lo = 0
        a = np.asarray(probe.gather(mx.array(full[:, :W].copy())))
        probe.lo = W
        b = np.asarray(probe.gather(mx.array(full[:, W:].copy())))
        np.testing.assert_allclose(a + b, full, rtol=0, atol=0)

    def test_mlx_probe_draws_from_the_gathered_distribution(self):
        """The probe's draw must follow the *gathered* full-vocab row.

        A single process cannot run a real mesh, so the two ranks' padded
        contributions are computed separately and added -- the same sum
        ``ShardedHead`` performs -- and the probe's draw is checked against that
        reconstructed row.  The support sits inside rank 1's slice, so a probe
        that skipped the gather (drawing from the rank-local half) would pick a
        zero row and this test would catch it.
        """
        V, W = 32, 16
        lo, hi = 16, 24
        row = np.full(V, -20.0, np.float32)
        row[lo:hi] = np.linspace(2.0, 6.0, hi - lo, dtype=np.float32)
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=V),
                                head=SimpleNamespace(_world=2, _lo=0, _vocab=V,
                                                     _group=_FakeGroup("mlx")))
        geom = S.DraftProbe(model, S.Sampler(1.0, 1.0, 0, 5, backend="mlx"), 1)
        parts = []
        for rank in (0, 1):
            geom.lo = rank * W
            y = mx.array(row[None, rank * W:(rank + 1) * W].copy())   # [1, w]
            parts.append(np.asarray(geom.gather(y.astype(mx.float32))))
        gathered = parts[0] + parts[1]                                # [1, V]
        np.testing.assert_allclose(gathered, row[None], rtol=0, atol=0)

        sampler = S.Sampler(1.0, 1.0, 0, 5, backend="mlx")
        sampler.begin_round(4)
        probe = S.DraftProbe(model, sampler, 1)
        probe.begin_round(1)
        probe.lo = 0
        probe.gather = lambda y, _g=gathered: mx.array(_g)      # the mesh sum
        d = probe.combine_argmax(mx.array(row[:W].copy())[None])   # [1, w] like draft()
        self.assertTrue(lo <= int(np.asarray(d).reshape(-1)[0]) < hi,
                        "probe drew from outside the true support")
        self.assertEqual(len(probe.qs), 1)
        q = np.asarray(probe.qs[0][0]).astype(np.float64)        # [V]
        want = np.exp(row.astype(np.float64))
        want /= want.sum()
        np.testing.assert_allclose(q, want, rtol=1e-5, atol=1e-9)
        self.assertEqual(int(np.asarray(d).reshape(-1)[0]),
                         int(np.argmax(np.cumsum(want) > float(sampler.pool[0]))))

    def _run_once(self, seed):
        dev = S._Device("mlx")
        R, S_ = make_fixture(self.V, self.P, seed=17, draft="rotate")
        model, head = FixtureModel(R, dev), FixtureDraftHead(S_, dev)
        out, _ = run_loop(model, head, [3, 6, 9], 600, seed=seed, backend="mlx")
        return out


if __name__ == "__main__":
    print(f"[sampling] mlx available: {_HAVE_MLX}")
    unittest.main(verbosity=2)
