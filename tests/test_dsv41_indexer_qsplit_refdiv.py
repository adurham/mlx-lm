# Copyright © 2026 Adam Durham (hermes-gw)
"""QSPLIT-path reference-divergence PIN — the bounded regression for the qsplit surface.

Companion to ``bench/dsv41_indexer_qsplit_refdiv.py`` (the full matrix; exit-code
selection gate) and to the wave-1 audit's ``tests/test_dsv41_indexer_qsplit_band_matrix.py``
(which could not reach the strip/merge regime or the projection-kernel class — see its
own ``k=6`` geometry).

What this file pins, and why each is worth a test:

* the GATE (``query_band``) — bands exactly when ``n % world == 0`` AND the BAND's rows
  exceed ``_FENCE_MIN_ROWS`` (D2), and never at decode / verify shapes;
* the PREFILL-CHUNK MATRIX — banded reconstruction == replicated run, byte-exact, on the
  hierarchical and tiled routes at production geometry (``index_topk=512``);
* the D2 HAZARD, measured — an explicitly injected below-fence band does move selections
  on the tiled fallback, which is the evidence that the D2 fence is load-bearing;
* the PROJECTION-KERNEL CLASS (``de6caa2``) — a row-count-dependent projection must not
  separate the arms, while a faithful model of the pre-fix expression must diverge;
* the D3 READ-OUT (``6a5c78f``) — every projection is CALLED with ``n_full`` rows;
* the STRIP-WIDTH ASYMMETRY — the band-derived exact-pass strip differs from the
  chunk-derived one at every shape in the census, its merge LENGTH is therefore
  different, and the shipped path was nevertheless not separated by any tie structure
  the harness can build (including a 900-way exact tie straddling k);
* NON-VACUITY — a margin-sized perturbation flips a selection, and the tie lattice puts
  exact ties on rank k, so the comparisons above can see a difference.

Run (mlx-lm worktree root — NOT from the vendored ``~/repos/exo/mlx-lm``, whose parent
``tests/`` package wins the import race)::

    cd <worktree> && PYTHONPATH=$PWD <exo-venv>/bin/python -m pytest \
        tests/test_dsv41_indexer_qsplit_refdiv.py -q
"""

from __future__ import annotations

import unittest

import numpy as np

from mlx_lm.models.deepseek_v41 import indexer as IX

from bench.dsv41_indexer_qsplit_refdiv import (
    FENCE,
    Group,
    INDEX_TOPK,
    RESULTS,
    RowKernelLinear,
    argpartition_tie_probe,
    cell,
    inputs_for,
    nb_for,
    restore_knobs,
    save_knobs,
    section_a,
    section_f,
    section_g,
    section_h,
    section_i,
)


class _Fixture(unittest.TestCase):
    """Save/restore every import-read module knob (the sections mutate them)."""

    def setUp(self) -> None:
        self._saved = save_knobs()

    def tearDown(self) -> None:
        restore_knobs(self._saved)


class GateAndGeometryTest(_Fixture):
    def test_gate_engages_exactly_at_the_d2_fence(self) -> None:
        out = section_a()
        self.assertEqual(out["gate_mismatch"], 0)
        self.assertGreater(out["banded_shapes"], 0)
        self.assertEqual(out["first_banded_n"], {2: 34, 4: 68})
        self.assertEqual(out["odd_n_banded"], {2: [34], 4: []})

    def test_dtype_and_estrip_asymmetry_census(self) -> None:
        out = section_a()
        self.assertEqual(out["asym_dtype_mismatch"], 0,
                         "D2: the band resolved a different row dtype than the chunk")
        self.assertEqual(out["asym_estrip_mismatch"], out["asym_cells"],
                         "the band-derived exact-pass strip differs at EVERY cell")
        self.assertGreater(out["asym_strip_count_mismatch"], 0,
                           "no cell had a different exact-pass strip COUNT")
        self.assertGreater(out["asym_pre_d3_route_mismatch"], 0,
                           "the pre-D3 route rule never differs -- the D3 test is vacuous")


class PrefillMatrixTest(_Fixture):
    """Non-first prefill chunks, band edges, phases, worlds, roles, key lattices."""

    def test_non_first_chunks_at_production_contexts(self) -> None:
        for i, (n, world, ctx) in enumerate(((34, 2, 57_344), (78, 2, 57_344),
                                            (512, 2, 57_344), (132, 4, 77_824),
                                            (130, 2, 131_072), (512, 2, 51_200))):
            row = cell(seed=90_000 + i, tag="t_matrix", role="source", route="hier",
                       world=world, n=n, nb=nb_for(ctx), mode="lattice", phase=0)
            self.assertGreater(row["start_pos"], 0, "not a non-first chunk")
            self.assertTrue(row["gated"], (n, world))
        # a chunk whose rows do NOT divide across the ranks stays replicated (D2)
        for n, world in ((78, 4), (130, 4), (129, 2)):
            row = cell(seed=90_500, tag="t_matrix", role="source", route="hier",
                       world=world, n=n, nb=nb_for(57_344), mode="lattice", phase=0)
            self.assertFalse(row["gated"], (n, world))

    def test_band_seam_phases(self) -> None:
        for phase in (0, 1, 3, 4):
            for world in (2, 4):
                cell(seed=91_000 + phase + world, tag="t_phase", role="source",
                     route="hier", world=world, n=130, nb=nb_for(57_344),
                     mode="lattice", phase=phase)

    def test_partial_last_band_and_unaligned_seam(self) -> None:
        # 130/world=2 -> 65-row bands (a 1-column tail tile at a 64-wide tile); 100/world=4
        # -> 25-row bands; nb=520 is a 8-column last tile at width 512.
        for n, world, nb in ((130, 2, 520), (132, 4, 519), (100, 4, 528)):
            row = cell(seed=92_000 + n, tag="t_partial_band", role="source",
                       route="tiled", world=world, n=n, nb=nb, mode="lattice", phase=0)
            self.assertTrue(row["padded_zero_outside"])
            self.assertEqual(row["max_contributors"], 1)

    def test_consumer_role_consumes_the_banded_candidates(self) -> None:
        for world, n, ctx in ((2, 78, 57_344), (4, 132, 77_824)):
            row = cell(seed=93_000 + world, tag="t_consumer", role="consumer",
                       route="hier", world=world, n=n, nb=nb_for(ctx), mode="ties",
                       phase=0)
            self.assertTrue(row["cons_band_slice_equal"])

    def test_batch_two_is_banded_per_batch(self) -> None:
        row = cell(seed=94_001, tag="t_batch", role="source", route="hier", world=2,
                   n=78, nb=nb_for(57_344), b=2, mode="lattice", phase=0)
        self.assertEqual(row["rows_compared"], 2 * 78)


class DecodeVerifyGateTest(_Fixture):
    """Decode / verify shapes at real contexts: the gate must refuse, no collective."""

    def test_small_n_never_bands_at_any_context(self) -> None:
        for n in (1, 2, 3, 4, 5, 8, 16, 17, 32, 33):
            for world in (2, 4):
                self.assertIsNone(IX.query_band(n, Group(0, world)), (n, world))
        for ctx in (32_768, 57_344, 77_824, 262_144):
            for route in ("untiled", "tiled"):
                row = cell(seed=95_000 + ctx, tag="t_decode", role="source", route=route,
                           world=2, n=4, nb=nb_for(ctx), mode="lattice", phase=0)
                self.assertFalse(row["gated"])
                self.assertIsNone(row["band"])

    def test_d2_hazard_is_measurable_on_the_tiled_fallback(self) -> None:
        """An injected below-fence band on a chunk ABOVE the fence moves selections.

        This is the measurement that makes the D2 fence load-bearing rather than
        decorative: the band's own regime differs from the chunk's, and the difference is
        selection-visible on the tiled fallback route (and, at these shapes, invisible on
        the hierarchical route because the hierarchy prunes nothing at nb=2048).
        """
        seen = {}
        for i, n in enumerate((20, 26)):
            for route in ("tiled", "hier"):
                row = cell(seed=96_000 + i, tag="t_d2", role="source", route=route,
                           world=2, n=n, nb=nb_for(131_072), mode="ties", phase=0,
                           explicit_band=True, check=False, expect_d2_hazard=True)
                seen[(route, n)] = row["d2_hazard_rows"]
        self.assertGreater(seen[("tiled", 20)], 0,
                           "the D2 hazard is not reachable -- the fence may be decoration")
        self.assertGreater(seen[("tiled", 26)], 0)


class TopkEdgeTest(_Fixture):
    """index_topk=512 edges and the merged selection's duplicate/dedupe behaviour."""

    def test_nb_near_k_duplicates_and_merge(self) -> None:
        for nb in (512, 513, 520, 1024):
            for mode in ("lattice", "ties"):
                row = cell(seed=97_000 + nb, tag="t_topk", role="source", route="tiled",
                           world=2, n=1024, nb=nb, mode=mode, phase=0)
                self.assertEqual(row["dup_real_indices"], 0)
                self.assertEqual(row["k"], min(INDEX_TOPK, nb))

    def test_merge_impl_matches_the_buffer_impl(self) -> None:
        for impl in ("buffer", "merge"):
            for mode in ("lattice", "ties"):
                cell(seed=98_000 + len(mode), tag="t_merge", role="source", route="tiled",
                     world=2, n=1024, nb=2048, mode=mode, tiled_impl=impl, phase=0)


class SmallNRowPrecisionTest(_Fixture):
    """bf16 small-n rows: ``L2_FULL=0`` / ``SMALLN_ROW_BF16=1`` (fp32 by default)."""

    def test_bf16_small_n_row_matches_the_default_on_banded_shapes(self) -> None:
        for l2_full, sn_bf16 in ((True, True), (False, False), (False, True)):
            for route in ("hier", "tiled"):
                row = cell(seed=99_000 + route.__len__(), tag="t_bf16", role="source",
                           route=route, world=2, n=78, nb=nb_for(131_072), mode="ties",
                           phase=0, l2_full=l2_full, smalln_bf16=sn_bf16)
                self.assertEqual(row["dtype_band"], row["dtype_full"])


class ProjectionKernelClassTest(_Fixture):
    """The ORIGINAL live divergence class: a row-count-dependent query projection."""

    def test_row_count_dependent_projection_cannot_separate_the_arms(self) -> None:
        out = section_f()
        self.assertGreater(out["stub_row_dependence_differing"], 0,
                           "the row-count stub is not row-count dependent")
        for entry in out["immunity"]:
            self.assertTrue(entry["shipped_bytes_equal"],
                            f"the shipped banded path diverged at eps={entry['eps']}")
            self.assertTrue(entry["pre_fix_plain_reference_equal"],
                            "the pre-fix model diverged from an invariant projection")
        self.assertTrue(any(e["pre_fix_differing_rows"] > 0 for e in out["immunity"]),
                        "the pre-fix model never diverged -- vacuous probe")
        self.assertIsNotNone(out["pre_fix_threshold_eps"])

    def test_projection_is_called_with_the_full_chunk(self) -> None:
        out = section_h()
        self.assertTrue(out["all_calls_full"])
        for key, seen in out["observed"].items():
            self.assertEqual(seen["wq_rows"], [seen["n_full"]], key)
            self.assertEqual(seen["weights_proj_rows"], [seen["n_full"]], key)


class StripWidthAsymmetryTest(_Fixture):
    """The one shape-derived quantity the D2/D3 fixes do NOT align: the exact strip."""

    def test_strip_width_differs_but_moves_no_selection(self) -> None:
        out = section_g()
        self.assertTrue(all(a["differs"] for a in out["asymmetry"]),
                        "an asymmetry cell did not differ -- the census is stale")
        self.assertFalse(out["selection_sensitive"],
                         "the band-derived exact strip DID move a selection")
        for entry in out["direct_estrip"]:
            self.assertTrue(entry["bytes_equal"])
        for entry in out["band_vs_full"]:
            self.assertGreater(entry["kth_tied_rows"], 0,
                               "the tie lattice produced no boundary tie -- vacuous")
        for entry in out["wide_tie"]:
            self.assertEqual(entry["differing_rows"], 0)
            self.assertTrue(entry["band_slice_equal"])

    def test_the_tie_break_primitive_is_length_sensitive(self) -> None:
        """Documents WHY the strip asymmetry is a hazard at all rather than a non-issue.

        The shipped primitives pick a DIFFERENT subset of an exactly-tied top group
        depending on the merge layout (here 0.77 of k overlap), so the mechanism is real
        in the primitive even though the full hier path was not separated by it.
        """
        tie = argpartition_tie_probe()
        self.assertGreater(tie["tied_columns"], INDEX_TOPK)
        self.assertFalse(tie["one_pass_equals_two_round"])
        self.assertLess(tie["overlap_one_vs_two"], 1.0)


class NonVacuityTest(_Fixture):
    def test_margin_and_tie_controls(self) -> None:
        out = section_i()
        self.assertGreater(out["margin_flips"], 0)
        self.assertGreater(out["tie_rows_at_k"], 0)
        self.assertGreater(out["min_margin"], 0.0)


class RowKernelMatrixTest(_Fixture):
    """The matrix with a row-count-DEPENDENT projection installed (the live class).

    This is the arm wave-1's harness could not have: with a plain ``nn.Linear`` a
    reverted ``de6caa2`` is invisible, so the live divergence passed every predecessor
    suite.  Here the projection's OUTPUT depends on the row count it is called with, so
    the shipped call order is the only thing keeping the arms identical.
    """

    def test_matrix_is_immunity_under_a_row_count_dependent_kernel(self) -> None:
        for i, (eps, n, world, nb, route) in enumerate((
                (1e-4, 78, 2, 1024, "hier"), (1e-2, 132, 4, 3000, "hier"),
                (1e-4, 512, 2, 4096, "hier"), (1e-2, 130, 2, 3000, "hier"))):
            cell(seed=110_000 + i, tag="t_row_kernel", role="source", route=route,
                 world=world, n=n, nb=nb, mode="lattice", phase=0, row_kernel_eps=eps)

    def test_consumer_role_with_a_row_count_dependent_kernel(self) -> None:
        cell(seed=111_000, tag="t_row_kernel", role="consumer", route="hier", world=2,
             n=78, nb=1024, mode="ties", phase=0, row_kernel_eps=1e-3)

    def test_stub_really_is_row_count_dependent(self) -> None:
        """Non-vacuity of the stub itself: 39 rows vs 78 rows must give different q."""
        idx, _, _x, qr, _sp, _fv, _ik = inputs_for(112_000, "plain", 78, 1024)
        idx.wq_b = RowKernelLinear(idx.wq_b, eps=1e-4)
        band_q = np.array(idx.wq_b(qr[:, 0:39, :]))
        full_q = np.array(idx.wq_b(qr))[:, 0:39, :]
        self.assertGreater(int((band_q != full_q).sum()), 0)


def setUpModule() -> None:
    assert IX._FENCE_MIN_ROWS == FENCE, (
        f"this pin assumes the shipped fence {FENCE}, got {IX._FENCE_MIN_ROWS}")
    assert IX._INDEXER_COMPILE, "the compiled score body is the shipped default"
    RESULTS.clear()
