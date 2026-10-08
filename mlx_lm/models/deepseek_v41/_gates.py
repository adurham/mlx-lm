# Copyright © 2026 Adam Durham (hermes-gw)
"""Shared row-count gates for the DSv4.1 indexer / sparse-attention levers.

Two performance levers in this package keyed on the FORWARD's query-row count
``n`` (== ``m``; they are the same quantity within one forward):

* ``sparse_attention.py`` C1 column-partitioned gather -- the boundary
  derivation (``_column_boundary`` / ``_leading_window_columns``) issues two
  host round-trips and only pays off for large-m prefill. Gated ``m >
  _FENCE_MIN_ROWS`` since ``deploy/next17-levers`` (lever-1).
* ``indexer.py`` hierarchical / streamed exact pass (``DSV41_INDEXER_HIER``) --
  the coarse-pass + streamed-exact machinery is pure overhead at decode (n=1)
  and verify (n=4), where the fallback's score row is a single/small row.
  Gated ``n > _FENCE_MIN_ROWS`` since ``deploy/next18`` (lever-2).

Hoisting the threshold into ONE module retires the twin-gate foot-gun: before
this, ``sparse_attention.py`` and ``attention.py`` each read their own
``DSV41_SPARSE_COLSPLIT`` default (ON vs OFF) and ``_FENCE_MIN_ROWS`` lived only
in ``sparse_attention.py`` with a second, independent copy implied for
``indexer.py``. Both modules now import the SAME constant, so the two levers
can never disagree on the threshold.

Env contract (unchanged for A/B):

* ``DSV41_SPARSE_FENCE_MIN_ROWS`` (default ``16``) -- the threshold itself.
  ``=0`` disables the guard on BOTH levers (always run the fenced/C1 path and
  always run HIER), i.e. restores historical behaviour.
* ``DSV41_INDEXER_HIER=0`` -- forces the hierarchical indexer OFF
  unconditionally, independent of the row count (the lever-2 A/B kill switch).
* ``DSV41_SPARSE_COLSPLIT=0`` -- forces the C1 gather OFF (lever-1 A/B).

Read once at import, like the neighbouring ``DSV41_INDEXER_TILE*`` knobs.
"""

from __future__ import annotations

import os

#: Row-count threshold shared by both row-gated levers. A forward with
#: ``n <= _FENCE_MIN_ROWS`` (decode m=1, verify m=4) takes the cheap/fallback
#: path on both levers; a large-m prefill keeps the full fenced/HIER path.
#: ``0`` disables the guard (always the full path) -- the historical default.
_FENCE_MIN_ROWS = int(os.environ.get("DSV41_SPARSE_FENCE_MIN_ROWS", "16"))
