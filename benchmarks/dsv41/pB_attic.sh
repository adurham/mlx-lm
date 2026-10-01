#!/bin/sh
# pB_attic.sh -- compare ORIGINAL vs NEW indexer across shapes, one subprocess
# per (root, shape). Small tensors only (<= ~1.5 GB peak each), never queued.
set -e
HOME_DIR=$HOME
ORIG=${PB_ORIG_PKG:-$HOME_DIR/dsv41-ws/B_orig}
NEW=${PB_PKG:-$HOME_DIR/dsv41-ws/B}
PY=${PB_PY:-$HOME_DIR/repos/exo/.venv/bin/python}
T=$(mktemp -d)
fail=0
for spec in "2 32 512 0" "2 32 4096 0" "2 32 16384 0" "20 32 16384 0" \
            "20 32 8192 0" "20 48 16384 8192" "24 32 16384 0" "8 32 4112 8192"; do
  set -- $spec
  L=$1; N=$2; NB=$3; SP=$4
  for root in "$ORIG" "$NEW"; do
    tag=$(basename "$root")
    P48_PKG=$root PB_LAYER=$L PB_N=$N PB_NB=$NB PB_SP=$SP PB_OUT="$T/${tag}_${L}_${N}_${NB}_${SP}" \
      "$PY" "$NEW/pB_attic_one.py"
  done
  a="$T/$(basename $ORIG)_${L}_${N}_${NB}_${SP}"
  b="$T/$(basename $NEW)_${L}_${N}_${NB}_${SP}"
  if [ -f "$a.idx.npy" ] && [ -f "$b.idx.npy" ]; then
    same=$("$PY" - "$a.idx.npy" "$b.idx.npy" <<'EOF'
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
ar = [sorted(int(c) for c in r if c >= 0) for r in a[0]]
br = [sorted(int(c) for c in r if c >= 0) for r in b[0]]
print("EXACT" if a.tobytes() == b.tobytes() else
      ("SETEQ" if ar == br else "DIFF"))
EOF
)
    ca=""; cb=""
    [ -f "$a.cand.npy" ] && ca=$(md5 -q "$a.cand.npy" 2>/dev/null || md5sum "$a.cand.npy" | cut -d' ' -f1)
    [ -f "$b.cand.npy" ] && cb=$(md5 -q "$b.cand.npy" 2>/dev/null || md5sum "$b.cand.npy" | cut -d' ' -f1)
    cand="cand_none"
    if [ -n "$ca" ] || [ -n "$cb" ]; then
      [ "$ca" = "$cb" ] && cand="cand_EXACT" || cand="cand_DIFF"
    fi
    case "$same" in
      EXACT|SETEQ) verdict=PASS ;;
      *) verdict=FAIL; fail=$((fail+1)) ;;
    esac
    echo "[attic] $verdict L=$L n=$N nb=$NB sp=$SP orig_vs_new=$same $cand"
  else
    echo "[attic] FAIL L=$L n=$N nb=$NB sp=$SP (missing outputs)"; fail=$((fail+1))
  fi
done
rm -rf "$T"
echo "[attic] ATTIC_DONE failures=$fail"
exit $fail
