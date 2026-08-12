#!/usr/bin/env bash
# openfold3-to-4x §7 step 3: one benchlocked session, arms alternating on P on P.
#
#   on = pristine detached worktree at origin/main 8fc678b5 (git status --porcelain empty, checked)
#   P  = this worktree with §7 steps 1/2/5 -- pair bias pre-scaled, fused SDPA at HiFi4+fp32 dst
#
# Byte-identical harness both arms (perf/of3deep/decomp.py, cmp'd). Two runs per arm give an A/A
# floor per arm and bracket the session. Every CIF is retained -- the last attempt could not compute
# its RMSD because the CIFs were gone.
set -u
PY=/home/ttuser/tt-bio-dev/env/bin/python3
WT=/home/ttuser/.coworker/wt/openfold3-to-4x
BASE=/tmp/of3_base
OUT=$WT/perf/of3_4x/ab
mkdir -p "$OUT"
for i in 1 2; do
  for arm in on P; do
    [ "$arm" = on ] && root=$BASE || root=$WT
    echo "=== arm $arm run $i  ($root)  $(date -Is) ==="
    (cd "$root" && TT_VISIBLE_DEVICES=3 \
      TT_BIO_LEASE_HOLDER=worker:openfold3-to-4x \
      PYTHONPATH="$root" "$PY" perf/of3deep/decomp.py \
        --out "$OUT/${arm}_$i.json" --keep-cif "$OUT/cif_${arm}_$i") \
      2>&1 | grep -E "^  (cold|top:|diff:|dm:|trunk:|host:|prep:)|^=== run|Error|error|Traceback|assert"
    echo "--- exit ${PIPESTATUS[0]} arm $arm run $i $(date -Is)"
  done
done
echo "ALL ARMS DONE $(date -Is)"
