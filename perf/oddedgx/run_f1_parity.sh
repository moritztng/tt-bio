#!/usr/bin/env bash
# Run 2, lever A gate 1: is F1 at kt=12 bit-exact against the ops it replaces?
# The reference is _trimul_out_proj, production's own tail, not a config-matched minimal_matmul.
# c_z=256 first: that arm is the REGRESSION test for the per-call block lookup -- esmfold2 and
# protenix-v2 fire F1 today and must stay torch.equal. Then c_z=384, the new one.
# Acceptance: torch.equal, max_abs_diff exactly 0.0, at every shape, in BOTH arms.
set -u
source /home/ttuser/.coworker/wt/opendde-beat-dgx-h200/perf/oddedgx/env.sh
cd $WT || exit 1
for CZ in 384 256; do
  echo "=== f1_parity c_z=$CZ $(date -Is) ==="
  $BL opendde-beat-dgx-h200 -- $PY -u perf/trimul_f1/f1_parity.py --cz $CZ \
      32 64 128 256 512
  echo "c_z=$CZ RC=$?"
done
echo "done $(date -Is)"
