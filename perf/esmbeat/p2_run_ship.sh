#!/usr/bin/env bash
# Step 4: after the defaults are flipped, prove the flip landed. `ship` touches no gate, so it
# measures the module default and nothing else. This is the number the page cell publishes.
#
# Accept only if, in every fold of this run:
#   lm_handoff == [n, 0] with n > 0, dual_noc[0] > 0, cif 295867277b9c137f, plddt 0.9285
# and the median sits within noise of the BD arm measured in step 1. A `ship` that still reads
# ~31.95 s means the default did not land (GOALS.md: an OFF gate is not a win).
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=5400
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p2 -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esm3p4land/fold_ab.py \
  --model esmfold2 --size 512 --rounds 4 --arms ship \
  --out perf/esmbeat/p2_ship_512_c0_postmerge.json
echo "EXIT=$?"
