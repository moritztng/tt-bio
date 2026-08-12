#!/bin/bash
# RUN E: the byte-identical-only landing. Run A measured `g12` and `e6` separately and `all`
# together, but never the two byte-identical levers as one arm, so section 11.9 could not put a
# number on the part of this work that ships with NO accuracy decision. With the run-D structural
# RMSD at 1.25 A CA, that number is the one that matters most. Arms on,ge,on: the trailing `on`
# gives the A/A floor in the same process, and `ge` MUST return CIF sha256 50aa1e46583bd5a8.
WT=/home/ttuser/.coworker/wt/opendde-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-512aa-deep-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=0.6 BENCHLOCK_LOAD_WAIT_S=600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-512aa-deep-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms on,ge,on --keep-cif perf/odde512/cif_e \
      --out perf/odde512/ab_opendde_ge.json
echo "RC_E=$?"
echo "=== RUN E DONE ==="
