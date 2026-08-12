#!/bin/bash
# RUN D: the parity leg section 9.6 check 4 needs and run A could not give. Every arm in run A folded
# into the same struct_dir, so the mm12/all CIF was overwritten by the trailing `on` arm and only its
# sha256 survived. This keeps both arms' structures so the two can be compared by RMSD, not just
# declared different. Queues behind C and B on the same benchlock.
WT=/home/ttuser/.coworker/wt/opendde-512aa-deep-perf
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-512aa-deep-perf PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_MAXLOAD=0.6 BENCHLOCK_LOAD_WAIT_S=600
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-512aa-deep-perf -- \
  $PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 512 \
      --arms on,all --keep-cif perf/odde512/cif --out perf/odde512/ab_opendde_parity.json
echo "RC_D=$?"
echo "=== RUN D DONE ==="
