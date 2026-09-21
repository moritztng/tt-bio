#!/usr/bin/env bash
# PROTOCOL SS6: fire the bond term on a target whose crop carries one, and measure its
# gradient contribution. CPU only, no card.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxheads
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs:$W"
export OMP_NUM_THREADS=${OMP:-4}
echo "=== bond coverage start $(date -u +%FT%TZ) ==="
nice -n 19 "$PY" perf/of3t_auxheads/bond_coverage.py \
    --data-dir /home/ttuser/of3t-data-xhost/datasets \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --stage finetune_1 --crop 384 --index 0 --dtype float32 --rank-template /home/ttuser/of3t_rebase/bundle_min_043/batch_step003.pt \
    --out perf/of3t_auxheads/bond_coverage_finetune1.json
echo "=== bond exit $? $(date -u +%FT%TZ) ==="
echo BOND_ALLDONE
