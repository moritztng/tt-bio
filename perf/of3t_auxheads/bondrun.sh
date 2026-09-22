#!/usr/bin/env bash
# PROTOCOL SS6: fire the bond term on a target whose crop carries one, and measure its
# gradient contribution. CPU only, no card.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=${OMP:-4}
echo "=== bond coverage start $(date -u +%FT%TZ) ==="
nice -n 19 "$PY" perf/of3t_auxheads/bond_coverage.py \
    --data-dir /home/ttuser/of3t-data-xhost/datasets \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --stage finetune_1 --crop 384 --index 0 --dtype float32 --rank-template "$REF_BUNDLE/batch_step003.pt" \
    --out perf/of3t_auxheads/bond_coverage_finetune1.json
echo "=== bond exit $? $(date -u +%FT%TZ) ==="
echo BOND_ALLDONE
