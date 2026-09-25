#!/usr/bin/env bash
# The gradient arm, on pc. CPU only, no card.
#
# Crop 256, not finetune_1's own 640 or the campaign's usual 384: at 384 the forward finishes in
# 269 s and the backward is OOM-killed at 22 GB RSS on a 30 GB box. The bond term does not depend
# on the crop beyond the crop carrying the bond, and the run asserts that for the batch it drew.
set -uo pipefail
W=${W:-/home/moritz/.coworker/wt/of3t-bondcov}
S=${S:-/home/moritz/.coworker/scratch/of3t-bondcov}
cd "$W"
export PYTHONPATH="$S/of3pkg043:$W"          # openfold3 0.4.3, the campaign's pinned release
export OMP_NUM_THREADS=${OMP:-6}
IDX=${1:-12}                                  # 12 = 4g5j chain 1; see the datapoint_cache
TAG=${2:-4g5j_c1}
echo "=== bond gradient idx $IDX crop ${CROP:-256} start $(date -u +%FT%TZ) ==="
nice -n 19 /home/moritz/of3-upstream-venv/bin/python perf/of3t_auxheads/bond_coverage.py \
    --package openfold3 \
    --data-dir "$S/datasets" \
    --cache-file "$S/datasets/training_cache_with_templates_subset_2.json" \
    --checkpoint /home/moritz/.boltz/of3-p2-155k.pt \
    --stage finetune_1 --crop "${CROP:-256}" --index "$IDX" --dtype float32 \
    --rank-template "$S/batch_step003.pt" \
    --out "perf/of3t_bondcov/bond_gradient_${TAG}.json"
echo "=== exit $? $(date -u +%FT%TZ) ==="
