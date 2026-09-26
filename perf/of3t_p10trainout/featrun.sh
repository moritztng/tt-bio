#!/bin/bash
# Dump the graded corpus. CPU only, no card.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10trainout
PY=/home/ttuser/tt-bio-dev/env/bin/python
D=/home/ttuser/of3t-data-xhost/datasets
O=/home/ttuser/of3t_p10trainout/corpus
L=/tmp/of3t/p10trainout
cd "$W" || exit 1
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W")"
ref_assert "$PY"
export OMP_NUM_THREADS=${OMP:-6}
mkdir -p "$L" "$O"
split=$1; cache=$2; extra=${3:-}
echo "=== featurise $split start $(date -u +%FT%TZ) ===" >> "$L/feat.log"
nice -n 19 "$PY" perf/of3t_p10trainout/featurise.py \
    --data-dir "$D" --cache-file "$D/$cache" --split "$split" \
    --stage initial_training --crop 384 --seed 20260926 \
    --rank-template "$REF_BUNDLE/batch_step003.pt" \
    --out "$O/$split" $extra >> "$L/feat_$split.log" 2>&1
echo "=== featurise $split done $(date -u +%FT%TZ) rc=$? ===" >> "$L/feat.log"
