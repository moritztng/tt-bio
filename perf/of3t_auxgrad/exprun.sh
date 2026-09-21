#!/usr/bin/env bash
# Deliverable 2: a real OpenFold3 fold through the shipped CLI, with the confidence head run
# three ways per sample inside the spawned worker. Usage: exprun.sh <stem> <seed> [predict args]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-auxgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
# The hook dir must come first so `site` finds this sitecustomize, and it is inert unless
# EXPOSURE_HOOK=1. Both are inherited by the spawn child, which is the process that matters.
export PYTHONPATH="$W/perf/of3t_auxgrad/exposure_hook:$W"
export EXPOSURE_HOOK=1 EXPOSURE_SRC="$W/perf/of3t_auxgrad"
export OMP_NUM_THREADS=8
CARD=${CARD:-1}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=1,$CARD TT_BIO_LEASE_HOLDER=worker:of3t-auxgrad
STEM=$1; SEED=$2; shift 2
export EXPOSURE_OUT="$W/perf/of3t_auxgrad/$STEM.json"
echo "=== exposure $STEM seed $SEED start $(date -u +%FT%TZ) card $CARD ==="
"$PY" -m tt_bio.main predict "${TARGET:-examples/9bk6.yaml}" \
    --model openfold3 --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --out_dir "/home/ttuser/of3t_auxgrad_logs/out_$STEM" --seed "$SEED" \
    --msa_cache_only --msa_dir /home/ttuser/of3t_auxgrad_logs/msa "$@"
echo "=== exit $? $(date -u +%FT%TZ) ==="
