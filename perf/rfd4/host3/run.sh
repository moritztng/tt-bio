#!/bin/bash
# rfd4-perf-host3 device chain on one qb2 chip: the fair harness at 512 and 256, a batch of one design (A) against a
# batch of two designs run as two interleaved width-1 lanes (B), ABAB rounds so drift hits both arms. Same measure
# seeds on both arms, so each B batch's design 0 is the A batch's design on that seed (identity check after). Each round
# writes its own directory: a design CIF is never overwritten, and A in two rounds is the run-to-run identity control.
#   flock ~/spd_qb2_chip<N>.lock bash perf/rfd4/host3/run.sh <chip> <outdir>
# The quality grade pairs design 0 of every B batch with A on 31 seeds:
#   ROUNDS=g SIZES=256 NA=31 NB=31 SEED=2000 ... run.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-host3 PYTHONPATH=$WT
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "start chip $N sha $(git rev-parse HEAD) py $PY"
# A: 10 batches of 1; B: 5 batches of 2 as lanes of 1. Three rounds = 30 designs a side a size.
NA=${NA:-10}; NB=${NB:-5}; SEED=${SEED:-1000}
for round in ${ROUNDS:-1 2 3}; do
  for n in ${SIZES:-512 256}; do
    for arm in A B; do
      case $arm in
        A) args="--batch 1 --batches $NA" ;;
        B) args="--batch 2 --lane-width 1 --batches $NB" ;;
      esac
      timeout 3600 $PY perf/rfd4/fair/run.py --spec $IN/pdl1-$n.yaml --weights $W --out $R/r$round/$n-$arm --part p300c \
        --seed $SEED $args --max-foreign-cores 99 --shared-box > $R/r$round-$n-$arm.log 2>&1
      stamp "r$round $n $arm rc=$?"
    done
  done
done
stamp ALLDONE
