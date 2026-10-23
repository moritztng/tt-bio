#!/bin/bash
# rfd4-perf-b128 device pass 1 at 128 tokens: census of the default at batch 1 / 4 / 16 / 32, the triangle module at
# batch 4 and 16 with and without pair_batch, then the census at batch 16 / 32 with pair_batch.
#   run1.sh <chip> <tree> <out>
set -u
C=$1; T=$2; R=$3
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
book() { python3 $HOME/spd/feed/book.py add rfd4-perf-b128 $C ${HOLD_S:-5400} --note "RFD4 bet 28 b128 run1" | cut -d' ' -f1; }
start=$(book); stamp "booked start $start"
while [ "$(date +%s)" -lt "$start" ]; do sleep 60; start=$(book); done
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 14400 9 || { stamp "no lock"; exit 3; }
export TT_BIO_LEASE_HOLDER=worker:rfd4-perf-b128
stamp "start sha $(cat $T/SHA 2>/dev/null) chip $C"
BASE=$(cd $T && PYTHONPATH=$T $PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))")
for b in 1 16 4 32; do
  bash $T/perf/rfd4/e2e/elt_prof.sh $R/census-b$b $C --batch $b > $R/census-b$b.log 2>&1
  stamp "census-b$b rc=$?"
done
for b in 16 4; do
  (cd $T && TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C PYTHONPATH=$T:$T/perf timeout 1800 $PY perf/rfd4/b128/module.py \
    --weights $W --out $R/module-b$b.json --batch $b --arms "$BASE|$BASE,pair_batch") > $R/module-b$b.log 2>&1
  stamp "module-b$b rc=$?"
done
for b in 16 32; do
  TT_BIO_LEVERS=$BASE,pair_batch bash $T/perf/rfd4/e2e/elt_prof.sh $R/census-pb-b$b $C --batch $b > $R/census-pb-b$b.log 2>&1
  stamp "census-pb-b$b rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-b128 >/dev/null
stamp ALLDONE
