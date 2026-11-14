#!/bin/bash
# rfd4-perf-b128 device pass 5, on main with pair_bf16z + pair_tail_cm in RFD4's default. The batched back move's test
# and the triangle module at b16, then whole designs seed-matched across widths, 32 a side: 128 width 16 with and
# without pair_batch, width 1 default, width 1 + pair_batch (must be the default's bits), 256 width 4 with and without
# pair_batch. grade.sh grades the outdir. A red kernel test skips every pair_batch arm.
#   run5.sh <chip> <tree> <out> [designs, default 32]
set -u
C=$1; T=$2; R=$3; D=${4:-32}
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
W=$HOME/rfd4-bundle
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
book() { python3 $HOME/spd/feed/book.py add rfd4-perf-b128 $C ${HOLD_S:-3600} --note "RFD4 bet 28 b128 run5" | cut -d' ' -f1; }
start=$(book); stamp "booked start $start"
while [ "$(date +%s)" -lt "$start" ]; do sleep 60; start=$(book); done
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 14400 9 || { stamp "no lock"; exit 3; }
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-b128 PYTHONPATH=$T:$T/perf
cd $T
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L - {'pair_batch'})))")
stamp "start sha $(cat $T/SHA 2>/dev/null) chip $C"
arm() {  # name n width levers designs
  case "$4" in *pair_batch*) [ "$pk" -ne 0 ] && { stamp "$1 skipped, pk-test red"; return; };; esac
  TT_BIO_LEVERS=$4 timeout 3600 $PY perf/rfd4/b128/designs.py --spec $HOME/rfd4-inputs/pdl1-$2.yaml --weights $W \
    --out $R/$1 --width $3 --designs $5 > $R/$1.log 2>&1
  stamp "$1 rc=$?"
}
timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/test_permute_back_pk.py > $R/pk-test.log 2>&1
pk=$?; stamp "pk-test rc=$pk"
TT_BIO_LEASE_CARDS=$C timeout 1800 $PY perf/rfd4/b128/module.py --weights $W --out $R/module-b16.json --batch 16 \
  --arms "$BASE|$BASE,pair_batch" > $R/module-b16.log 2>&1
stamp "module-b16 rc=$?"
arm n128-w16-pb 128 16 $BASE,pair_batch $D
arm n128-w16-base 128 16 $BASE $D
arm n128-w1-base 128 1 $BASE $D
arm n128-w1-pb 128 1 $BASE,pair_batch 8
arm n256-w4-pb 256 4 $BASE,pair_batch $D
arm n256-w4-base 256 4 $BASE $D
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-b128 >/dev/null
stamp ALLDONE
