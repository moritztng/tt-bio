#!/bin/bash
# rfd4-land11 device chain on one qb2 chip for main 5c3f66b63+ (trimul_t1b8 + pair_batch, both default-on): device
# tests, whole designs seed-matched across widths (128 w16 vs w1, 256 w4 vs w1, 32 a side, default levers), an 8-seed
# 512 sanity on land10's seeds and venv, a 512 cell, census v8 at 512 b1 and 256 b4. Books its chip, waits for the
# window, takes the chip flock; grade.sh grades the outdir on the CPU after.
#   bash perf/rfd4/land11/run.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; PYB=$HOME/tt-bio-dev/env/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-land11 PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add rfd4-land11 $N ${HOLD_S:-5400} --note 'RFD4 land11 grade + census v8')"
until [ "$($BOOK active $N 2>/dev/null)" = rfd4-land11 ]; do $BOOK add rfd4-land11 $N ${HOLD_S:-5400} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE) levers $($PY -c 'from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(",".join(sorted(L)))')"
for leg in ${3:-devtests n128-w16 n128-w1 n256-w4 n256-w1 sanity d512 census512 census256}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    n*-w*) n=${leg:1:3}; w=${leg#*-w}; timeout 3600 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --width $w --designs 32 ;;
    sanity) timeout 2400 $PYB perf/rfd4/e2e/design.py --spec $IN/pdl1-512.yaml --weights $W --rng upstream \
                --seeds $(seq 0 7) --out $R/sanity ;;
    d512) timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-512.yaml --weights $W \
                --out $R/d512 --designs 6 --rng upstream ;;
    census512) bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 512 \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-512-v8.json ;;
    census256) bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 256 --batch 4 \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-256-b4-v8.json ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm rfd4-land11
stamp ALLDONE
