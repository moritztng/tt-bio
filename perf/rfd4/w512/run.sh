#!/bin/bash
# rfd4-perf-w512 device chain on one qb2 chip: what a second (or fourth) design in flight costs at 512 and 384.
# Books its chip, waits for the window, takes the chip flock, runs the legs in order; one log per leg in the outdir.
#   bash perf/rfd4/w512/run.sh <chip> <outdir> "<legs>"
# Legs:
#   n<N>-w<W>-d<D>-c<C>[-<tag>]  whole designs (b128/designs.py): width W, D designs a timed call, C timed calls.
#                                D = W is one trunk call of W designs; D = 2W is two interleaved lanes of width W.
#   c<N>-b<B>[-2g]               census: device kernel time per family, one eager call of 2 passes at batch B;
#                                -2g adds trimul_t1_2g (T1 may hold 2 GiB of outputs)
#   g<N>-w<W>                    31 designs on seeds 2000.. for the paired grade (grade.sh)
set -u
N=$1; R=$2; LEGS=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; ME=rfd4-perf-w512
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$ME PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $ME $N ${HOLD_S:-7200} --note 'RFD4 bet 36 w512: widths 1/2/4 at 512 and 384')"
until [ "$($BOOK active $N 2>/dev/null)" = $ME ]; do $BOOK add $ME $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
ALL=$($PY -c 'from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(",".join(sorted(L)))')
stamp "start chip $N tree $(git rev-parse --short HEAD) levers $ALL"
for leg in $LEGS; do
  case $leg in
    n*-w*-d*-c*) IFS=- read n w d c _ <<< "$leg"
      timeout 5400 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-${n:1}.yaml --weights $W --out $R/$leg \
        --width ${w:1} --designs ${d:1} --calls ${c:1} ;;
    c*-b*) IFS=- read n b x <<< "$leg"
      lv=$ALL; [ "${x:-}" = 2g ] && lv=$ALL,trimul_t1_2g
      TT_BIO_LEVERS=$lv bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n ${n:1} --batch ${b:1} \
        && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/$leg.json ;;
    g*-w*) IFS=- read n w <<< "$leg"
      timeout 7200 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-${n:1}.yaml --weights $W --out $R/$leg \
        --width ${w:1} --designs 31 --seed 2000 ;;
    *) echo "unknown leg $leg" ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $ME
stamp ALLDONE
