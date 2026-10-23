#!/bin/bash
# rfd4-land10 device chain on one qb2 chip for main bfd1581d7: device tests, an 8-seed 512 sanity against bf16z run2's
# stack designs (same seeds, bf16z's venv), board cells at 512/256/128 (6 designs, first cold), the host/device split
# per size, and census v7 at 512 and 256 (elt_prof with the pass_glue signposts). Books its chip, waits for the
# window, takes the chip flock, grades the sanity on the CPU after.
#   bash perf/rfd4/land10/run.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python; PYB=$HOME/tt-bio-dev/env/bin/python
BF=$HOME/scratch/rfd4-bf16z/tree-b3b3bb758/artifacts/bf16z/run2/design/stack
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-land10 PYTHONPATH=$WT
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add rfd4-land10 $N 4200)"
until [ "$($BOOK active $N 2>/dev/null)" = rfd4-land10 ]; do $BOOK add rfd4-land10 $N 4200 > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE) py $PY"
for leg in ${3:-devtests sanity d512 d256 d128 split512 split256 split128 census512 census256}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    sanity) timeout 2400 $PYB perf/rfd4/e2e/design.py --spec $IN/pdl1-512.yaml --weights $W --rng upstream \
                --seeds $(seq 0 7) --out $R/sanity ;;
    d*) n=${leg#d}; timeout 3600 $PY perf/rfd4/e2e/design.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 6 --rng upstream ;;
    split*) n=${leg#split}; timeout 2400 $PY perf/rfd4/e2e/host_split.py --spec $IN/pdl1-$n.yaml --weights $W \
                --out $R/$leg --designs 4 ;;
    census*) n=${leg#census}; bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-$n-v7.json ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm rfd4-land10
# sanity: ATOM-record identity per seed against bf16z run2's stack, and grade.py for the record
f() { ls $1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }
mkdir -p $R/grade-sanity
for k in $(seq 0 7); do
  a=$(f $R/sanity $k); b=$(f $BF $k)
  echo "s$k $(cmp -s <(grep ^ATOM $a) <(grep ^ATOM $b) && echo identical || echo DIFFERS)" >> $R/sanity-identity.txt
  $PY perf/rfd4/e2e/grade.py --ours $a --upstream $b --floor $(f $BF $(( (k+1)%8 ))) --out $R/grade-sanity/s$k.json \
    > /dev/null 2>&1 || echo "s$k grade failed" >> $R/sanity-identity.txt
done
stamp ALLDONE
