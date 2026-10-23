#!/bin/bash
# rfd4-land12 device chain on one qb2 chip for main e3ce63249+ (relu pair_block_hifi2 + tri6 multicast + host3, on
# top of t1b8 + bf16z): device tests; 31 seeds at 512 for the default (smain) and for the default minus the three
# bit-moving precision levers (sbase), which grades the stack as one; 8 seeds through the CLI`s --fast; 128 w16 vs w1
# and 256 w4 vs w1 (32 designs a side); census v9 at 512 b1 and 256 b4. Books its chip, waits for the window, takes
# the chip flock; grade.sh grades the outdir on the CPU after.
#   bash perf/rfd4/land12/run.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-land12
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-9000} --note "RFD4 land12 stack grade + census v9")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-9000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L - {\"pair_block_hifi2\", \"trimul_t1b8\", \"pair_bf16z\"})))")
stamp "start chip $N tree $(cat TREE 2>/dev/null) base=$BASE"
S="--spec $IN/pdl1-512.yaml --weights $W --rng upstream"
for leg in ${3:-devtests smain sbase sfast n128-w16 n128-w1 n256-w4 n256-w1 census512 census256}; do
  case $leg in
    devtests) timeout 3000 $PY -m pytest -q -p no:cacheprovider -m device --junitxml=$R/devtests.xml \
                tests/test_rfd4_*.py tests/test_trimul_*.py ;;
    smain) timeout 5400 $PY perf/rfd4/e2e/design.py $S --seeds $(seq 0 30) --out $R/smain ;;
    sbase) TT_BIO_LEVERS=$BASE timeout 5400 $PY perf/rfd4/e2e/design.py $S --seeds $(seq 0 30) --out $R/sbase ;;
    sfast) timeout 2400 $PY perf/rfd4/e2e/design.py $S --fast --seeds $(seq 0 7) --out $R/sfast ;;
    n*-w*) n=${leg:1:3}; w=${leg#*-w}; timeout 3600 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --width $w --designs 32 ;;
    census512) bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 512 \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-512-v9.json \
                && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-512.json > $R/top-ops-512.txt ;;
    census256) bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 256 --batch 4 \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-256-b4-v9.json \
                && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-256.json > $R/top-ops-256.txt ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
