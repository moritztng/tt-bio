#!/bin/bash
# relu run2: the trunk's z / s vs float64 with pair_block_hifi2 (zprec.py, 256 tokens), then 31 seeds at 512 per arm,
# RFD4 default vs default + pair_block_hifi2, graded per seed (grade.py, floor = base seed k vs k+1) and as a population
# (quality.py binder). Books the card on qb2's arbiter and waits for it.
#   perf/rfd4/relu/run2.sh <card> [legs]     (run from the tree on qb2)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/relu/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-zprec256 design grade}; OWN=rfd4-perf-relu
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$W:$W/perf
mkdir -p $O/design $O/grade; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "book"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = $OWN ]; do
  python3 $HOME/spd/feed/book.py add $OWN $C 6000 --note "RFD4 bet 32 relu run2 grade" >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
S=$B,pair_block_hifi2
f() { ls $O/design/$1/pdl1_binder_512_s$2_r*_generated.cif | tail -1; }   # s0 also has the cold run r0
for leg in $LEGS; do
  case $leg in
    zprec*) n=${leg#zprec}; timeout 3600 $PY perf/rfd4/bf16z/zprec.py --n $n --lever pair_block_hifi2 \
              --out $O/$leg.json > $O/$leg.log 2>&1 ;;
    design) for a in base stack; do L=$([ $a = base ] && echo $B || echo $S)
              TT_BIO_LEVERS=$L timeout 5400 $PY perf/rfd4/e2e/design.py --spec $HOME/rfd4-inputs/pdl1-512.yaml \
                --weights $HOME/rfd4-bundle --out $O/design/$a --rng upstream --seeds $(seq 0 30) > $O/design/$a.log 2>&1
              stamp "design $a rc=$?"; done ;;
    grade) exec 9>&-
           python3 $HOME/spd/feed/book.py rm $OWN
           for k in $(seq 0 30); do $PY perf/rfd4/e2e/grade.py --ours $(f stack $k) --upstream $(f base $k) \
               --floor $(f base $(( (k+1)%31 ))) --out $O/grade/s$k.json >/dev/null 2>&1; done
           cp $HOME/scratch/rfd4-land5/QUALITY.base.json $O/quality512.json
           $PY perf/rfd4/quality.py grade --feature binder --tt $O/design/stack --ref $O/design/base --allow-no-oracle \
             --label "RFD4 default + pair_block_hifi2 vs default, 512 PD-L1 binder, 31 seeds" \
             --out $O/quality512.json > $O/quality512.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
python3 $HOME/spd/feed/book.py rm $OWN 2>/dev/null
stamp ALLDONE
