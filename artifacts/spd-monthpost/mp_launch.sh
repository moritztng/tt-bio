#!/bin/bash
# mp_launch.sh TAG CARD: stage release TAG's tree on qb1 and run the gapboard page protocol on it (p150a, one card).
# Out: qb1:~/spd/monthpost/out/<sha9>/<cell>/{result.json,run.log,aiclk.tsv}. Not under gapboard's harvest globs.
set -eu
TAG=$1 CARD=$2; R=/home/moritz/tt-bio; G=/home/ttuser/spd/monthpost
SHA=$(git -C $R rev-parse --short=9 "$TAG^{commit}")
SRC=$(mktemp -d); git -C $R archive "$TAG^{commit}" | tar -x -C $SRC; echo $SHA > $SRC/TREE_SHA
ssh -o BatchMode=yes qb1 "mkdir -p $G/trees $G/out/$SHA"
rsync -a --delete $SRC/ qb1:$G/trees/m$SHA/; rm -rf $SRC
rsync -a "$(dirname "$0")/mp_chain.sh" qb1:$G/mp_chain.sh
T=$G/trees/m$SHA O=$G/out/$SHA SEQ=scripts/gpu_vs_tt/fixtures/prot512.seq SEQ_SHA=141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d
{ for m in boltz2 protenix-v2 opendde openfold3 openbind esmfold2 esmfold2-fast rf3; do for mode in normal fast; do
    f=; [ $mode = fast ] && f=" --fast"
    echo "$m-$mode 7200 python perf/of3_4xpd/xmodel_ab.py --model $m --tree $T --size 512 --repeat 4 --label $mode$f --out $O/$m-$mode/result.json"
  done; done
  for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do b=7; tm=1800; [ $m = esmc-6b ] && b=1 tm=2400
    for mode in normal fast; do f=; [ $mode = fast ] && f=" --fast"
      echo "$m-$mode $tm python perf/wh-embed/embed_split_screen.py --model $m$f --seq-file $SEQ --expect-sha $SEQ_SHA --batch-size 8 --warmup 2 --repeat 7 --n-seqs $b --assert-batch $b --out $O/$m-$mode/result.json"
  done; done; } | ssh -o BatchMode=yes qb1 "cat > $O/jobs.txt"
[ "${NOJOBS:-}" ] && ssh -o BatchMode=yes qb1 ": > $O/jobs.txt"
[ "${NOSTART:-}" ] && exit 0
ssh -n -o BatchMode=yes qb1 "cd $G && setsid -f bash mp_chain.sh $CARD $SHA > out/$SHA/chain.out 2>&1 < /dev/null"
sleep 3; ssh -o BatchMode=yes qb1 "cat $O/chain.log"
