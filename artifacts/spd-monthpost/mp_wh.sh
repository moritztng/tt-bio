#!/bin/bash
# mp_wh.sh TAG BOX [PRIO]: stage release TAG's tree on SPD Galaxy 172.16.102.BOX and drop the WH page-protocol cells
# (whmain/gen.sh's exact command lines, spd-orchestrator's run_cell.sh) into the pool backlog as row spd-monthpost.
# Out: ~/spd/spd-monthpost/wh<sha9>/<cell>/ (outside gapboard's ~/spd/spd-orchestrator/wh*/* harvest glob).
set -eu
TAG=$1 BOX=$2 PRIO=${3:-4}; R=/home/moritz/tt-bio; H=ubuntu@172.16.102.$BOX; o="-o BatchMode=yes -o ConnectTimeout=10"
sha=$(git -C $R rev-parse --short=9 "$TAG^{commit}")
T=/home/ubuntu/spd/spd-monthpost/trees/m$sha
SRC=$(mktemp -d); git -C $R archive "$TAG^{commit}" | tar -x -C $SRC; echo $sha > $SRC/TREE_SHA
ssh $o $H "mkdir -p $T ~/spd/pool/backlog/spd-monthpost"
rsync -a --delete -e "ssh $o" $SRC/ $H:$T/; rm -rf $SRC
q() { ssh $o $H "j=$PRIO-spd-monthpost-wh$sha-$1.sh; cd ~/spd/pool/backlog/spd-monthpost && cat > .\$j && mv .\$j \$j"; }
for m in boltz2 protenix-v2 opendde openfold3 openbind esmfold2 esmfold2-fast rf3; do for mode in normal fast; do
  O=/home/ubuntu/spd/spd-monthpost/wh$sha/$m-$mode
  cmd="python perf/of3_4xpd/xmodel_ab.py --model $m --tree $T --size 512 --repeat 4 --label $mode $([ $mode = fast ] && echo --fast || :) --out $O/result.json"
  printf '# spd-monthpost: %s %s on %s (%s), WH page protocol (cold + 4 warm)\n# mem: 20\nTREE=%s bash /home/ubuntu/spd/spd-orchestrator/run_cell.sh $CHIP %s 7200 -- %s\n' "$m" "$mode" "$TAG" "$sha" "$T" "$O" "$cmd" | q $m-$mode
done; done
for m in esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m saprot-1.3b; do for mode in normal fast; do
  bs=$([ $m = esmc-6b ] && echo 1 || echo 7); O=/home/ubuntu/spd/spd-monthpost/wh$sha/$m-$mode
  cmd="python perf/wh-embed/embed_split_screen.py --model $m $([ $mode = fast ] && echo --fast || :) --seq-file scripts/gpu_vs_tt/fixtures/prot512.seq --expect-sha 141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d --batch-size 8 --warmup 2 --repeat 7 --n-seqs $bs --assert-batch $bs --out $O/result.json"
  printf '# spd-monthpost: %s %s on %s (%s), WH embed protocol\n# mem: 20\nTREE=%s bash /home/ubuntu/spd/spd-orchestrator/run_cell.sh $CHIP %s 2400 -- %s\n' "$m" "$mode" "$TAG" "$sha" "$T" "$O" "$cmd" | q $m-$mode
done; done
ssh $o $H "ls ~/spd/pool/backlog/spd-monthpost | grep -c wh$sha"
