#!/bin/bash
# spd-bheth on qb1, after bheth_chain_qb1.sh proved the 12x10 grid on c730: does Ethernet dispatch fold every size and
# model a Blackhole user runs today, and with what digests? Same venv (ttnn 0.68.0+bh.eth2), one p150a card by flock.
# Protenix-v2 l256..l1536 1 cold + 1 warm, then OpenDDE, OpenFold3, Boltz-2 on c730 cold only; Tensix vs ETH each.
#   [QB1_CARD=N] bheth_ladder_qb1.sh ENGINE_SHA
set -u
SHA=${1:?engine sha}
B=~/spd-bheth; R=$B/runs/ladder-$(date -u +%m%dT%H%M); mkdir -p $R
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/queue.log; }
node_of(){ case $1 in 0) echo 1;; 1) echo 2;; 2) echo 3;; 3) echo 0;; esac; }   # qb1 logical -> /dev/tenstorrent node
PY=$B/venv/bin/python
git -C $B/tree fetch -q origin && git -C $B/tree checkout -q $SHA || { say "no engine $SHA"; exit 1; }
cd $B/tree; export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=spd-bheth
$PY -c "from tt_bio import metal_overlay as M; assert M.bh_eth_dispatch_supported()" || { say "capability check failed"; exit 1; }
say "engine $(git rev-parse --short HEAD), ttnn $($PY -c 'import importlib.metadata as m; print(m.version("ttnn"))')"

CARD=${QB1_CARD:-0}; exec 9>~/spd_qb1_card$CARD.lock
flock -w 43200 9 || { say "no qb1 card $CARD in 12 h"; exit 1; }
N=$(node_of $CARD); export TT_VISIBLE_DEVICES=$CARD
dead(){ [ "$(cat /sys/class/tenstorrent/tenstorrent!$N/tt_heartbeat 2>/dev/null)" = 4294967295 ]; }
say "lock held: qb1 card $CARD (node $N), load $(cut -d' ' -f1-3 /proc/loadavg)"

run(){  # name model inputs warm arm
  nice -n 5 timeout -s INT 5400 $PY perf/spd/bench.py --out $R/$1 --chip $CARD --model $2 --inputs $3 --warm $4 --arm "$5" \
    > $R/$1.log 2>&1
  say "$1 rc=$? $(grep -c '"ev": "rep"' $R/$1.log) reps, $(grep -c '"err": null' $R/$1.log) clean"
  dead && { say "node $N ARC dead after $1, stopping"; exit 2; }
}
for m in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth; do run pv2_${m%%:*} protenix-v2 l256,l512,l1024,l1536 1 $m; done
for model in opendde openfold3 boltz2; do
  for m in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth; do run ${model}_${m%%:*} $model c730 0 $m; done
done
say "end"
