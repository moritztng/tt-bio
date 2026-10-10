#!/bin/bash
# spd-bheth on qb1, after bheth_chain_qb1.sh proved the 12x10 grid on c730: does Ethernet dispatch fold every size and
# model a Blackhole user runs today, and with what digests? Same venv (ttnn 0.68.0+bh.eth2), one p150a card by flock.
# Protenix-v2 l256..l1536 1 cold + 1 warm, then OpenDDE, OpenFold3, Boltz-2 on c730 cold only; Tensix vs ETH each.
#   [QB1_CARDS="1 0 2 3"] [AFTER_PID=N] bheth_ladder_qb1.sh ENGINE_SHA [repeat|rest]
# rest: what the 10-10 ladder lost when qb1's disk filled at pv2_eth l1536 (ETH l1536, the other models), then repeat.
# Every plan waits for 20 GiB free on / before it takes a card: a full disk turns each fold into rc=120 in a second.
# repeat: Protenix-v2 l256 + l1024 again in both arms, same seeds, to tell a run-to-run flip from a grid-numerics change
# (the first ladder's ETH digests differed from Tensix at l256 and l1024, matched at l512).
# Takes the first listed card that is free and has no release-gate leg (flock_first.sh) waiting on it, so it never
# jumps a release leg. It polls instead of blocking in flock: a blocked waiter gets SIGSTOPped by flock_first.sh and
# its -w timer fires the moment it is continued (10-10, 12 h lost that way).
set -u
SHA=${1:?engine sha}; PLAN=${2:-ladder}
while [ -n "${AFTER_PID:-}" ] && kill -0 $AFTER_PID 2>/dev/null; do sleep 60; done
B=~/spd-bheth; R=$B/runs/$PLAN-$(date -u +%m%dT%H%M); mkdir -p $R
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/queue.log; }
node_of(){ case $1 in 0) echo 1;; 1) echo 2;; 2) echo 3;; 3) echo 0;; esac; }   # qb1 logical -> /dev/tenstorrent node
PY=$B/venv/bin/python
git -C $B/tree fetch -q origin && git -C $B/tree checkout -q $SHA || { say "no engine $SHA"; exit 1; }
cd $B/tree; export PYTHONPATH=$PWD TT_BIO_LEASE_HOLDER=spd-bheth
$PY -c "from tt_bio import metal_overlay as M; assert M.bh_eth_dispatch_supported()" || { say "capability check failed"; exit 1; }
say "engine $(git rev-parse --short HEAD), ttnn $($PY -c 'import importlib.metadata as m; print(m.version("ttnn"))')"

take(){  # poll the cards for up to 12 h; on success fd 9 holds card $CARD
  local end=$(( $(date +%s) + 43200 )) c
  while [ "$(date +%s)" -lt $end ]; do
    [ "$(df -BG --output=avail / | tail -1 | tr -dc 0-9)" -ge 20 ] || { sleep 60; continue; }
    for c in ${QB1_CARDS:-1 0 2 3}; do
      pgrep -f "flock_first.sh $HOME/spd_qb1_card$c.lock " >/dev/null && continue
      exec 9>>~/spd_qb1_card$c.lock; flock -n 9 && { CARD=$c; return 0; }; exec 9>&-
    done
    sleep 30
  done
  return 1
}
take || { say "no qb1 card in 12 h"; exit 1; }
N=$(node_of $CARD); export TT_VISIBLE_DEVICES=$CARD
dead(){ [ "$(cat /sys/class/tenstorrent/tenstorrent!$N/tt_heartbeat 2>/dev/null)" = 4294967295 ]; }
say "lock held: qb1 card $CARD (node $N), load $(cut -d' ' -f1-3 /proc/loadavg)"

run(){  # name model inputs warm arm
  nice -n 5 timeout -s INT 5400 $PY perf/spd/bench.py --out $R/$1 --chip $CARD --model $2 --inputs $3 --warm $4 --arm "$5" \
    > $R/$1.log 2>&1
  say "$1 rc=$? $(grep -c '"ev": "rep"' $R/$1.log) reps, $(grep -c '"err": null' $R/$1.log) clean"
  dead && { say "node $N ARC dead after $1, stopping"; exit 2; }
}
if [ $PLAN = rest ]; then
  run pv2_eth protenix-v2 l1536 1 eth:TT_BIO_BH_ETH_DISPATCH=1
  for model in opendde openfold3 boltz2; do
    for m in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth:TT_BIO_BH_ETH_DISPATCH=1; do run ${model}_${m%%:*} $model c730 0 $m; done
  done
  PLAN=repeat; R=$R/repeat; mkdir -p $R
fi
if [ $PLAN = repeat ]; then
  for m in eth:TT_BIO_BH_ETH_DISPATCH=1 tensix:TT_BIO_BH_ETH_DISPATCH=0; do run pv2_${m%%:*} protenix-v2 l256,l1024 1 $m; done
  say "end"; exit 0
fi
for m in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth:TT_BIO_BH_ETH_DISPATCH=1; do run pv2_${m%%:*} protenix-v2 l256,l512,l1024,l1536 1 $m; done
for model in opendde openfold3 boltz2; do
  for m in tensix:TT_BIO_BH_ETH_DISPATCH=0 eth:TT_BIO_BH_ETH_DISPATCH=1; do run ${model}_${m%%:*} $model c730 0 $m; done
done
say "end"
