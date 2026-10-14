#!/bin/bash
# mp_heavy.sh CHIP OUT (on qb1): the c730 heavy fold, old v0.11.0 vs main, on ONE p150a card under its flock.
# OUT/jobs.txt lines "NAME TIMEOUT TREE ARGS..." -> cd TREE; python perf/spd/bench_mp.py --out OUT/NAME --chip CHIP ARGS
# (main's perf/spd/bench.py overlaid as bench_mp.py, engine = the tree's own tt_bio). bench.py samples AICLK per rep.
set -u
CHIP=$1 O=$2
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/ | head -1); export PATH=${VENV%/}/bin:$PATH
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=spd-monthpost TT_BIO_LEASE_DIR=$HOME/spd/monthpost/leases
REL="(spd_qb1_card|TT_VISIBLE_DEVICES=)${CHIP}[. ].*(release_gate|gate_fanout|perf_ab)"
rel() { pgrep -f "$REL" >/dev/null; }
take() { while :; do while rel; do sleep 60; done
  exec 9> "/home/ttuser/spd_qb1_card$CHIP.lock"; flock -w 43200 9 || return 1
  rel || return 0; exec 9>&-; sleep 60; done; }
echo "$(date -u +%FT%TZ) heavy chain start card $CHIP pid $$" >> $O/chain.log
take || { echo "$(date -u +%FT%TZ) flock timeout" >> $O/chain.log; exit 3; }
while read -r N TM TREE ARGS; do
  [ -z "$N" ] || [ "${N:0:1}" = "#" ] && continue
  [ -f $O/$N/run.log ] && grep -q "end rc=0" $O/$N/run.log && continue
  rel && { exec 9>&-; take || exit 3; }
  mkdir -p $O/$N; echo "$(date -u +%FT%TZ) start chip $CHIP tree $(cat $TREE/TREE_SHA 2>/dev/null) args: $ARGS" >> $O/$N/run.log
  ( cd $TREE && PYTHONPATH=$TREE timeout -s TERM $((TM + 120)) timeout -s INT "$TM" python perf/spd/bench_mp.py --out $O/$N --chip $CHIP $ARGS ) > $O/$N/harness.log 2>&1 9>&-
  echo "$(date -u +%FT%TZ) end rc=$?" >> $O/$N/run.log
done < $O/jobs.txt
echo "$(date -u +%FT%TZ) chain done" >> $O/chain.log
