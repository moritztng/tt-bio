#!/bin/bash
# bc4-bet-gates: the real BindCraft 2 loop (perf/spd/bc2_census.py, devgap's settings) under
# perf/bc4_gates/route_probe.py, arms base / fix / base. base = every gate blind to its operands;
# fix = this branch. Waits for the card's fleet lock, samples AICLK and load at 1 Hz. Runs a
# snapshot of HEAD (git archive), so the worktree may move on while it waits.
#   CARD=1 bash perf/bc4_gates/chain.sh
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
CARD=${CARD:?CARD=<leased chip>}
BR=$(git rev-parse --short=9 HEAD); D=/home/moritz/.coworker/state/bc4/bets/gates/raw/chain_$BR; mkdir -p $D
S=$WT/_bc4out/src_$BR; rm -rf $S; mkdir -p $S; git archive $BR | tar -x -C $S; cd $S
PY=$HOME/bcx_e2e_venv/bin/python
export PYTHONPATH=$S TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bc4-bet-gates}
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params TT_BIO_TRACE_RECOMPUTE=1 TT_BIO_TRACE_REGION_MB=2048
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
echo "start $(date +%s) sha $BR card $CARD snapshot=$S load $(cut -d" " -f1 /proc/loadavg)" > $D/card
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> $D/aiclk.tsv 9>&- & P=$!
trap "kill $P" EXIT
run() { local n=$1 arm=$2
  echo "$n start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
  BC4_GATES_ARM=$arm timeout -s INT -k 120 1800 $PY perf/bc4_gates/route_probe.py $D/$n.json -- \
     perf/spd/bc2_census.py --out $D/$n --at 4 > $D/$n.log 2>&1
  echo "$n rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card; }
run a1 base; run b1 fix; run a2 base
touch $D/done
