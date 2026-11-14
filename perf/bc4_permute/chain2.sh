#!/bin/bash
# bc4-bet-permute chain 2: the real BindCraft 2 loop (perf/spd/bc2_census.py, devgap's settings) under
# route_probe, arms base / fix / base. base = the operand-blind tape decline; fix = dc7157ea9.
WT=/home/ttuser/.coworker/wt/bc4-bet-permute
BR=$(git -C $WT rev-parse --short=9 HEAD); D=/home/moritz/.coworker/state/bc4/bets/permute/raw/chain2_$BR; mkdir -p $D
PY=$HOME/bcx_e2e_venv/bin/python
C=${TT_VISIBLE_DEVICES:?no leased card}
echo "start $(date +%s) sha $BR card $C dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) load $(cut -d" " -f1 /proc/loadavg)" > $D/card
export TT_BIO_LEASE_HOLDER=worker:bc4-bet-permute JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params
export TT_BIO_TRACE_RECOMPUTE=1 TT_BIO_TRACE_REGION_MB=2048
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$C/tt_aiclk)" "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> $D/aiclk.tsv & P=$!
run() { local n=$1 arm=$2
  echo "$n start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card
  (cd $WT && BC4_PERMUTE_ARM=$arm PYTHONPATH=$WT timeout -s INT -k 120 1800 $PY perf/bc4_permute/route_probe.py \
     $D/$n.json -- perf/spd/bc2_census.py --out $D/$n --at 4 > $D/$n.log 2>&1)
  echo "$n rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> $D/card; }
run a1 base; run b1 fix; run a2 base
kill $P; touch $D/done
