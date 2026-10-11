#!/bin/bash
# run_qb1.sh CARD JOBFILE : run each line of JOBFILE (NAME TIMEOUT CMD...) on qb1 CARD, in order, each under the
# card's SPD flock, with AICLK of every node + loadavg sampled every 0.5 s during the job (OUT/aiclk.tsv).
# A job whose run.log already says rc=0 is skipped, so the chain can be re-launched. Never waits on a release leg.
set -u
CARD=$1 JOBS=$2
R=$HOME/spd/spd-gap-of3
export PATH=${VENV:-$HOME/scratch/shipflow/venv-d9118561c7fe}/bin:$PATH
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_HOLDER=spd-gap-of3
while read -r N TMO TREE CMD; do
  [ -z "$N" ] || [ "${N:0:1}" = "#" ] && continue
  OUT=$R/out/$N; mkdir -p "$OUT"
  grep -q "end rc=0" "$OUT/run.log" 2>/dev/null && continue
  say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
  while pgrep -f "release_gate.py|gate_fanout.py" >/dev/null; do sleep 60; done
  exec 9> "/home/ttuser/spd_qb1_card$CARD.lock"
  flock -w 14400 9 || { say "flock busy"; continue; }
  say "start card $CARD tree $(cat $TREE/TREE_SHA) cmd: $CMD"
  ( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do
      printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done
  ) >> "$OUT/aiclk.tsv" 2>/dev/null &
  SAMP=$!
  ( cd "$TREE" && PYTHONPATH=$TREE timeout -s TERM $((TMO + 120)) timeout -s INT "$TMO" $CMD --out "$OUT/result.json" ) \
    > "$OUT/harness.log" 2>&1
  rc=$?
  kill $SAMP 2>/dev/null
  exec 9>&-
  say "end rc=$rc"
done < "$JOBS"
echo "$(date -u +%FT%TZ) chain $CARD $JOBS done" >> $R/chains.log
