#!/bin/bash
# rfd4-perf-p512: A (main) vs P (two trunk calls in flight) on the fair harness path (perf/rfd4/fair/run.py), ABBA,
# on one qb2 chip whose p300 board-mate is held idle, so the board's clock is this run's alone.
#   A=<sha> P=<sha> bash perf/rfd4/p512/abba.sh <chip> <outdir> [phases]
# phases (default "n512 n224 dram"): n512 = 16 legs b4l1, 2 warm batches each; n224 = 8 legs b20l10, 1 batch each;
# dram = one --dram leg per arm at 512 b4l1. Every leg: <outdir>/<phase>/<leg>/runs.jsonl + designs.pt per batch.
set -u
N=$1; R=$2; PHASES=${3:-n512 n224 dram}; WT=$(cd "$(dirname "$0")/../../.." && pwd); OWN=rfd4-perf-p512
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; MATE=$((N ^ 1))
mkdir -p $R; R=$(cd $R && pwd)
for arm in A P; do t=$R/tree-$arm; [ -d $t ] && continue; sha=${!arm}; mkdir -p $t
  git -C $WT archive $sha | tar -x -C $t; echo $sha > $t/TREE
  rm -rf $t/perf/rfd4/p512; cp -r $WT/perf/rfd4/p512 $t/perf/rfd4/p512; done  # one tool, arms differ in tt_bio
node() { for s in /sys/class/tenstorrent/*; do echo "$(basename "$(readlink -f "$s/device")") ${s##*!}"; done |
  sort | sed -n "$(($1 + 1))s/.* //p"; }
held() { fuser /dev/tenstorrent/$(node $N) /dev/tenstorrent/$(node $MATE) 2>&1 | grep -o '[0-9]\+' | tr '\n' ' '; }
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg) held[$(held)]" >> $R/done; }
stamp "booked $($BOOK add $OWN $N,$MATE ${HOLD_S:-12600} --note "p512 ABBA, board pair idle")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N,$MATE ${HOLD_S:-12600} > /dev/null; sleep 60; done
exec 8>"$HOME/spd_qb2_chip$N.lock"; flock -w 14400 8 || { stamp "flock $N timeout"; exit 3; }
exec 9>"$HOME/spd_qb2_chip$MATE.lock"; flock -w 14400 9 || { stamp "flock $MATE timeout"; exit 3; }
until [ -z "$(held)" ]; do sleep 20; done
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp "start chip $N (mate $MATE idle) A=$(cat $R/tree-A/TREE) P=$(cat $R/tree-P/TREE)"
leg() {  # phase leg size batch lane batches [extra]
  local t=$R/tree-${2:0:1} o=$R/$1/$2
  until [ -z "$(held)" ]; do sleep 20; done
  (cd $t && PYTHONPATH=$t timeout 3600 $PY perf/rfd4/p512/leg.py ${7:-} --spec $IN/pdl1-$3.yaml --weights $W \
     --out $o --part dev --shared-box --batch $4 --lane-width $5 --batches $6 --max-attempts $6 \
     --max-foreign-cores 4 --quiet-timeout 300) > $o.log 2>&1
  stamp "$1/$2 rc=$?"
}
mkdir -p $R/n512 $R/n224 $R/dram
for p in $PHASES; do case $p in
  n512) for l in A1 P1 P2 A2 A3 P3 P4 A4 A5 P5 P6 A6 A7 P7 P8 A8; do leg n512 $l 512 4 1 2; done ;;
  n224) for l in A1 P1 P2 A2 A3 P3 P4 A4; do leg n224 $l 224 20 10 1; done ;;
  dram) for l in A1 P1; do leg dram $l 512 4 1 1 --dram; done ;;
esac; done
exec 8>&- 9>&-
$BOOK rm $OWN > /dev/null
stamp ALLDONE
