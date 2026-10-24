#!/bin/bash
# rfd4-perf-cfg512: the 512 run configuration on main, fair harness path (perf/rfd4/fair/run.py through leg.py),
# on one qb2 chip whose p300 board-mate is held idle. Configs differ only in run.py's --batch at lane width 1.
#   bash perf/rfd4/cfg512/run.sh <chip> <outdir> [legs]
# legs (default "A1 B1 C1 C2 B2 A2"): A = b4l1 (fair7), B = b8l1, C = b16l1; every leg measures 16 designs on seeds
# from 1010 (A 4 batches, B 2, C 1), so seed 1010's designs 0-3 exist in every leg. Out: <outdir>/<leg>/runs.jsonl,
# designs.pt per batch, <outdir>/done.
set -u
N=$1; R=$2; LEGS=${3:-A1 B1 C1 C2 B2 A2}; SIZE=${SIZE:-512}
WT=$(cd "$(dirname "$0")/../../.." && pwd); OWN=rfd4-perf-cfg512
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; MATE=$((N ^ 1))
mkdir -p $R; R=$(cd $R && pwd); T=$R/tree
if [ ! -d $T ]; then mkdir -p $T; git -C $WT archive HEAD | tar -x -C $T; git -C $WT rev-parse HEAD > $T/TREE; fi
node() { for s in /sys/class/tenstorrent/*; do echo "$(basename "$(readlink -f "$s/device")") ${s##*!}"; done |
  sort | sed -n "$(($1 + 1))s/.* //p"; }
held() { fuser /dev/tenstorrent/$(node $N) /dev/tenstorrent/$(node $MATE) 2>/dev/null | grep -o '[0-9]\+' | tr '\n' ' '; }
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg) held[$(held)]" >> $R/done; }
stamp "booked $($BOOK add $OWN $N,$MATE ${HOLD_S:-12600} --note "cfg512 ABBA, board pair idle")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N,$MATE ${HOLD_S:-12600} > /dev/null; sleep 60; done
# A release-gate process naming either chip has absolute priority: a leg starts only after 120 s with no gate and no fd.
gate() { pgrep -af "shipflow|RELEASE_GATE_RUN_DIR|release_gate|gate_fanout" | grep -v -e pgrep -e book.py |
  grep -qE "(TT_VISIBLE_DEVICES=|spd_qb2_chip)($N|$MATE)[^0-9]"; }
quiet() { local q=0; while [ $q -lt 6 ]; do $BOOK add $OWN $N,$MATE ${HOLD_S:-12600} > /dev/null; if gate || [ -n "$(held)" ]; then q=0; else q=$((q + 1)); fi; sleep 20; done; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
quiet; stamp "start chip $N (mate $MATE idle) tree=$(cat $T/TREE) size=$SIZE"
leg() {  # leg batch batches
  local o=$R/$1
  quiet
  exec 8>"$HOME/spd_qb2_chip$N.lock" 9>"$HOME/spd_qb2_chip$MATE.lock"
  flock -w 600 8 && flock -w 600 9 || { exec 8>&- 9>&-; stamp "$1 lock busy, retry"; leg "$@"; return; }
  (cd $T && PYTHONPATH=$T timeout 3600 $PY perf/rfd4/cfg512/leg.py --spec $IN/pdl1-$SIZE.yaml --weights $W \
     --out $o --part dev --shared-box --batch $2 --lane-width 1 --batches $3 --max-attempts $3 \
     --max-foreign-cores 4 --quiet-timeout 300) > $o.log 2>&1
  local rc=$?; exec 8>&- 9>&-
  stamp "$1 rc=$rc"
}
for l in $LEGS; do case ${l:0:1} in A) leg $l 4 4 ;; B) leg $l 8 2 ;; C) leg $l 16 1 ;; esac; done
$BOOK rm $OWN > /dev/null
stamp ALLDONE
