#!/bin/bash
# Run one device command on CARD once the card has no open device fd and its fleet lock is ours, with in-run
# AICLK and load at 1 Hz beside it. Output dir D gets sha, dirty, card (start/end, rc), aiclk.tsv, run.log.
#   CARD=<n> D=<out dir> bash perf/bc4_block/when_free.sh python perf/bc4_block/mm_rate.py --out $D/mm.jsonl
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"; mkdir -p "$D"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-block TT_BIO_LEASE_TIMEOUT=7200
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 30; done
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 60 1800 "$@" > "$D/run.log" 2>&1
echo "rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
