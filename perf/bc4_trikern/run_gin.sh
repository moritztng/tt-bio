#!/bin/bash
# Detached runner for gin224.py: waits for the card to have no open fd and its fleet lock, samples AICLK and
# load at 1 Hz beside the probe.   CARD=0 bash perf/bc4_trikern/run_gin.sh
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
CARD=${CARD:-0}
D=$WT/_bc4out/gin224_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-trikern
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; echo "$CARD" > "$D/cardno"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 60 1500 "$PY" perf/bc4_trikern/gin224.py --out "$D/gin.jsonl" > "$D/run.log" 2>&1
echo "rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
