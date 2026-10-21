#!/bin/bash
# Detached qb2 runner for probe.py: takes the first of CARDS with no open fd and its fleet lock, samples AICLK and
# load at 1 Hz beside the probe.   CARDS="0 1 2" bash perf/bc4_sites/run_probe.sh
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/probe; mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
while :; do
  for c in ${CARDS:-0 1 2}; do
    fuser /dev/tenstorrent/$c >/dev/null 2>&1 && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"; flock -n 9 || continue
    fuser /dev/tenstorrent/$c >/dev/null 2>&1 && { exec 9>&-; continue; }
    CARD=$c; break 2
  done
  sleep 30
done
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-sites
cat sha > "$D/sha" 2>/dev/null; echo "$CARD" > "$D/cardno"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "start $(date +%s) card $CARD load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 60 1200 "$PY" perf/bc4_sites/probe.py --out "$D/probe.jsonl" > "$D/run.log" 2>&1
echo "rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
