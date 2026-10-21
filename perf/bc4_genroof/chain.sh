#!/bin/bash
# bc4-bet-genroof per-call probe on the first free qb2 chip: CARDS="3 2" bash perf/bc4_genroof/chain.sh (detached)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/probe_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
CARD=
while [ -z "$CARD" ]; do
  for c in ${CARDS:-3 2}; do
    fuser /dev/tenstorrent/$c >/dev/null 2>&1 && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"; flock -n 9 || { exec 9>&-; continue; }
    fuser /dev/tenstorrent/$c >/dev/null 2>&1 && { exec 9>&-; continue; }
    CARD=$c; break
  done
  [ -n "$CARD" ] || sleep 20
done
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-genroof
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; echo "$CARD" > "$D/cardno"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "probe start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 60 1500 "$PY" perf/bc4_genroof/gin_probe.py --out "$D/probe.jsonl" > "$D/probe.log" 2>&1
echo "probe rc=$? end $(date +%s)" >> "$D/card"
kill $P; touch "$D/done"
