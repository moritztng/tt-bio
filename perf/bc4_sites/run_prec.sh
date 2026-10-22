#!/bin/bash
# Detached qb2 runner for splitk_prec.py, after run_probe.sh's done marker: first free chip of CARDS with its lock.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/probe; PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
until [ -e "$D/done" ]; do sleep 30; done
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
echo "prec start $(date +%s) card $CARD aiclk $(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" >> "$D/card"
timeout -s INT -k 60 600 "$PY" perf/bc4_sites/splitk_prec.py --out "$D/prec.jsonl" > "$D/prec.log" 2>&1
echo "prec rc=$? end $(date +%s)" >> "$D/card"; touch "$D/prec.done"
