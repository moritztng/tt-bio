#!/bin/bash
# bc4-bet-trikern chain on one qb2 chip: gin_sweep (per call), then the BC2 ABBA of GIN_WHOLE.
#   CARD=<n> bash perf/bc4_trikern/chain.sh   (detached)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
CARD=${CARD:-3}
D=$WT/_bc4out/chain_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-trikern
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 20; done
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"; echo "$CARD" > "$D/cardno"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "sweep start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 60 1200 "$PY" perf/bc4_trikern/gin_sweep.py --out "$D/sweep.jsonl" > "$D/sweep.log" 2>&1
echo "sweep rc=$? $(date +%s)" >> "$D/card"
timeout -s INT -k 120 5400 "$PY" perf/spd/bc2_evo_devprof.py \
    --params "${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}" --out "$D/alt" \
    --warm 4 --reps "${REPS:-24}" --alternate tt_bio.trimul_tail.GIN_WHOLE > "$D/alt.log" 2>&1
echo "alt rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
