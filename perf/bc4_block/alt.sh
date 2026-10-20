#!/bin/bash
# In-process A/B of call-time module flags on the BindCraft 2 iteration body: one bc2_evo_devprof process
# alternates the flags per rep in ABBA order, so host load and clock drift fall on both arms alike. Waits for the
# card to have no open device fd and for its fleet lock. AICLK and load sampled at 1 Hz beside it.
#   CARD=<n> FLAG=tt_bio.reblock_permute._PFPK[,...] TAG=pk REPS=24 bash perf/bc4_block/alt.sh   (detached)
# Read: python3 perf/bc4_block/alt_read.py <out dir>
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/${TAG:-pk}alt_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-block TT_BIO_LEASE_TIMEOUT=7200
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$CARD.lock"; flock 9
while fuser /dev/tenstorrent/$CARD >/dev/null 2>&1; do sleep 30; done
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d" " -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 120 9000 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/alt" \
    --warm 4 --reps "${REPS:-24}" --alternate "${FLAG:-tt_bio.reblock_permute._PFPK}" > "$D/alt.log" 2>&1
echo "rc=$? end $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
