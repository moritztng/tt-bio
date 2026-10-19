#!/bin/bash
# bc4-bet-fuse in-process A/B: one bc2_evo_devprof process alternates a call-time flag per rep in ABBA
# order, so host load and clock drift fall on both arms alike (qb2 runs at load 12-20 with BC4 and SPD
# rows on the other cards). AICLK sampled at 1 Hz beside it.
#   CARD=<umd index> [FLAG=tt_bio.af2._AF2_T2CM[,tt_bio.tenstorrent._TRIMUL_GIN_BIASED] TAG=t2cm REPS=24] bash perf/bc4_fuse/alt.sh   (worktree root, detached)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/${TAG:-t2cm}alt_$(git rev-parse --short HEAD); mkdir -p "$D"
PY=${PY:-$HOME/bcx_e2e_venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz}
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fuse TT_BIO_LEASE_TIMEOUT=3600
git rev-parse HEAD > "$D/sha"; git status --porcelain --untracked-files=no > "$D/dirty"
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$CARD/tt_aiclk)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk.tsv" 9>&- & P=$!
echo "start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
timeout -s INT -k 120 6000 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" --out "$D/alt" \
    --warm 4 --reps "${REPS:-24}" --alternate "${FLAG:-tt_bio.af2._AF2_T2CM}" > "$D/alt.log" 2>&1
echo "rc=$? end $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> "$D/card"
kill $P; touch "$D/done"
