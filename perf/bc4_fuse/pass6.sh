#!/bin/bash
# bc4-bet-fuse pass 6 chain: gin-alone census at the fixed back move, then the T2CM + GIN stack as an
# in-process ABBA against off. CARD=<umd index>, detached, worktree root.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/pass6_$(git rev-parse --short HEAD); mkdir -p "$D"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fuse TT_BIO_LEASE_TIMEOUT=7200
PY=$HOME/bcx_e2e_venv/bin/python; PARAMS=$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz
echo "gin start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
TT_BIO_TRIMUL_GIN_BIASED=1 timeout -s INT -k 60 7800 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" \
  --out "$D/gin_cen" --warm 1 --reps 2 --census > "$D/gin_cen.log" 2>&1
echo "gin rc=$? end $(date +%s)" >> "$D/card"
CARD=$CARD TAG=stack REPS=48 FLAG=tt_bio.af2._AF2_T2CM,tt_bio.tenstorrent._TRIMUL_GIN_BIASED bash perf/bc4_fuse/alt.sh
echo "alt end $(date +%s)" >> "$D/card"; touch "$D/done"
