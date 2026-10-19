#!/bin/bash
# bc4-bet-fuse pass 7 chain: the three-chain stack (T2CM + GIN + triangle-attention tail) census with the
# tail's in-run float64 check on its first calls, then the stack as an in-process ABBA against off.
# CARD=<umd index>, detached, worktree root, under flock ~/spd_qb2_chip$CARD.lock (the SPD feed reads it as busy).
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT"
D=$WT/_bc4out/pass7_$(git rev-parse --short HEAD); mkdir -p "$D"
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bc4-bet-fuse TT_BIO_LEASE_TIMEOUT=7200
PY=$HOME/bcx_e2e_venv/bin/python; PARAMS=$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz
echo "cen start $(date +%s) load $(cut -d" " -f1 /proc/loadavg)" >> "$D/card"
TT_BIO_AF2_T2CM=1 TT_BIO_TRIMUL_GIN_BIASED=1 TT_BIO_TRIATT_TAIL_UNRECORDED=1 TT_BIO_TRIATT_TAIL_BIAS_BH=1 \
  TT_BIO_TRIATT_TAIL_CHECK=16 TT_BIO_TRIATT_TAIL_CHECK_LOG=$D/tail_check.jsonl \
  timeout -s INT -k 60 7800 "$PY" perf/spd/bc2_evo_devprof.py --params "$PARAMS" \
  --out "$D/stack3_cen" --warm 1 --reps 2 --census > "$D/stack3_cen.log" 2>&1
echo "cen rc=$? end $(date +%s)" >> "$D/card"
CARD=$CARD TAG=stack3 REPS=48 FLAG=tt_bio.af2._AF2_T2CM,tt_bio.tenstorrent._TRIMUL_GIN_BIASED,tt_bio.triatt_qkv.TAIL_UNRECORDED,tt_bio.triatt_qkv.TAIL_BIAS_BH \
  bash perf/bc4_fuse/alt.sh
echo "alt end $(date +%s)" >> "$D/card"; touch "$D/done"
