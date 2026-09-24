#!/bin/bash
# One device arm: BindCraft 2 campaign, tt-bio Evoformer, qb1 card 3 (UMD 3 = PCI c1:00.0 =
# /dev/tenstorrent/0). Bucket 32, BindCraft 2 default. AICLK sampled from sysfs every 5 s
# DURING the run -- tt-smi hangs >2 min on this host (cardblock-qb1-1), sysfs does not.
set -u
SEED="$1"
WT=/home/ttuser/.coworker/wt/bcx-seeds
OUT="$WT/perf/bcx_seeds/runs/device_qb1_seed${SEED}"
CLK=/sys/class/tenstorrent/tenstorrent!0/tt_aiclk
mkdir -p "$OUT"
cd "$WT" || exit 1

export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-seeds
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

# AICLK + load sampler, 5 s, for the whole timed region.
( while :; do
    printf %s "{\"t\":$(date -u +%s),\"aiclk\":$(cat "$CLK" 2>/dev/null || echo null)"
    printf ",\"load\":%s}\n" "$(cut -d\  -f1 /proc/loadavg)"
    sleep 5
  done ) >>"$OUT/aiclk.jsonl" 2>/dev/null &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

date -u +"START %FT%TZ seed=$SEED sha=$(git rev-parse HEAD)" >>"$OUT/run.log"
/home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm device --trajectories 1 --seed "$SEED" --bucket 32 --out "$OUT" >>"$OUT/run.log" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null
date -u +"END %FT%TZ rc=$RC" >>"$OUT/run.log"
exit $RC
