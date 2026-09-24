#!/bin/bash
# The padding control the orchestrator asked for: one device trajectory at length_bucket_size
# 1 on a seed both other arms have run. post_seed100 ran the same seed at BindCraft 2's
# default 32 and reference_qb1_seed100 runs it at 1, so a paired margin between them today
# mixes the device against the padding. At bucket 1 the complex is whatever BindCraft 2 drew
# and its predict path pads nothing; the splice still pads the token axis to 32 with its own
# mask, which is the part `mutate_predict.json` graded clean at 25 pad tokens.
#
# Same program post_seed100 ran on. The commits since are the is_allocated() guard and three
# analysis scripts, none of them on the loop's path.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
cd "$WT" || exit 1
OUT=$WT/perf/bcx_predictor/runs/b1_seed100_try2
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-predictor
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
/home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/aiclk_sample.py" "$OUT/aiclk.jsonl" 5 &
CLK=$!
/home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm device --trajectories 1 --seed 100 --bucket 1 --out "$OUT" >>"$OUT/run.log" 2>&1
RC=$?
kill "$CLK" 2>/dev/null
echo "b1_seed100 rc=$RC $(date -u +%FT%TZ)" >> "$WT/perf/bcx_predictor/runs/device_progress.log"
