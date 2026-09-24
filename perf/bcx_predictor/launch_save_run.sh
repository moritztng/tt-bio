#!/bin/bash
# One device trajectory with save_failed_trajectories on, so the rejection leaves a
# sequence behind. The three earlier device trajectories kept only their losses CSV, which
# is why the sharp metric test -- refold the device's OWN binder with BindCraft 2's
# predictor and compare confidence on the same sequence -- was impossible on them.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
until ! pgrep -f "run_arm.py --arm device" >/dev/null; do sleep 30; done
OUT=$WT/perf/bcx_predictor/runs/device_save_seed0
mkdir -p "$OUT"
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-predictor
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
/home/ttuser/bcx_e2e_venv/bin/python "$WT/perf/bcx_predictor/run_arm.py" \
  --arm device --trajectories 1 --seed 0 --bucket 0 --out "$OUT" >>"$OUT/run.log" 2>&1
echo "save-run rc=$? $(date -u +%FT%TZ)" >> "$WT/perf/bcx_predictor/runs/device_progress.log"
