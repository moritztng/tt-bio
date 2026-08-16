#!/usr/bin/env bash
# Isolate the affinity diffusion cost on the AFTER tree: same fold, 1 affinity
# sample instead of 5. (wall5 - wall1) / 4 = per-sample affinity diffusion.
set -u
W=/home/ttuser/.coworker/wt/boltz2-affinity-device-only
O=$W/perf/boltz2-affinity-device-only
PY=/home/ttuser/tt-bio-dev/env/bin/python3
until grep -q RESEED_EXIT $O/reseed.log 2>/dev/null; do sleep 15; done
run() {
  cd $W; export PYTHONPATH=$W TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-device-only
  for r in 1 2 3; do
    D=$(mktemp -d /tmp/aff1_XXXX); S=$(date +%s.%N)
    $PY -m tt_bio.main predict $W/examples/affinity_fkg.yaml --out_dir "$D" \
      --model boltz2 --single_sequence --recycling_steps 3 --sampling_steps 200 \
      --diffusion_samples 1 --seed 0 --affinity_mw_correction \
      --sampling_steps_affinity 200 --diffusion_samples_affinity 1 > "$D/run.log" 2>&1
    RC=$?; E=$(date +%s.%N)
    echo "after affinity_1sample rep$r rc=$RC wall=$(echo "$E-$S"|bc)" >> $O/wall_after_1sample.txt
    rm -rf "$D"
  done
}
bash /home/ttuser/.coworker/scripts/benchlock.sh boltz2-affinity-device-only -- bash -c "$(declare -f run); run" >> $O/aff1.log 2>&1
echo "AFF1_EXIT=$? $(date -u +%FT%TZ)" >> $O/wall_after_1sample.txt
