#!/usr/bin/env bash
# Lever A accuracy gate: a full 200-step trajectory with RFD3_FOLD_PROCESS_Z on and off, same seed,
# same fixture. The fold is algebraically exact but re-rounds, so the bar is a trajectory
# comparison, not a sha: PCC >= 0.99 and an all-atom Kabsch RMSD small against the 6.525 A that got
# RFD3_FAST_GRID rejected and the 25.305 A of a seed change.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/tt-bio
STEPS=${STEPS:-200}
for arm in off on; do
  if [ "$arm" = on ]; then F=1; else F=0; fi
  rm -rf "perf/p42/fold_$arm"
  S=$(date +%s)
  env RFD3_FOLD_PROCESS_Z=$F TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH="$PWD" \
      "$PY" design --model rfd3 perf/dsfix/fixtures/rfd3_R4.json \
      --out_dir "perf/p42/fold_$arm" --num_designs 2 --batch_size 2 \
      --num_timesteps "$STEPS" --seed 7 --from_pdb --device_ids 0 \
      > "perf/p42/fold_$arm.log" 2>&1
  echo "arm=$arm rc=$? steps=$STEPS wall=$(( $(date +%s) - S ))s"
done
echo "=== ACCURACY GATE ==="
/home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/rfd3_cif_rmsd.py \
  perf/p42/fold_off/R4_b100_0.cif perf/p42/fold_on/R4_b100_0.cif \
  perf/p42/fold_off/R4_b100_1.cif perf/p42/fold_on/R4_b100_1.cif
