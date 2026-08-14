#!/usr/bin/env bash
# Re-verify the two levers on CURRENT MAIN, at the pinned rfd3_R4 fixture.
# Bit-exactness is the bar: both arms must write the shipped shas 87af014c.../4aad0140...
set -u
WT=/home/ttuser/.coworker/wt/rfd3-optimize-fixture-land
cd "$WT" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/tt-bio
for arm in off on; do
  if [ "$arm" = on ]; then SB=1; TM=1; else SB=0; TM=0; fi
  rm -rf "perf/land/${arm}"
  S=$(date +%s)
  env RFD3_SPARSE_BIAS=$SB RFD3_TUNE_MATMUL=$TM TT_VISIBLE_DEVICES=3 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-fixture-land PYTHONPATH="$WT" \
      "$PY" design --model rfd3 perf/dsfix/fixtures/rfd3_R4.json \
      --out_dir "perf/land/${arm}" --num_designs 2 --batch_size 2 \
      --num_timesteps 200 --seed 7 --from_pdb --device_ids 3 \
      > "perf/land/${arm}.log" 2>&1
  echo "arm=$arm rc=$? wall=$(( $(date +%s) - S ))s"
done
echo "=== SHA ==="
sha256sum perf/land/*/*.cif
