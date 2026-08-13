#!/usr/bin/env bash
# Whole-invocation wall for the two levers, both arms in ONE benchlock hold.
#
# An unlocked both-on run came out 413s against an unlocked both-off 391s, which contradicts the
# -16.43 % the interleaved ms/step A/B measured under a lock. Either the box was busy or there is a
# per-process cost the median warm step cannot see (the calibrator's ~28.6s, plus whatever the fused
# kernel's JIT build costs). This settles it: same hold, alternating, two reps each.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/tt-bio
for rep in 1 2; do
  for arm in off on; do
    if [ "$arm" = on ]; then SB=1; TM=1; else SB=0; TM=0; fi
    rm -rf "perf/p42/wall_${arm}${rep}"
    S=$(date +%s)
    env RFD3_SPARSE_BIAS=$SB RFD3_TUNE_MATMUL=$TM TT_VISIBLE_DEVICES=0 \
        TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH="$PWD" \
        "$PY" design --model rfd3 perf/dsfix/fixtures/rfd3_R4.json \
        --out_dir "perf/p42/wall_${arm}${rep}" --num_designs 2 --batch_size 2 \
        --num_timesteps 200 --seed 7 --from_pdb --device_ids 0 \
        > "perf/p42/wall_${arm}${rep}.log" 2>&1
    echo "arm=$arm rep=$rep rc=$? wall=$(( $(date +%s) - S ))s"
  done
done
echo "=== SHA ==="
sha256sum perf/p42/wall_*/*.cif
