#!/usr/bin/env bash
# Lever B end-to-end parity + whole-invocation wall clock on the pinned rfd3_R4 fixture.
#
# The per-op argument for RFD3_TUNE_MATMUL=1 is the calibrator's own bitwise gate. This is the
# fold-level check: the same seed, the same fixture, two arms, and the CIF sha256 must match.
# It also gives the number the median warm step cannot see -- the whole wall clock, which is what
# decides whether the one-time calibration pays for itself on a single-batch invocation.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/tt-bio
STEPS=${STEPS:-40}
for arm in A B; do
  if [ "$arm" = B ]; then T=1; else T=0; fi
  rm -rf "perf/p42/cif_$arm"
  S=$(date +%s)
  env RFD3_TUNE_MATMUL=$T TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH="$PWD" \
      "$PY" design --model rfd3 perf/dsfix/fixtures/rfd3_R4.json \
      --out_dir "perf/p42/cif_$arm" --num_designs 2 --batch_size 2 \
      --num_timesteps "$STEPS" --seed 7 --from_pdb --device_ids 0 \
      > "perf/p42/cif_$arm.log" 2>&1
  echo "arm=$arm rc=$? steps=$STEPS wall=$(( $(date +%s) - S ))s"
  sha256sum "perf/p42/cif_$arm"/*.cif 2>/dev/null || echo "  (no CIF written -- see perf/p42/cif_$arm.log)"
done
echo "=== CIF PARITY ==="
if diff -q <(cd perf/p42/cif_A && sha256sum *.cif | sed 's#cif_A/##') \
           <(cd perf/p42/cif_B && sha256sum *.cif | sed 's#cif_B/##') >/dev/null 2>&1; then
  echo "BYTE-IDENTICAL"
else
  echo "DIFFER"
fi
