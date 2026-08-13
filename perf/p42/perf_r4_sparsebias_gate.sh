#!/usr/bin/env bash
# The fused sparse-bias kernel at the production batch: bit-exactness gate.
#
# Unlike lever A this kernel is claimed BIT-EXACT -- it writes the same values in the same
# positions as the template/scatter/typecast chain it replaces, gated on torch.equal at batch 1
# (scripts/rfd3_port/p36_bias_kernel_probe.py). So the bar here is a byte-identical CIF, not a
# trajectory PCC. The RMSD line is only a diagnostic for the case where the sha does not match.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/tt-bio
STEPS=${STEPS:-200}
for arm in off on; do
  if [ "$arm" = on ]; then F=1; else F=0; fi
  rm -rf "perf/p42/sb_$arm"
  S=$(date +%s)
  env RFD3_SPARSE_BIAS=$F TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH="$PWD" \
      "$PY" design --model rfd3 perf/dsfix/fixtures/rfd3_R4.json \
      --out_dir "perf/p42/sb_$arm" --num_designs 2 --batch_size 2 \
      --num_timesteps "$STEPS" --seed 7 --from_pdb --device_ids 0 \
      > "perf/p42/sb_$arm.log" 2>&1
  echo "arm=$arm rc=$? steps=$STEPS wall=$(( $(date +%s) - S ))s"
done
echo "=== SHA ==="
sha256sum perf/p42/sb_off/*.cif perf/p42/sb_on/*.cif
if cmp -s perf/p42/sb_off/R4_b100_0.cif perf/p42/sb_on/R4_b100_0.cif && \
   cmp -s perf/p42/sb_off/R4_b100_1.cif perf/p42/sb_on/R4_b100_1.cif; then
  echo "VERDICT: BYTE-IDENTICAL"
else
  echo "VERDICT: DIFFER -- diagnostic below"
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/rfd3_cif_rmsd.py \
    perf/p42/sb_off/R4_b100_0.cif perf/p42/sb_on/R4_b100_0.cif \
    perf/p42/sb_off/R4_b100_1.cif perf/p42/sb_on/R4_b100_1.cif
fi
