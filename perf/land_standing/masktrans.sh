#!/bin/bash
# TT_BIO_MASK_TRANS: the inference fold A/B its own comment says it needs before it can ship.
#
# The defect is about PADDED rows -- upstream ends `_transition` with `x = linear_out(x) * mask`
# at seven hard-coded call sites and our port passes no mask, so on a padded row the MLP passes
# its own biases into the residual and that compounds over 48 blocks. A fixture at exactly 832
# tokens has no pad rows at all and would read the lever as inert, so this folds **800 residues**,
# which OpenFold3 pads to 832: 32 real pad rows.
#
# Three arms, and the middle one is the instrument check the lever's own comment demands: an
# ALL-ONES mask is the one input that must leave every number bit-identical to the unmasked arm.
# If `ones` is not byte-identical to `off`, the lever is not the mask and nothing here means what
# it says.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/masktrans
IN=$WT/perf/land_standing/out/deepfix/cdk2_deep_800.yaml
MSA=$WT/perf/land_standing/out/deepfix/msa
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

run () {
  tag=$1
  on=$2
  ones=$3
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      TT_BIO_MASK_TRANS="$on" TT_BIO_MASK_TRANS_ONES="$ones" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --msa_dir "$MSA" --msa_cache_only --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  cif=$(sha256sum "$OUT/res_$tag"/openfold3_results_cdk2_deep_800/structures/*.cif 2>/dev/null | cut -c1-16)
  echo "LEG $tag mask=$on ones=$ones rc=$rc secs=$((t1 - t0)) cif=$cif"
  grep -o '"plddt": [0-9.]*\|"ptm": [0-9.]*' \
    "$OUT/res_$tag/openfold3_results_cdk2_deep_800/results.json" 2>/dev/null | tr '\n' ' '
  echo
}

run off   0 0
run ones  1 1
run on    1 0
run off2  0 0
echo MASK_DONE
