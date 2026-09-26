#!/bin/bash
# The model's own confidence at 832, off vs on, with a control that fixes the floor.
#
# The 832 CA-RMSD question failed for want of resolution because the fixture's seed floor is
# 19.6 A. pLDDT does not have that problem here: at 704 two shipped-default runs at the same seed
# came back bit-identical end to end (0.357573 twice), so at fixed seed this pipeline has no
# noise at all and any difference between the arms is the lever. Leg 3 re-establishes that at 832
# rather than assuming it transfers from 704.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/plddt832
IN=$OUT/inputs/aa832/cdk2apo_832.yaml
CARD=3
cd "$WT" || exit 1

run_leg () {
  tag=$1
  dk=$2
  t0=$(date +%s.%N)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      TT_BIO_TRIATT_DIVIDING_K="$dk" \
      /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict --model openfold3 \
        --single_sequence --sampling_steps 20 --diffusion_samples 1 \
        --out_dir "$OUT/res_$tag" "$IN" > "$OUT/log_$tag.txt" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "LEG $tag dividing_k=$dk rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  grep -o '"plddt": [0-9.]*' "$OUT/res_$tag/openfold3_results_cdk2apo_832/results.json" 2>/dev/null
}

run_leg off1 0
run_leg on   1
run_leg off2 0
echo ALL_LEGS_DONE
