#!/usr/bin/env bash
# One benchlock hold covering every device step this row owes, in dependency order. The box is
# carrying three other perf rows and each acquisition costs a queue wait, so re-acquiring per step
# would cost more wall clock than the steps themselves.
#
#   0  does the fused SDPA serve a bfp8 operand at all, off its own served/declined counters
#   1  is the narrowed destination a ROUNDING of the float64 answer, or something wrong
#   2  298 aa fold A/B  -- also the functional check: a fallback that throws shows up here
#   3  512 aa fold A/B  -- the cell, and the size the 298 aa control is known not to cover
#   4  which fast paths serve with the flag off vs on, off scripts/lever_census.py
#
# Step 1 runs before any fold because a fold A/B on a transform that is wrong rather than imprecise
# is wasted card time.
set -u
cd /home/ttuser/.coworker/wt/c14-bfp8-fastpath || exit 1
P=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:c14-bfp8-fastpath"
NOISE='DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http|^ --- '

# $? after a pipeline is the LAST stage's status, i.e. grep's, which is 1 whenever the filter
# matched nothing. Read PIPESTATUS[0] instead or every step reports a failure it did not have.
run() {                     # run <label> <tail-lines> -- <cmd...>
  local label="$1" n="$2"; shift 3
  echo "=== STEP $label $(date -uIs)"
  "$@" 2>&1 | grep -avE "$NOISE" | tail -"$n"
  echo "=== STEP $label rc=${PIPESTATUS[0]} $(date -uIs)"
}

run 0_fusedprobe 60 -- $E $P perf/c14_bfp8/fusedprobe.py \
  --out perf/c14_bfp8/fusedprobe_qb2c2.json

run 1_float64 60 -- $E $P perf/c14_bfp8/qkv_ref.py \
  --out perf/c14_bfp8/qkv_ref_qb2c2.json

run 2_fold298 45 -- $E $P perf/c14_bfp8/fold_ab.py --flag TT_BIO_TRIATT_B8 \
  --sizes 298 --blocks 2 --folds 2 --card 2 \
  --cifdir perf/c14_bfp8/cifs298 --out perf/c14_bfp8/fold_ab_298.json

run 3_fold512 45 -- $E $P perf/c14_bfp8/fold_ab.py --flag TT_BIO_TRIATT_B8 \
  --sizes 512 --blocks 2 --folds 2 --card 2 \
  --cifdir perf/c14_bfp8/cifs512 --out perf/c14_bfp8/fold_ab_512.json

# KERNEL-PATH, read off the levers' OWN served/declined counters rather than inferred from a time.
# A fast path that declines a bfp8 operand is a FAILURE of this region, not a cost, so the two
# censuses must agree lever for lever except where the region deliberately changes a dtype.
run 4_census_off 14 -- $E env -u TT_BIO_TRIATT_B8 $P scripts/lever_census.py --tt-bio $P \
  --label triatt_b8_off --out perf/c14_bfp8/census_298_off.json \
  -- -m tt_bio.main predict perf/size512/fixtures/cdk2x2_298.yaml \
  --sampling_steps 200 --recycling_steps 3 --seed 0 \
  --out_dir perf/c14_bfp8/census_out_off

run 4_census_on 14 -- $E env TT_BIO_TRIATT_B8=1 $P scripts/lever_census.py --tt-bio $P \
  --label triatt_b8_on --out perf/c14_bfp8/census_298_on.json \
  -- -m tt_bio.main predict perf/size512/fixtures/cdk2x2_298.yaml \
  --sampling_steps 200 --recycling_steps 3 --seed 0 \
  --out_dir perf/c14_bfp8/census_out_on

echo "=== CHAIN DONE $(date -uIs)"
