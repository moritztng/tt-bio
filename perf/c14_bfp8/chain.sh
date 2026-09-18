#!/usr/bin/env bash
# CLOCK-INSENSITIVE steps only. A release gate has been folding since 12:14Z and finishes ~15:15Z,
# and benchlock falls through its load wait by design, so anything wall-clock measured in this
# window is suspect: two sibling rows were already caught by it today (one A/A floor blew to
# +0.2396 s, an earlier session came back with an A/A CI of +/-0.3724 s, wider than the 0.251 s
# lever). What does NOT need a quiet host: served-vs-declined, a float64 comparison, a CIF digest
# and an Angstrom reading. Those run now; the timed A/B is a separate launch after the gate.
#
#   0  does the fused SDPA serve a bfp8 operand at all, off its own served/declined counters
#   1  is the narrowed destination a ROUNDING of the float64 answer, or something wrong
#   2  which fast paths serve with the flag off vs on, off scripts/lever_census.py
#   3  298 aa folds -- STRUCTURE only from this run; its seconds are recorded and marked suspect
set -u
cd /home/ttuser/.coworker/wt/c14-bfp8-fastpath || exit 1
P=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:c14-bfp8-fastpath"
NOISE='DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http|^ --- '

# $? after a pipeline is the LAST stage's status, i.e. grep's, which is 1 whenever the filter
# matched nothing. Read PIPESTATUS[0] or every step reports a failure it did not have.
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

# KERNEL-PATH, read off the levers' OWN served/declined counters rather than inferred from a time.
# A fast path that declines a bfp8 operand is a FAILURE of this region, not a cost, so the two
# censuses must agree lever for lever except where the region deliberately changes a dtype.
run 2_census_off 14 -- $E env -u TT_BIO_TRIATT_B8 $P scripts/lever_census.py --tt-bio $P \
  --label triatt_b8_off --out perf/c14_bfp8/census_298_off.json \
  -- -m tt_bio.main predict perf/size512/fixtures/cdk2x2_298.yaml \
  --sampling_steps 200 --recycling_steps 3 --seed 0 \
  --out_dir perf/c14_bfp8/census_out_off

run 2_census_on 14 -- $E env TT_BIO_TRIATT_B8=1 $P scripts/lever_census.py --tt-bio $P \
  --label triatt_b8_on --out perf/c14_bfp8/census_298_on.json \
  -- -m tt_bio.main predict perf/size512/fixtures/cdk2x2_298.yaml \
  --sampling_steps 200 --recycling_steps 3 --seed 0 \
  --out_dir perf/c14_bfp8/census_out_on

run 3_struct298 45 -- $E $P perf/c14_bfp8/fold_ab.py --flag TT_BIO_TRIATT_B8 \
  --sizes 298 --blocks 2 --folds 2 --card 2 \
  --cifdir perf/c14_bfp8/cifs298 --out perf/c14_bfp8/struct_298.json

echo "=== CHAIN DONE $(date -uIs)"
