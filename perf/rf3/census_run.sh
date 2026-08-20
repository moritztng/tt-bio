#!/bin/bash
# Lever census on RF3, one size per invocation. The census runs the CLI it is given as a
# subprocess with its counter hook first on PYTHONPATH, so pointing --tt-bio at the env
# python and the CLI at the bench harness censuses this port without a code change.
set -u
WT=${WT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
AA=$1
OUT=${OUT:-$WT/perf/rf3/results/census_rf3_${AA}.json}
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:rf3-perf}
# The pass-4 recommended RF3 configuration: every lever this campaign landed, on.
export TT_BIO_TRIATT_FUSED_HIFI=1
export TT_BIO_RF3_GLN_ROW_FOLD=1
export TT_BIO_OPM_SMALL_DEPTH=1
"$PY" scripts/lever_census.py \
  --tt-bio "$PY" \
  --pythonpath "$WT:/home/ttuser/rf3_perf_deps" \
  --label "rf3-$AA" --out "$OUT" \
  -- perf/rf3/tt_rf3_bench.py --aa "$AA" --n_recycles 1 --num_steps 3 --reps 1 \
     --tag census --out /dev/null
