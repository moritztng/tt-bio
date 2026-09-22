#!/usr/bin/env bash
# of3t-f64route: the inference A/B for one model, with its A/A floor.
#   runinf.sh <model> <card> [fixture]
# base = HEAD~1 (this branch WITHOUT the route commit), so `base == off` is a statement about
# exactly one commit and nothing else. `on` forces TT_BIO_HOST_F64_SOFTMAX_AB=all, which selects
# every construction site in the process -- the strongest form of the claim that an inference
# fold cannot reach the host float64 softmax however the variable is set.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-f64route
O=/tmp/of3t/of3t-f64route
cd "$W"
M=$1; CARD=$2; FIX=${3:-perf/size512/fixtures/cdk2x2_128.yaml}
export TT_VISIBLE_DEVICES=$CARD
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-f64route
/home/ttuser/tt-bio-dev/env/bin/python3 perf/of3t_d137tapegate/inference_ab_with_aa_floor.py \
  --model "$M" --fixture "$W/$FIX" \
  --base-tree "$O/base" --tree "$W" \
  --python /home/ttuser/tt-bio-dev/env/bin/python3 \
  --workdir "$O/inf_$M" --card "$CARD" --reps 2 \
  --tree-must-have host_softmax_or_none --base-must-lack host_softmax_or_none \
  --lever-env TT_BIO_HOST_F64_SOFTMAX_AB --lever-value all \
  --out "$W/perf/of3t_f64route/INFERENCE_AB_${M}.json"
