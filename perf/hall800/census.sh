#!/usr/bin/env bash
# Which shipped perf levers actually fire at 800 tokens, on the production config.
#
#   perf/hall800/census.sh <label> <fixture.yaml>
#
# scripts/lever_census.py is the same instrument docs/size_ladder_baseline.d was recorded
# with, so the rows are directly comparable to the committed 512/640/768 columns. The fold
# is the served config, not the ladder's cheap one, because a gate can resolve differently
# under --fast and at 1 sample than it does single-sequence at 6 steps.
set -uo pipefail
cd "$(dirname "$0")/../.."
LABEL=${1:?label}; FIX=${2:?fixture}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/hall800/runs/census_$LABEL
rm -rf "$D"; mkdir -p "$D"
python3 perf/hall800/clk.py --node 0 --target 1350 --out "$D/clock_n0.jsonl" > "$D/clk.log" 2>&1 &
CLK=$!
trap 'kill -TERM $CLK 2>/dev/null' EXIT
sleep 2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:hall-capacity-800aa
export PYTHONPATH="$PWD"
timeout 2400 $PY scripts/lever_census.py --tt-bio $PY --label "$LABEL" \
    --out "$D/census.json" -- \
    -m tt_bio.main predict "$FIX" \
    --model protenix-v2 --fast --diffusion_samples 1 --max_parallel_samples 1 \
    --seed 42 --msa_dir perf/hall800/msa --msa_cache_only \
    --devices 0 --out_dir "$D/out" --override \
    > "$D/census.log" 2>&1
rc=$?
kill -TERM $CLK 2>/dev/null
echo "census rc=$rc -> $D/census.json"
exit $rc
