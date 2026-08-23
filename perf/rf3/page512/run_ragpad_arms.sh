#!/usr/bin/env bash
# The missing perf cell: RF3's 512 aa page fold with TT_BIO_SDPA_RAGGED_PAD on.
#
# Protocol, taken from run_bisect_arms.sh which learned it the hard way:
#   * interleaved, not all-of-one-then-the-other (all-A-then-all-B read +13.3% on this box where
#     interleaved read +5.2%),
#   * two processes per arm, `--repeat 2` each, so the pair is the A/A control and the pooled
#     sample is the same four warm folds a single `--repeat 4` would have given,
#   * every timed run inside benchlock, because a co-tenanted timing is a wrong number.
# a1 pad=0 is measured in-session on purpose: the published 49.411 s was taken at main 25d0e40e
# and the tree has moved (perf-page-cell-is-historical-not-live-baseline).
# TT_BIO_SDPA_RAGGED_CENSUS counts ragged-vs-aligned fused-SDPA calls per site; at 512 tokens the
# fix should fire on nothing, and the counter is what turns "should" into a reading.
set -u
WT=/home/ttuser/.coworker/wt/rf3-4x-with-accuracy-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE="TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:rf3-4x-with-accuracy-land"
export BENCHLOCK_LOAD_WAIT_S=2400 BENCHLOCK_WAIT_S=3600
OUT=$WT/perf/rf3/page512
CEN=$WT/perf/rf3/page512/census
mkdir -p "$CEN"
cd "$WT" || exit 1

run() {  # run <tag> <arm> <pad>
  tag=$1; arm=$2; pad=$3
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-4x-with-accuracy-land -- \
    env PYTHONPATH="$PP" $LEASE TT_BIO_SDPA_RAGGED_PAD=$pad \
        TT_BIO_SDPA_RAGGED_CENSUS="$CEN/$tag" \
    "$PY" perf/rf3/page512_tt.py --repeat 2 --arm "$arm" --label "$tag" \
      --out "$OUT/rp_${tag}_qb2c1.json" > "$OUT/rp_${tag}.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc $(date -u +%H:%M:%S) ==="
  grep -E "median|warm_walls|digest" "$OUT/rp_${tag}.log" | tail -4
  echo "$tag rc=$rc" >> "$OUT/ragpad_chain.status"
}

run a1off_p1 a1 0
run a1on_p1  a1 1
run a1off_p2 a1 0
run a1on_p2  a1 1
run a2off_p1 a2 0
run a2on_p1  a2 1
echo RAGPAD_ARMS_DONE >> $OUT/ragpad_chain.status
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
