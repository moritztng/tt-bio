#!/usr/bin/env bash
# Plan step 4: do the two landed, opt-in, bit-exact levers compose at the pinned rfd3_R4 fixture?
#
#   A   both off                                  -- what a user gets today
#   S   RFD3_SPARSE_BIAS=1                        -- the atom-side fused sparse-bias kernel
#   ST  RFD3_SPARSE_BIAS=1 RFD3_TUNE_MATMUL=1     -- with the token-side calibrator on top
#
# TUNE_MATMUL alone is already measured at 1.130x (perf/p42/ab_tune/), so it is not re-run; S and ST
# against A in the same hold give the atom-side term and the composition. Arms interleaved, 2 reps.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/p42/ab_compose
mkdir -p "$OUT"
for rep in 1 2; do
  for arm in A S ST; do
    case $arm in
      A)  SB=0; TM=0 ;;
      S)  SB=1; TM=0 ;;
      ST) SB=1; TM=1 ;;
    esac
    echo "=== arm=$arm rep=$rep sparse=$SB tune=$TM $(date -Is) ==="
    env RFD3_SPARSE_BIAS=$SB RFD3_TUNE_MATMUL=$TM TT_VISIBLE_DEVICES=0 \
        TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH=$PWD "$PY" \
        scripts/rfd3_port/p42_drain_attribution.py \
        --num_timesteps 30 --designs 2 --out "$OUT/${arm}${rep}.json" 2>&1 \
      | grep -E "^\[drain\]|^\[done\]|Error|Traceback"
  done
done
echo "=== ALL DONE $(date -Is) ==="
