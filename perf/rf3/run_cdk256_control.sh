#!/usr/bin/env bash
# The aligned-length accuracy control the brief asks for, built at cdk2_256 on pc card 0.
#
# cdk2_128 was that control and cannot serve as one: the persistent-mask kernel refuses every call
# at that fixture on `memory_config`, so the arm is dark there and a5's coordinates come back
# byte-identical to a1's (state doc 15b). An aligned control has to be a rung where the arm runs.
# 256 mod 32 == 0, so the ragged pad fires on nothing and the rung isolates the compute config.
#
# The two halves run in parallel because accuracy_cell supports it: --ref-only needs no card and
# --dev-only needs no reference, and they are joined afterwards under the same draw-hash assertion.
# Every arm gets its own a1 on this card and a1 runs twice, because pc card 0 miscomputes some
# matmuls at a low rate (`pc-card0-512aa-fold-nondeterminism`) and a single reading would carry the
# card's fault as if it were the arm's.
set -u
WT=/home/moritz/.coworker/wt/rf3-fused-hifi-precision-arm
PY=/home/moritz/tt-bio/env/bin/python3
PP=$WT:/home/moritz/rf3_perf_deps
R=$WT/perf/rf3/results
L=$R/logs
cd "$WT"
LEASE="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rf3-fused-hifi-precision-arm"

# 1. the reference half, no card, in the background. 16.5 s/seed at 117 aa on this CPU, so a few
#    minutes at 256.
env PYTHONPATH=$PP OMP_NUM_THREADS=10 timeout 5400 "$PY" -u scripts/rf3_port/accuracy_cell.py \
    --fixture cdk2_256 --seeds 0,1,2,3,4 --ref-only \
    --work "$R/accuracy_cdk2_256" --out "$R/accuracy_cdk2_256_ref.json" \
    > "$L/cdk256_ref.log" 2>&1 &
REFPID=$!
echo "ref half pid $REFPID $(date -Is)"

# 2. the device halves on the card. --dev-only writes `pending` and its route counters when the
#    reference is not there yet, which is also the answer to "does the arm fire at 256 at all".
dev() { # dev <arm> <tag>
  local arm=$1 tag=$2
  [ -s "$R/pc_${tag}.json" ] && { echo "skip $tag (have it)"; return 0; }
  echo "=== $(date -Is) START $tag"
  env PYTHONPATH=$PP $LEASE timeout 3000 "$PY" -u scripts/rf3_port/accuracy_cell.py \
      --fixture cdk2_256 --arm "$arm" --seeds 0,1,2,3,4 --dev-only \
      --ref-cache "$R/accuracy_cdk2_256" --work "$R/pc_${tag}" --out "$R/pc_${tag}.json" \
      > "$L/pc_${tag}.log" 2>&1
  echo "=== $(date -Is) END $tag rc=$?"
}
dev a1 cdk256_a1_p1
dev a5 cdk256_a5
dev a1 cdk256_a1_p2

wait $REFPID; echo "ref half rc=$? $(date -Is)"

# 3. join: the device halves cached their coordinates before the reference existed, so score them
#    now against it. --rescore reads the cache, opens no card and loads no model.
for tag in cdk256_a1_p1 cdk256_a5 cdk256_a1_p2; do
  env PYTHONPATH=$PP "$PY" -u scripts/rf3_port/accuracy_cell.py \
      --fixture cdk2_256 --seeds 0,1,2,3,4 --rescore \
      --work "$R/pc_${tag}" --ref-cache "$R/accuracy_cdk2_256" \
      --out "$R/pc_${tag}_scored.json" > "$L/pc_${tag}_rescore.log" 2>&1
  echo "rescore $tag rc=$?"
done
echo "CDK256 CONTROL DONE $(date -Is)"
