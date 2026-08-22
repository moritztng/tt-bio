#!/usr/bin/env bash
# 768/1024 aa non-regression for the accurate-softmax scope fix, on the instrument 2399c8f8's
# 1024 aa win was measured with.
#
# NOT under benchlock, deliberately, and the run records why. Another worker held a sustained
# multi-model perf campaign on cards 0/1/3 of this box for the whole window this could run in, so
# an exclusive timed measurement was not available. Taking the lock anyway would have written a
# claim of exclusivity that was false. Instead:
#
#   * trunk_decompose times host wall-clock between device syncs, so absolute seconds here are
#     INFLATED by co-tenancy and are not publishable. The arms run back-to-back at the same
#     interleave, so arm-to-arm comparison survives; loadavg is recorded per run to price it.
#   * The L1 rung and fp32_softmax latch counters this checks are load-immune: which rung the
#     allocator picks is decided by tensor shape and L1 capacity, not by host load. That check is
#     the one 2399c8f8's win actually rests on, and it is exact here.
#
# Interleaved fix/pre per rung, so a load drift between rungs cannot masquerade as an arm effect.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-regression-512aa-bisect
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE_ENV="TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-perf-regression-512aa-bisect"
OUT=$WT/perf/rf3/results
cd "$WT" || exit 1

run() {  # run <arm> <aa>
  arm=$1; aa=$2; tag="${arm}_${aa}"
  extra=""
  [ "$arm" = pre ] && extra="TT_BIO_ACCURATE_SOFTMAX_AB=rf3.tri_att"
  echo "=== $tag start $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg) ==="
  env PYTHONPATH="$PP" $LEASE_ENV $extra \
    "$PY" perf/rf3/trunk_decompose.py --aa "$aa" --n_recycles 2 \
      --out "$OUT/bisect_dec_${tag}.json" > "$OUT/bisect_dec_${tag}.log" 2>&1
  echo "=== $tag exit $? $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg) ==="
  grep -E "s/recycle|fp32_softmax |latched|rung" "$OUT/bisect_dec_${tag}.log" | head -8
}

run fix 768
run pre 768
run fix 1024
run pre 1024
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
