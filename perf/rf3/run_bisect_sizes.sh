#!/usr/bin/env bash
# 768/1024 aa non-regression for the accurate-softmax scope fix, on the instrument 2399c8f8's
# 1024 aa win was measured with. `pre` restores the regressed route, `fix` is the new default.
#
# Same benchlock discipline as the 512 aa arms: LOAD_WAIT well above the default, because
# benchlock measures anyway with a warning once the default 900 s elapses, and a co-tenanted arm
# is a wrong number rather than a slow one.
#
# Do NOT reach for perf/rf3/gate_ab.sh or exp_ladder.sh here: both hardcode a worktree that no
# longer exists.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-regression-512aa-bisect
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE_ENV="TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-perf-regression-512aa-bisect"
export BENCHLOCK_LOAD_WAIT_S="${BENCHLOCK_LOAD_WAIT_S:-2400}"
export BENCHLOCK_WAIT_S="${BENCHLOCK_WAIT_S:-3600}"
OUT=$WT/perf/rf3/results
cd "$WT" || exit 1

run() {  # run <arm> <aa>
  arm=$1; aa=$2; tag="${arm}_${aa}"
  extra=""
  [ "$arm" = pre ] && extra="TT_BIO_ACCURATE_SOFTMAX_AB=rf3.tri_att"
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-perf-regression-512aa-bisect -- \
    env PYTHONPATH="$PP" $LEASE_ENV $extra \
    "$PY" perf/rf3/trunk_decompose.py --aa "$aa" --n_recycles 2 \
      --out "$OUT/bisect_dec_${tag}.json" > "$OUT/bisect_dec_${tag}.log" 2>&1
  echo "=== $tag exit $? $(date -u +%H:%M:%S) ==="
  grep -E "s/recycle|fp32_softmax |latched" "$OUT/bisect_dec_${tag}.log" | head -6
}

run fix 768
run pre 768
run fix 1024
run pre 1024
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
