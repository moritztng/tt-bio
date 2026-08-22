#!/usr/bin/env bash
# The A/B for the accurate-softmax scope fix at 512 aa, on the page's own harness.
#
# `pre` restores the regressed route through the site selector, `fix` is the new shipped default.
# Interleaved, not all-of-one-then-the-other: all-A-then-all-B reads +13.3% on this box where
# interleaved reads +5.2%. The two processes per arm are also the A/A control.
#
# BENCHLOCK_LOAD_WAIT_S is raised well above its default: benchlock proceeds with a warning once
# the default 900 s elapses, and a co-tenanted arm is a wrong number rather than a slow one. A
# foreign parity gate was folding on this box when the first attempt ran, so waiting is the point.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-regression-512aa-bisect
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE_ENV="TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-perf-regression-512aa-bisect"
REPEAT="${REPEAT:-2}"
export BENCHLOCK_LOAD_WAIT_S="${BENCHLOCK_LOAD_WAIT_S:-2400}"
export BENCHLOCK_WAIT_S="${BENCHLOCK_WAIT_S:-3600}"
OUT=$WT/perf/rf3/page512
cd "$WT" || exit 1

run() {  # run <tag> [extra-env...]
  tag=$1; shift
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-perf-regression-512aa-bisect -- \
    env PYTHONPATH="$PP" $LEASE_ENV "$@" \
    "$PY" perf/rf3/page512_tt.py --repeat "$REPEAT" --arm a0 --label "$tag" \
      --out "$OUT/bisect_${tag}_qb2c2.json" > "$OUT/bisect_${tag}.log" 2>&1
  echo "=== $tag exit $? $(date -u +%H:%M:%S) ==="
  grep -E "^\[$tag\] (median|warm)|acquired|WARNING" "$OUT/bisect_${tag}.log"
}

for tag in "$@"; do
  case $tag in
    pre*) run "$tag" TT_BIO_ACCURATE_SOFTMAX_AB=rf3.tri_att ;;
    fix*) run "$tag" ;;
  esac
done
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
