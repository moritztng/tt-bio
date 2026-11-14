#!/bin/bash
# silu_bench.py arms on one chip under its flock, each arm its own process, ROUNDS rounds alternating base f32.
#   perf/spd_swiglu/run_silu.sh CHIP OUT [ROUNDS] [silu_bench.py args...]
# Same environment as perf/spd/run_chip.sh: ~/japanfold/env.sh if present, private lease dir, LOCK/WAIT/PY overrides.
# The overlay arms JIT the whole kernel set once per runtime root (~2 min warm cache after); the first round pays it.
set -u
CHIP=$1 OUT=$2 ROUNDS=${3:-2}; shift 3 2>/dev/null || shift $#
ROW=${TT_BIO_LEASE_HOLDER:-spd-swiglu}
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
say "chip $CHIP waiting for flock"
flock -w "${WAIT:-600}" 9 || { say "flock busy, stopping"; exit 3; }
say "start chip $CHIP head $(git rev-parse --short HEAD) rounds $ROUNDS"
for r in $(seq 1 "$ROUNDS"); do
  for arm in base f32; do
    # One JIT cache per arm: tt-metal does not key binaries on header contents (metal_overlay.enable).
    TT_METAL_CACHE=$HOME/spd/$ROW/jit/$arm timeout -s TERM 1500 timeout -s INT 1380 "${PY:-python}" perf/spd_swiglu/silu_bench.py --arm $arm \
      --out "$OUT/$arm.r$r.json" "$@" > "$OUT/$arm.r$r.log" 2>&1
    say "round $r arm $arm rc=$?"
  done
done
say "end"
