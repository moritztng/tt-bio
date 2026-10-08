#!/bin/bash
# Run bench.py arms back to back on one chip, each arm its own process, under the chip's flock.
#   perf/spd/run_chip.sh CHIP OUTROOT "ARM ARM ..." INPUTS TIMEOUT_S [bench.py args...]
#   perf/spd/run_chip.sh 0 ~/spd/spd-bench/base "exact fast:fast" c730 5400 --warm 3 --share 32
# Writes OUTROOT/<arm name>/bench.jsonl per arm and OUTROOT/run.log. On a JapanFold box it sources
# ~/japanfold/env.sh and uses a private lease dir, so it never waits on (or is taken by) the agent's leases.
# Stop: SIGINT, then SIGTERM. The flock is ~/spd/locks/chip<CHIP>.lock (state/spd/CHIPS.md); LOCK=path overrides it
# (the BH boxes name theirs), WAIT=seconds how long to queue on it (default 600), PY=interpreter (default python).
set -u
CHIP=$1 OUT=$2 ARMS=$3 INPUTS=$4 TMO=$5; shift 5
ROW=${TT_BIO_LEASE_HOLDER:-spd-bench}
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "${LOCK:-$HOME/spd/locks/chip$CHIP.lock}"
say "chip $CHIP waiting for flock"
flock -w "${WAIT:-600}" 9 || { say "flock busy, stopping"; exit 3; }
say "start chip $CHIP head $(git rev-parse --short HEAD) arms [$ARMS] inputs $INPUTS"
for arm in $ARMS; do
  name=${arm%%:*}
  timeout -s TERM $((TMO + 120)) timeout -s INT "$TMO" \
    "${PY:-python}" perf/spd/bench.py --out "$OUT/$name" --chip "$CHIP" --arm "$arm" --inputs "$INPUTS" "$@" \
    > "$OUT/$name.log" 2>&1
  say "arm $arm rc=$?"
done
say "end"
