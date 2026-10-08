#!/bin/bash
# spd-bh: everything one Blackhole chip owes the BOARD, in order, under the chip flock from state/spd/CHIPS.md.
# 1. c730 with spd-bench harness, arms exact (the BH "before" line), fast:fast,
#    1 cold + 3 warm each. 2. triangle-attention chunk surface (perf/spd_bh/plan_triatt.py), TA and TT.
# usage: [ARMS=...] [SURFACES="TA TT"] queue.sh RUNROOT CHIP LOCK      e.g. queue.sh ~/spd-bh/runs/p300c-c3 3 ~/spd_qb2_chip3.lock
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${1:?run root}; CHIP=${2:?logical chip}; LOCK=${3:?flock path}
PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
export PYTHONPATH=$WT TT_BIO_LEASE_HOLDER=spd-bh
mkdir -p "$ROOT"; exec >> "$ROOT/queue.log" 2>&1
say(){ echo "$(date -u +%FT%TZ) $*"; }
say "wait for $LOCK, engine $(git -C "$WT" rev-parse --short HEAD) host $(hostname) chip $CHIP"
exec 9>"$LOCK"; flock -w 14400 9 || { say "no lock in 4 h"; exit 1; }
say "lock held, load $(cat /proc/loadavg)"
cd "$WT"
for ARM in ${ARMS:-exact fast:fast}; do
  N=${ARM%%:*}
  say "arm $N start"
  nice -n 5 timeout 5400 "$PY" perf/spd/bench.py --out "$ROOT/$N" --chip "$CHIP" --arm "$ARM" --inputs c730 --warm 3 \
    > "$ROOT/$N.log" 2>&1
  say "arm $N rc=$?"
done
for C in ${SURFACES-TA TT}; do
  "$PY" perf/spd_bh/plan_triatt.py "$ROOT/plan_$C.json" $C
  say "surface $C start"
  TT_VISIBLE_DEVICES=$CHIP nice -n 5 timeout 2400 "$PY" perf/lpx_sdpa/bench.py "$ROOT/surface_$C" "$ROOT/plan_$C.json" "$CHIP" \
    > "$ROOT/surface_$C.log" 2>&1
  say "surface $C rc=$?"
done
say "done, load $(cat /proc/loadavg)"
