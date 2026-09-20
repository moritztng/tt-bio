#!/bin/bash
# Record one model's whole p300c ladder in ONE process, then check it in the next process.
# $1 = card, $2 = model.  Env: MAXLOAD (1-min loadavg to wait for), WAIT_S, CEILING.
#
# Why the wait is OUTSIDE any lock: quiet-waiting while holding benchlock starves every other
# row's queue (`benchlock-quiet-wait-inside-lock-starves-queue`). This polls /proc/loadavg and
# starts when the box is calm enough.
#
# Why it REFUSES instead of starting anyway when the wait runs out: the gate scores two exponents
# over 256->512 and 512->768, runtime_s carries a size-independent host term, and on boltz2
# (4.1 s, 11.0 s, k 1.42) an additive 5 s of contention reads k 0.81 -- outside a +-0.50 band.
# A baseline recorded under that term is a gate hole for every later quiet run, which is worse
# than no record at all. So a loaded box produces a refusal and a resume, never a number.
#
# Why record and check run back to back on the SAME card: it is the only thing that keeps the
# host term common to both sides.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"; model="$2"
maxload="${MAXLOAD:-8.0}"; wait_s="${WAIT_S:-2400}"
log="$WT/perf/sizeladder_0920/logs"; mkdir -p "$log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
load1() { cut -d' ' -f1 /proc/loadavg; }

waited=0
while :; do
  l=$(load1)
  awk "BEGIN{exit !($l <= $maxload)}" && break
  if [ "$waited" -ge "$wait_s" ]; then
    echo "$(stamp) REFUSED $model: loadavg $l still above $maxload after ${waited}s."\
         "A ladder timed here would bank an exponent nobody can reproduce. Resume when quiet."
    exit 75
  fi
  sleep 30; waited=$((waited+30))
done

echo "$(stamp) start record $model card $card loadavg $(load1) (waited ${waited}s)"
CEILING="${CEILING:-0.8}" bash "$WT/perf/sizeladder_0920/rec.sh" "$card" "$model" \
  > "$log/rec_${model}_c${card}.log" 2>&1
rc=$?
echo "$(stamp) record $model rc=$rc loadavg $(load1)"
[ "$rc" -ne 0 ] && exit "$rc"

echo "$(stamp) start check $model card $card loadavg $(load1)"
CEILING="${CEILING:-0.8}" bash "$WT/perf/sizeladder_0920/chk.sh" "$card" "$model" \
  > "$log/chk_${model}_c${card}.log" 2>&1
echo "$(stamp) check $model rc=$? loadavg $(load1)"
