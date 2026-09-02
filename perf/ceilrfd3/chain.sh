#!/bin/bash
# Walk an RFD3 size ladder on whichever Galaxy chip is free at that instant.
#
# The chips are shared with JapanFold production and with five sibling ceiling tasks, so a
# chip picked from an occupancy listing is already stale by the time the fold starts. The
# claim is therefore the device lease itself: the run is launched pinned to a candidate, and
# `DeviceInUseError` means "someone else has it", not "this size fails". Only a rung that
# reaches the model is recorded as a result.
#
#   TAG=fix DEADLINE=$(( $(date +%s) + 10800 )) bash perf/ceilrfd3/chain.sh 480 512 544 576
#
# The ladder stops at the FIRST failing rung, not at the largest rung that happens to pass:
# a size above a failure is not a ceiling. The failing rung and the last passing rung are
# then each re-run once, so the boundary is a repeated observation rather than a single shot.
set -u
SRC=${SRC:-$PWD}
PY=${PY:-/home/cust-team/mthuening/tt-bio/env/bin/python}
TAG=${TAG:-fix}
BINDER=${BINDER:-100}
CANDS=${CANDS:-"25 26 27 28 29 30 31"}
DEADLINE=${DEADLINE:-$(( $(date +%s) + 10800 ))}
EXTRA_ENV=${EXTRA_ENV:-}
CMD=${CMD:-perf/ceilrfd3/rfd3_cap.py}
STOP_ON_FAIL=${STOP_ON_FAIL:-1}
OUT=$SRC/perf/ceilrfd3/results/$TAG.jsonl
LOG=$SRC/perf/ceilrfd3/results/$TAG.log
mkdir -p "$SRC/perf/ceilrfd3/results"

run_rung() {   # $1 = total residues; echoes PASS / FAIL / NOCHIP
  local total=$1 len=$(( $1 - BINDER )) c out
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    for c in $CANDS; do
      [ "$(date +%s)" -lt "$DEADLINE" ] || break
      echo "[chain] $(date -Is) total=$total umd=$c try" >> "$LOG"
      out=$(cd "$SRC" && env RFD3_CAP_LEN=$len RFD3_CAP_BINDER=$BINDER RFD3_CAP_OUT=$OUT \
            RFD3_CAP_TAG=$TAG TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c \
            TT_BIO_LEASE_HOLDER=worker:ceiling-rfd3 TT_BIO_LEASE_TIMEOUT=10 \
            RFD3_DIGEST_OUT=$OUT $EXTRA_ENV "$PY" "$CMD" 2>&1 | tail -60)
      printf '%s\n' "$out" >> "$LOG"
      case "$out" in
        *DeviceInUseError*|*"is outside this job's card grant"*) continue ;;
      esac
      case "$out" in
        *"ok=True"*) echo PASS; return ;;
        *) echo FAIL; return ;;
      esac
    done
    sleep 45
  done
  echo NOCHIP
}

last_pass=""; first_fail=""
for total in "$@"; do
  r=$(run_rung "$total")
  echo "[chain] $(date -Is) total=$total -> $r" >> "$LOG"
  case "$r" in
    PASS)   last_pass=$total ;;
    FAIL)   first_fail=$total; [ "$STOP_ON_FAIL" = 1 ] && break ;;
    NOCHIP) echo "[chain] $(date -Is) out of time at total=$total" >> "$LOG"; break ;;
  esac
done

# Bisect the 32-residue grid between the last pass and the first failure. A ladder in steps
# of 64 that stops at a failure gives a ceiling good to 64; the token axis buckets to 32, so
# the published ceiling and its negative control have to be one bucket apart, not two.
if [ -n "$last_pass" ] && [ -n "$first_fail" ]; then
  while [ $(( first_fail - last_pass )) -gt 32 ]; do
    mid=$(( (last_pass + first_fail) / 2 / 32 * 32 ))
    [ "$mid" -le "$last_pass" ] && break
    r=$(run_rung "$mid")
    echo "[chain] $(date -Is) bisect total=$mid -> $r" >> "$LOG"
    case "$r" in PASS) last_pass=$mid ;; FAIL) first_fail=$mid ;; *) break ;; esac
  done
fi

for total in $first_fail $last_pass; do
  r=$(run_rung "$total")
  echo "[chain] $(date -Is) repeat total=$total -> $r" >> "$LOG"
done
echo "[chain] $(date -Is) DONE last_pass=$last_pass first_fail=$first_fail" >> "$LOG"
