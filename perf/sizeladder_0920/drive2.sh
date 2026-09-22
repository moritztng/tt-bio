#!/bin/bash
# Record one model's whole p300c ladder in ONE process, then check it. $1 = card, $2 = model.
# Env: MAXLOAD (1-min loadavg to wait for), WAIT_S, CEILING, STALL_S.
#
# Supersedes drive.sh, which had no way out of a wedge. Everything drive.sh did, plus:
#
#   A STALL WATCHDOG. boltz-2 at 768 aa wedges on this box: the fold prints `trunk 2/4` and then
#   writes nothing for minutes while the process stays alive at ~1 % CPU
#   (`blackhole-p300c-768aa-host-spin-wedge-signature`). It is nondeterministic and not
#   card-specific -- the 09-15 campaign hit it four times on cards 1/2/3 and concluded card 0 was
#   the safe one, and it wedged on card 0 here at 12:25Z. A queue with no watchdog sits on it
#   until somebody looks. This watches the newest fold log for the model and SIGKILLs the fold
#   when it stops growing, which is what the gate reports as a failed rung, so the queue moves on
#   instead of hanging. SIGKILL rather than SIGINT because that is the only signal observed to
#   clear this wedge.
#
# Why the quiet wait is OUTSIDE any lock: quiet-waiting while holding benchlock starves every
# other row's queue (`benchlock-quiet-wait-inside-lock-starves-queue`).
#
# Why it REFUSES rather than starting anyway when the wait runs out: an additive 5 s of
# contention moves boltz2's k256->512 from 1.42 to 0.81 against a +-0.50 band, and a baseline
# recorded under that term is a gate hole for every later quiet run.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"; model="$2"
maxload="${MAXLOAD:-8.0}"; wait_s="${WAIT_S:-2400}"; stall_s="${STALL_S:-300}"
log="$WT/perf/sizeladder_0920/logs"; work="$WT/perf/sizegate/work"; mkdir -p "$log"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
load1() { cut -d' ' -f1 /proc/loadavg; }

waited=0
while :; do
  l=$(load1)
  awk "BEGIN{exit !($l <= $maxload)}" && break
  if [ "$waited" -ge "$wait_s" ]; then
    echo "$(stamp) REFUSED $model: loadavg $l still above $maxload after ${waited}s;"\
         "a ladder timed here banks an exponent nobody can reproduce"
    exit 75
  fi
  sleep 30; waited=$((waited+30))
done

# $1 = pid of the thing to watch over. Kills the innermost fold when its log goes quiet.
watchdog() {
  local parent="$1" last=0 now quiet newest
  while [ -d "/proc/$parent" ]; do
    sleep 30
    newest=$(ls -t "$work/$model"-*.log 2>/dev/null | head -1)
    [ -z "$newest" ] && continue
    now=$(stat -c %Y "$newest" 2>/dev/null || echo 0)
    if [ "$now" != "$last" ]; then last=$now; continue; fi
    quiet=$(( $(date +%s) - now ))
    if [ "$quiet" -ge "$stall_s" ]; then
      # By explicit pid, never a pkill pattern in a compound (`batch-kill-by-pid-list-hits-wrong-
      # process`): only folds this model's own census launched, matched on the work dir path.
      for p in $(pgrep -f "tt_bio.main predict $work/" 2>/dev/null); do
        echo "$(stamp) STALL ${quiet}s on $(basename "$newest") -- SIGKILL fold pid $p"
        kill -9 "$p" 2>/dev/null
      done
      last=0
    fi
  done
}

t0=$(stamp)
echo "$t0 start record $model card $card loadavg $(load1) (waited ${waited}s)"
CEILING="${CEILING:-0.8}" bash "$WT/perf/sizeladder_0920/rec.sh" "$card" "$model" \
  > "$log/rec_${model}_c${card}.log" 2>&1 &
recpid=$!
watchdog "$recpid" & wdpid=$!
wait "$recpid"; rc=$?
kill "$wdpid" 2>/dev/null
t1=$(stamp)
echo "$t1 record $model rc=$rc loadavg $(load1)"
[ "$rc" -ne 0 ] && exit "$rc"

# A record that STARTED quiet can still be unusable: load falling during the ladder times the
# small rungs under contention and the large ones on an empty box, which bends the exponents
# without touching a lever row. protenix-v1 came out that way at 12:09-12:21Z. Judged here,
# before anything is attributed or committed, and BETWEEN the two passes -- the check below
# rewrites these same fold logs rung by rung.
python3 "$WT/perf/sizeladder_0920/rung_load.py" --window "$t0" "$t1" "$model" \
  > "$log/rungload_${model}_c${card}.txt" 2>&1
if grep -q UNUSABLE "$log/rungload_${model}_c${card}.txt"; then
  echo "$(stamp) $model RECORD UNUSABLE: $(grep spread "$log/rungload_${model}_c${card}.txt")"
else
  echo "$(stamp) $model record load-stable: $(grep spread "$log/rungload_${model}_c${card}.txt")"
fi

echo "$(stamp) start check $model card $card loadavg $(load1)"
CEILING="${CEILING:-0.8}" bash "$WT/perf/sizeladder_0920/chk.sh" "$card" "$model" \
  > "$log/chk_${model}_c${card}.log" 2>&1 &
chkpid=$!
watchdog "$chkpid" & wdpid=$!
wait "$chkpid"; rc=$?
kill "$wdpid" 2>/dev/null
echo "$(stamp) check $model rc=$rc loadavg $(load1)"
