#!/bin/sh
# The whole ladder, one fold at a time on pc card 0, resume-safe across worker passes.
#
# Detached on purpose (setsid): one rung is up to 75 min and the ladder is a dozen of them,
# so no single worker pass can hold it. Rooted in THIS slug's worktree, never a parent's,
# because a concluded slug's worktree is torn down under a live job.
#
# Rung order per model is 640 FIRST, then 1024. 640 is a control, not a consolation prize:
# Wormhole folds it at this depth in 268 s, so a 640 that stalls on pc says the card cannot
# measure this model at all and every larger rung would be unattributable. Only once 640 is
# proven does 1024 -- the question Moritz asked -- earn its 75 minutes.
set -u
B=/home/moritz/.coworker/wt/bh-1024-of3-openbind-rf3/perf/bh1024
cd "$B" || exit 1
LOG=$B/campaign.log
export RUNG_TIMEOUT=${RUNG_TIMEOUT:-4500}

# Is OUR engine actually on the card? Two traps, both hit for real this pass:
#   - a plain `pgrep -f 'tt_bio.main predict'` matches any SHELL whose command line merely
#     mentions that string, including a monitoring one-liner typed in another terminal. The
#     campaign then waits forever on a free card. The bracket makes the pattern unable to
#     match a command line that contains the pattern itself.
#   - so also require the match to actually be a python process, not a shell quoting it.
engine_running() {
  for p in $(pgrep -f 'tt_bio[.]main predict' 2>/dev/null); do
    case "$(cat /proc/"$p"/comm 2>/dev/null)" in python*) return 0 ;; esac
  done
  return 1
}

wait_for_card() {
  waited=0
  # Two reads, because either alone lies: lsof can report a chip free that a pool still
  # owns (memory japanfold-pool-excluded-chip-reads-free-via-lsof), and a pgrep for our own
  # engine misses any other holder. Busy if EITHER says busy.
  while sudo -n lsof /dev/tenstorrent/0 >/dev/null 2>&1 || engine_running; do
    waited=$((waited + 30))
    # A legitimate rung now runs up to 75 min, so the old 2 h giveup could fire on a healthy
    # neighbour rung and kill the campaign mid-ladder. 10 h covers any real holder.
    [ "$waited" -gt 36000 ] && { echo "GIVEUP card held >10h" >> "$LOG"; return 1; }
    sleep 30
  done
  return 0
}

# Recorded AND settled. A harness kill (worker pass ends, SIGTERM reaches the launcher only)
# leaves a NORESULT row that answers nothing; treating one as done silently dropped the
# headline rung from pass 1's ladder. record.py decides this and writes "conclusive".
done_already() {
  [ -f "$B/results.jsonl" ] || return 1
  grep "\"model\": \"$1\", \"rung\": \"$2\"" "$B/results.jsonl" 2>/dev/null \
    | grep "\"probe\": \"$3\"" | grep -q '"conclusive": true'
}

rung_ok() {
  grep "\"model\": \"$1\", \"rung\": \"$2\"" "$B/results.jsonl" 2>/dev/null \
    | grep -q '"status": "ok"'
}

# A rung that stalled leaves the chip dirty; the next rung must not inherit it.
reset_card() {
  echo "RESET card 0 $(date -u +%FT%TZ)" >> "$LOG"
  timeout 300 /home/moritz/.local/bin/tt-smi -r 0 >> "$LOG" 2>&1
  sleep 20
}

run_one() {  # model rung recyc samp
  done_already "$1" "$2" off && { echo "SKIP $1 $2 (settled)" >> "$LOG"; return 0; }
  wait_for_card || exit 3
  echo "START $1 $2 rec=$3 samp=$4 budget=${RUNG_TIMEOUT}s $(date -u +%FT%TZ)" >> "$LOG"
  sh "$B/run_rung.sh" "$1" "$2" "$3" "$4" off >> "$LOG" 2>&1
  echo "END   $1 $2 $(date -u +%FT%TZ)" >> "$LOG"
  rung_ok "$1" "$2" || reset_card
}

# model:recycling:sampling -- the engine's own resolvers (tt_bio/main.py
# _resolve_recycling_steps / _resolve_sampling_steps, which catalog.ENGINE_DEFAULTS mirrors),
# read out of the source this pass rather than taken on trust.
# Not lowered to make a rung pass: rf3's recycling also sets how many MSA samples the trunk
# sees, so dropping it changes memory AND quality and would answer a different question.
for spec in openfold3:3:200 openbind:3:200 rf3:10:50; do
  m=${spec%%:*}; rest=${spec#*:}; rec=${rest%%:*}; samp=${rest#*:}

  run_one "$m" cut_640 "$rec" "$samp"
  if ! rung_ok "$m" cut_640; then
    echo "CONTROL FAILED $m at 640 -- larger rungs unattributable, skipping model" >> "$LOG"
    continue
  fi

  run_one "$m" tile_1024 "$rec" "$samp"
  if rung_ok "$m" tile_1024; then
    echo "PASS $m at 1024 -- intermediate rungs not needed" >> "$LOG"; continue
  fi

  # 1024 failed, so the ceiling is somewhere in (640, 1024). Descend to find it.
  for rung in tile_896 tile_768; do
    run_one "$m" "$rung" "$rec" "$samp"
    if rung_ok "$m" "$rung"; then
      echo "CEILING $m at $rung" >> "$LOG"; break
    fi
  done
done
echo "CAMPAIGN DONE $(date -u +%FT%TZ)" >> "$LOG"
