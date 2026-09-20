#!/bin/bash
# Watch one card's ladder fold for the Blackhole host-spin wedge and reclaim the card when it hits.
#
# Why this exists: on 2026-09-20 three folds on qb1 wedged this way and each was found by hand,
# 17 to 33 minutes after it stopped moving. A wedge fails the whole model, so the cost is the walk,
# not the minutes, but the minutes are free to save.
#
# The signature is FORWARD PROGRESS, not CPU: a core pinned at 100 % with /proc/<pid>/io frozen,
# SIGINT and SIGTERM both ignored. wedge_check.py already implements that discriminator and is used
# here as the cheap screen, exactly as it asks to be. Its docstring calls a flag "evidence to look
# closer, never permission to kill", because a fold parked in a long device poll can look wedged
# over a 20 s window, so a flag is never acted on alone: it is confirmed by re-reading that pid's
# own counters over CONFIRM_S (default 300 s), a span no polling wait survives. The three wedges
# this was written for sat frozen for 17, 33 and 137 minutes.
#
# The spin is in the SPAWN CHILD, not the parent, and wedge_check only flags a pid above 50 % CPU,
# so every pid of the fold is screened and the flagged one is what gets confirmed. Confirming an
# aggregate would be wrong: the parent goes on ticking while its worker spins.
#
# On a confirmed wedge: SIGKILL that pid, by explicit pid, then reset the card. The reset is not
# hygiene, it is the finding. Every PCIe DPC containment on this box on 2026-09-20 followed a device
# OPEN on a card whose previous fold had been SIGKILLed, with card_health.py reading that card alive
# at 1350 MHz in between. Killing without resetting hands the next fold a card that dies 12 s after
# it opens it.
#
# Usage: wedge_watch.sh <card> [poll_s]
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CARD=$1
POLL=${2:-120}
CONFIRM_S=${CONFIRM_S:-300}
LOG=$WT/perf/sizegate/campaign/logs/wedge_watch.card$CARD.log
cd "$WT" || exit 1

say() { echo "[wedge-watch $CARD] $(date -u +%FT%TZ) $*" >> "$LOG"; }

# Every pid of this card's fold, parent AND children. The runner pins a workdir per card, so that
# string identifies the card; the lever_census wrapper carries it too and is not a fold.
#
# The children are the point. The compute runs in a `multiprocessing.spawn` child whose argv is
# `spawn_main(tracker_fd=...)` and carries NO workdir, no model and no card, so a selector that
# matches on the workdir alone finds only the parent -- measured here at 0.7 % and 0.0 % CPU while
# its spawn child ran at 391 %. That selector is blind to the one process the wedge lives in, and
# wedge_check ignores anything under 50 % CPU, so the pair would have reported "progressing"
# forever. Match the parent by workdir, then take its children from the process table.
fold_pids() {
  pgrep -f "work-card$CARD" | while read -r p; do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in *lever_census*) continue ;; esac
    case "$c" in *"tt_bio.main predict"*) echo "$p"; pgrep -P "$p" ;; esac
  done | sort -un
}

# syscalls + bytes out of /proc/<pid>/io, the counters wedge_check compares. Empty if unreadable,
# which the caller must treat as "cannot confirm" rather than as a number.
io_sum() {
  [ -r "/proc/$1/io" ] || return 1
  awk '/^(syscr|syscw|rchar|wchar):/ {s += $2} END {print s+0}' "/proc/$1/io" 2>/dev/null
}

say "watching, poll ${POLL}s, confirm ${CONFIRM_S}s"
while true; do
  PIDS=$(fold_pids)
  if [ -z "$PIDS" ]; then sleep "$POLL"; continue; fi
  ARGS=""
  for p in $PIDS; do ARGS="$ARGS --pid $p"; done
  # shellcheck disable=SC2086
  FLAGGED=$(python3 perf/c12_orchestrator/pair_guard/wedge_check.py $ARGS --window 20 \
            | awk '/WEDGED/ {print $2}')
  if [ -z "$FLAGGED" ]; then sleep "$POLL"; continue; fi

  for PID in $FLAGGED; do
    BEFORE=$(io_sum "$PID") || { say "pid $PID unreadable, cannot confirm"; continue; }
    say "pid $PID flagged by wedge_check, confirming over ${CONFIRM_S}s (io=$BEFORE)"
    sleep "$CONFIRM_S"
    AFTER=$(io_sum "$PID") || { say "pid $PID gone during confirmation, not a wedge"; continue; }
    if [ "$AFTER" != "$BEFORE" ]; then
      say "pid $PID moved during confirmation ($BEFORE -> $AFTER), live"
      continue
    fi
    say "WEDGED: pid $PID frozen at io=$AFTER for ${CONFIRM_S}s under load. SIGKILL."
    kill -9 "$PID" 2>/dev/null
    sleep 5
    say "resetting card $CARD before anything opens it again"
    timeout 300 "$HOME/.local/bin/tt-smi" -r "$CARD" >> "$LOG" 2>&1
    sleep 10
    if python3 perf/sizegate/campaign/card_health.py "$CARD" >> "$LOG" 2>&1; then
      say "card $CARD healthy after reset"
    else
      say "card $CARD STILL DEAD after reset, leaving it to the runner's own gate"
    fi
  done
done
