#!/bin/bash
# flock_first.sh [--book BOOK_PY SECONDS] LOCK CMD [ARG...]: run CMD holding the flock LOCK, ahead of every other waiter.
#
# A release gate leg on a shared card uses it (RELEASING.md, "A release never waits"): while it
# waits for the current holder to finish, every other process blocked on LOCK is SIGSTOPped.
# Nothing that holds the card is touched. The stopped pids are listed in LOCK.gate-stopped and
# stay stopped while the gate keeps the card, so its next leg finds the card free. A reaper,
# started before the wait, continues them FLOCK_FIRST_GRACE seconds (default 180) after this
# script exits, however it exits (SIGKILL and timeouts included), unless another flock_first.sh
# on the same LOCK is alive by then: that one's reaper continues them later.
#
# --book names the host's card booker (perf/spd/book.py, ~/spd/feed/windows.tsv). Once the lock is held the leg books
# its card for SECONDS with `book.py take`, which moves any booking it lands on to the next free slot, and the reaper
# drops the booking the moment this script exits. Without it the booker cannot see the leg: on 11-03 qb1 runners
# found their booked windows started under a release leg and rebooked by hand, one 11.5 h out.
set -u
book= span=
[ "${1:-}" = --book ] && { book=$2 span=$3; shift 3; }
lock=$1; shift
reg=$lock.gate-stopped
card=$(basename "$lock" | sed -nE 's/.*_(card|chip)([0-9]+)\.lock$/\2/p')
owner=release:$(basename "$lock" .lock):$$
[ -n "$card" ] || book=
grace=${FLOCK_FIRST_GRACE:-180}

stop_waiters() {
    local ino p
    ino=$(stat -c %i "$lock") || return
    # Blocked waiters are the "->" lines; field 6 is the pid, field 7 MAJ:MIN:INODE.
    for p in $(awk -v ino="$ino" '$2 == "->" { split($7, d, ":"); if (d[3] == ino) print $6 }' /proc/locks); do
        [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "$$" ] && continue    # our own flock -w
        # Another gate's wait: two gates stopping each other's waiters deadlock (10-09, qb1).
        [ "$(ps -o args= -p "$p" | cut -d' ' -f1)" = gate_fanout-wait ] && continue
        grep -qx "$p" "$reg" 2>/dev/null && continue
        kill -STOP "$p" 2>/dev/null && echo "$p" >> "$reg"
    done
}

setsid bash -c '
    while kill -0 "$4" 2>/dev/null; do sleep 2; done
    [ -n "$5" ] && python3 "$5" rm "$6"
    sleep "$1"
    pgrep -f "flock_first.sh (--book [^ ]+ [0-9]+ )?$2 " >/dev/null && exit 0
    [ -s "$3" ] && xargs -r kill -CONT < "$3" 2>/dev/null
    : > "$3"' reap "$grace" "$lock" "$reg" $$ "$book" "$owner" </dev/null >/dev/null 2>&1 &

# Runners that hold a card across many jobs (qb1's row chains, mp_heavy.sh, gapboard_chain.sh) hand
# it over between jobs only to a process whose argv names the lock and a gate ("<lock> ... gate_fanout").
# This leg's own argv carries neither, so a chain held qb1 card 2 for 2 h past a waiting gate leg
# (10-14). The sentinel follows this script, waiting and running, and dies with it even on SIGKILL.
( exec -a "flock_first $lock gate_fanout" bash -c 'while kill -0 "$0" 2>/dev/null; do sleep 5; done' $$ ) \
    </dev/null >/dev/null 2>&1 9>&- &

exec 9>>"$lock"
# The wait is named gate_fanout-wait so a host-wide release-priority watcher (qb1's relprio.sh
# matches "gate_fanout") never stops the gate's own waiter.
until flock -n 9; do stop_waiters; ( exec -a gate_fanout-wait flock -w 2 9 ) && break; done
[ -n "$book" ] && python3 "$book" take "$owner" "$card" "$span" --note "release leg (flock_first)" >/dev/null 9>&-
"$@" 9>&-
