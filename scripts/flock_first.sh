#!/bin/bash
# flock_first.sh LOCK CMD [ARG...]: run CMD holding the flock LOCK, ahead of every other waiter.
#
# A release gate leg on a shared card uses it (RELEASING.md, "A release never waits"): while it
# waits for the current holder to finish, every other process blocked on LOCK is SIGSTOPped.
# Nothing that holds the card is touched. The stopped pids are listed in LOCK.gate-stopped and
# stay stopped while the gate keeps the card, so its next leg finds the card free. A reaper,
# started before the wait, continues them FLOCK_FIRST_GRACE seconds (default 180) after this
# script exits, however it exits (SIGKILL and timeouts included), unless another flock_first.sh
# on the same LOCK is alive by then: that one's reaper continues them later.
set -u
lock=$1; shift
reg=$lock.gate-stopped
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
    sleep "$1"
    pgrep -f "flock_first.sh $2 " >/dev/null && exit 0
    [ -s "$3" ] && xargs -r kill -CONT < "$3" 2>/dev/null
    : > "$3"' reap "$grace" "$lock" "$reg" $$ </dev/null >/dev/null 2>&1 &

exec 9>>"$lock"
# The wait is named gate_fanout-wait so a host-wide release-priority watcher (qb1's relprio.sh
# matches "gate_fanout") never stops the gate's own waiter.
until flock -n 9; do stop_waiters; ( exec -a gate_fanout-wait flock -w 2 9 ) && break; done
"$@" 9>&-
