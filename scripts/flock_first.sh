#!/bin/bash
# flock_first.sh LOCK CMD [ARG...]: run CMD holding the flock LOCK, ahead of every other waiter.
#
# A release gate leg on a shared card uses it (RELEASING.md, "A release never waits"): while it
# waits for the current holder to finish, every other process blocked on LOCK is SIGSTOPped.
# Nothing that holds the card is touched. The stopped pids are listed in LOCK.gate-stopped and
# stay stopped while the gate keeps the card, so its next leg finds the card free; a reaper
# continues them FLOCK_FIRST_GRACE seconds (default 180) after a leg ends if no gate leg holds
# the card by then.
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
        grep -qx "$p" "$reg" 2>/dev/null && continue
        kill -STOP "$p" 2>/dev/null && echo "$p" >> "$reg"
    done
}

reap() {
    setsid bash -c '
        sleep "$1"
        exec 9>>"$2"
        flock -n 9 || exit 0          # a gate leg holds the card again; its own reaper continues them
        [ -s "$3" ] && xargs -r kill -CONT < "$3" 2>/dev/null
        : > "$3"' reap "$grace" "$lock" "$reg" </dev/null >/dev/null 2>&1 &
}

exec 9>>"$lock"
until flock -n 9; do stop_waiters; flock -w 2 9 && break; done
trap reap EXIT
"$@" 9>&-
