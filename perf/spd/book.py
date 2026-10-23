#!/usr/bin/env python3
"""The one card arbiter of a QuietBox (qb1, qb2): a booking file that every runner reads.

~/spd/feed/windows.tsv holds one line per booking, "START END OWNER CARDS NOTE" (epochs, cards comma-separated).
Release-gate legs never book: they keep absolute priority and every runner yields to them as before.

    book.py add OWNER CARDS SECONDS [--now] [--note TEXT]   book a window, print "START END"
    book.py rm OWNER                                        drop OWNER's booking
    book.py next CARD [--except OWNER]                      earliest start of a booking on CARD not yet over
    book.py active CARD [--except OWNER]                    owner of the booking on CARD now (rc 1 if none)
    book.py list

`add` replaces OWNER's booking. Without --now it starts at the earliest time all CARDS are clear: after every feed job
running on them (its worst-case end, ~/spd/feed/jobs/<job>.until) and outside every other booking. A taker re-runs `add`
while it waits, so its start moves earlier when the work ahead ends early and never moves later. --now books from now
regardless (a hold, e.g. the gapboard column on its card); a hold that overlaps a window yields to it at its next
boundary. Feed starts a job on a card only if the job's worst-case end comes before the card's next booking.

A started booking lapses after LAPSE_S with none of its cards busy (a /dev/tenstorrent node open), sampled by every
book.py call (feed waiters call `next` once a minute). A gap of more than GAP_S between samples restarts the grace, so
an unsampled hour never drops a booking. Lapses are logged to windows.tsv.lapsed; the owner books again with `add`.
Why: on 10-23 four RFD4 bookings held all of qb2 with one RFD4 process running and 14 feed jobs queued.

Why: on 10-16 the gapboard chain and a cfd3 take both claimed qb1 through renamed `sleep` sentinels; each yielded only
to "real" legs, and three cards sat idle for hours with 21 feed jobs queued.
"""
import fcntl
import glob
import os
import sys
import time

H = os.path.expanduser("~")
F = os.environ.get("SPD_WINDOWS", H + "/spd/feed/windows.tsv")
J = os.environ.get("SPD_FEED_JOBS", H + "/spd/feed/jobs")
NEVER = 9999999999
LAPSE_S = 1800
GAP_S = 300


def read(now):
    out = []
    try:
        lines = open(F).read().splitlines()
    except OSError:
        return out
    for line in lines:
        p = line.split("\t")
        if len(p) >= 4 and p[0].isdigit() and p[1].isdigit() and int(p[1]) > now:
            out.append((int(p[0]), int(p[1]), p[2], {int(c) for c in p[3].split(",") if c.isdigit()},
                        p[4] if len(p) > 4 else ""))
    return out


def write(rows):
    t = F + ".tmp"
    with open(t, "w") as f:
        for s, e, o, cs, n in sorted(rows):
            f.write("\t".join(["%d" % s, "%d" % e, o, ",".join(map(str, sorted(cs)))] + ([n] if n else [])) + "\n")
    os.replace(t, F)


def feed_end(cards):
    """Worst-case end of the feed jobs running on `cards`."""
    m = 0
    for sh in glob.glob(J + "/*.sh"):
        try:
            if (int(open(sh + ".card").read()) in cards and os.path.exists(sh + ".start")
                    and not os.path.exists(sh + ".end")):
                m = max(m, int(open(sh + ".until").read()))
        except (OSError, ValueError):
            pass
    return m


def busy_now():
    """Card indices (feed numbering: sysfs nodes sorted by PCI address) whose device node some process holds open."""
    sy = os.environ.get("TT_SYSFS", "/sys/class/tenstorrent")
    try:
        nodes = sorted((os.path.basename(os.path.realpath(d + "/device")), d.rsplit("!", 1)[-1])
                       for d in glob.glob(sy + "/*"))
    except OSError:
        return set()
    card = {"/dev/tenstorrent/" + n: i for i, (_, n) in enumerate(nodes)}
    out = set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            c = card.get(os.readlink(fd))
        except OSError:
            continue
        if c is not None:
            out.add(c)
    return out


def lapse(rows, now, busy=busy_now):
    """Drop started bookings none of whose cards was busy in the last LAPSE_S (see the module doc)."""
    seen = {}
    try:
        for line in open(F + ".seen"):
            k, v = line.split()
            seen[int(k)] = int(v)
    except (OSError, ValueError):
        seen = {}
    if now - seen.get(-1, 0) > GAP_S:
        seen = {}
    for c in busy():
        seen[c] = now
    keep, gone = [], []
    for r in rows:
        s, _, _, cs, _ = r
        for c in cs:
            seen.setdefault(c, now)
        idle = min(now - max(seen[c], s) for c in cs) if cs else 0
        (gone if s <= now and idle >= LAPSE_S else keep).append(r)
    seen[-1] = now
    t = F + ".seen.tmp"
    with open(t, "w") as f:
        f.writelines("%d %d\n" % kv for kv in sorted(seen.items()))
    os.replace(t, F + ".seen")
    if gone:
        with open(F + ".lapsed", "a") as f:
            for s, e, o, cs, _ in gone:
                f.write("%s lapsed %s cards %s, idle %d s\n" % (time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(now)), o,
                                                                ",".join(map(str, sorted(cs))), LAPSE_S))
    return keep


def earliest(rows, cards, dur, start):
    """First start >= `start` at which [start, start + dur) overlaps no booking on `cards`."""
    while True:
        hit = [e for s, e, _, cs, _ in rows if cs & cards and s < start + dur and e > start]
        if not hit:
            return start
        start = max(hit)


def main(a):
    now = int(time.time())
    os.makedirs(os.path.dirname(F), exist_ok=True)
    with open(F + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = read(now)
        kept = lapse(rows, now)
        if kept != rows:
            rows = kept
            write(rows)
        opt = lambda k: a[a.index(k) + 1] if k in a else None
        if a[0] == "add":
            owner, cards, dur = a[1], {int(c) for c in a[2].split(",")}, int(a[3])
            rows = [r for r in rows if r[2] != owner]
            s = now if "--now" in a else earliest(rows, cards, dur, max(now, feed_end(cards)))
            rows.append((s, s + dur, owner, cards, opt("--note") or ""))
            write(rows)
            print(s, s + dur)
        elif a[0] == "rm":
            write([r for r in rows if r[2] != a[1]])
        elif a[0] in ("next", "active"):
            c, x = int(a[1]), opt("--except")
            mine = [r for r in rows if c in r[3] and r[2] != x]
            if a[0] == "next":
                print(min([r[0] for r in mine], default=NEVER))
            else:
                on = [r[2] for r in mine if r[0] <= now]
                print(" ".join(on))
                return 0 if on else 1
        elif a[0] == "list":
            for s, e, o, cs, n in sorted(rows):
                print("%s  %s  %-20s cards %-8s %s" % (time.strftime("%H:%MZ", time.gmtime(s)),
                                                      time.strftime("%H:%MZ", time.gmtime(e)), o,
                                                      ",".join(map(str, sorted(cs))), n))
        else:
            print(__doc__, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["help"]))
