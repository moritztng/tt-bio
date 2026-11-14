#!/usr/bin/env python3
"""Tests for book.py's lapse rule: python3 test_book.py"""
import os, sys, tempfile
d = tempfile.mkdtemp()
# book reads its paths at import; set them only for that import so a pytest run does not leak them into test_feed
_env = {k: os.environ.get(k) for k in ("SPD_WINDOWS", "SPD_FEED_JOBS")}
os.environ["SPD_WINDOWS"] = d + "/w.tsv"
os.environ["SPD_FEED_JOBS"] = d + "/jobs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book
for k, v in _env.items():
    os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)

B = book.LAPSE_S
row = lambda s, o, cs: (s, s + 6 * 3600, o, set(cs), "")

def fresh():
    for f in (book.F + ".seen", book.F + ".lapsed"):
        if os.path.exists(f):
            os.remove(f)

# first sample restarts the grace: a booking started 4 h ago is not dropped on deploy
fresh(); t0 = 10**9
rows = [row(t0 - 4 * 3600, "a", [1])]
assert book.lapse(rows, t0, busy=lambda: set()) == rows
# idle LAPSE_S after that, sampled every minute: dropped and logged
for t in range(t0 + 60, t0 + B, 60):
    assert book.lapse(rows, t, busy=lambda: set()) == rows, t
assert book.lapse(rows, t0 + B, busy=lambda: set()) == []
assert "lapsed a cards 1" in open(book.F + ".lapsed").read()

# a card busy at a sample keeps the booking; a multi-card booking lives while any card is busy
fresh()
rows = [row(t0, "b", [1, 2])]
for t in range(t0, t0 + 3 * B, 60):
    assert book.lapse(rows, t, busy=lambda: {2} if (t - t0) % 600 == 0 else set()) == rows, t

# a booking that has not started never lapses
fresh()
rows = [row(t0 + 10 * B, "c", [0])]
for t in range(t0, t0 + 3 * B, 60):
    assert book.lapse(rows, t, busy=lambda: set()) == rows

# busy_now on this host returns a set and never raises
assert isinstance(book.busy_now(), set)
print("ok")

# a held card flock (a gate leg between two device opens) keeps a window from starting on that card; a waiter does not
import fcntl
book.LOCKS, book.PROC_LOCKS = d + "/spd_qbX_*.lock", d + "/locks"
lk = {c: d + "/spd_qbX_chip%d.lock" % c for c in (0, 1)}
for f in lk.values():
    open(f, "w").close()
ino = lambda c: os.stat(lk[c]).st_ino
open(book.PROC_LOCKS, "w").write("1: FLOCK  ADVISORY  WRITE 42 08:02:%d 0 EOF\n1: -> FLOCK  ADVISORY  WRITE 43 08:02:%d 0 EOF\n"
                                 % (ino(0), ino(1)))
assert book.flocked({0, 1}) == {0} and book.flocked({1}) == set()
# against the real /proc/locks: our own flock on chip 1 is seen
book.PROC_LOCKS = "/proc/locks"
with open(lk[1]) as h:
    fcntl.flock(h, fcntl.LOCK_EX)
    assert book.flocked({0, 1}) == {1}
assert book.flocked({0, 1}) == set()
print("ok")

# active: booking owners first, then running feed jobs, then an unbooked held flock or open node
os.makedirs(book.J, exist_ok=True)
book.PROC_LOCKS = d + "/locks"
open(book.PROC_LOCKS, "w").close()
book.LOCKS = d + "/spd_*.lock"
def job(n, card, start=True, end=False):
    sh = book.J + "/" + n + ".sh"
    open(sh, "w").write("true\n"); open(sh + ".card", "w").write("%d" % card); open(sh + ".until", "w").write("%d" % (t0 + 99))
    for x, on in ((".start", start), (".end", end)):
        if on: open(sh + x, "w").close()
job("8-f-ladr2-qb2-opendde_fast_1", 2)
job("waiting", 3, start=False)
job("finished", 3, end=True)
free = lambda: set()
rows = [row(t0 - 60, "gate", [1]), row(t0 + 600, "later", [0])]
assert book.holders(rows, 1, t0, busy=free) == ["gate"]
assert book.holders(rows, 1, t0, x="gate", busy=free) == []
assert book.holders(rows, 2, t0, busy=free) == ["feed:8-f-ladr2-qb2-opendde_fast_1"]
assert book.holders(rows, 3, t0, busy=free) == []            # waiting and ended jobs hold nothing
assert book.holders(rows, 0, t0, busy=free) == []            # a booking that has not started holds nothing
assert book.holders(rows, 0, t0, busy=lambda: {0}) == ["held:unbooked"]
assert book.holders(rows, 1, t0, x="gate", busy=lambda: {1}) == []   # the caller's own booking holds its node
assert book.feed_end({2}) == t0 + 99 and book.feed_end({3}) == 0
lk = d + "/spd_qb2_card3.lock"; open(lk, "w").close()
open(book.PROC_LOCKS, "w").write("1: FLOCK  ADVISORY  WRITE 4242 00:2a:%d 0 EOF\n" % os.stat(lk).st_ino)
assert book.holders(rows, 3, t0, busy=free) == ["held:unbooked"]
open(book.PROC_LOCKS, "w").write("1: -> FLOCK  ADVISORY  WRITE 4242 00:2a:%d 0 EOF\n" % os.stat(lk).st_ino)
assert book.holders(rows, 3, t0, busy=free) == []            # a waiter on the flock holds nothing
print("ok")
