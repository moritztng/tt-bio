#!/usr/bin/env python3
"""Tests for book.py's lapse rule: python3 test_book.py"""
import os, sys, tempfile
d = tempfile.mkdtemp()
os.environ["SPD_WINDOWS"] = d + "/w.tsv"
os.environ["SPD_FEED_JOBS"] = d + "/jobs"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book

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
