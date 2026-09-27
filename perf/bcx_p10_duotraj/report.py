#!/usr/bin/env python3
"""The amortised seconds per gradient round, over the window in which BOTH trajectories ran.

The quantity is `wall / rounds completed by both`, which is what `bcx-gpuref` reports
(89.93 s / 125 rounds) and the unit the campaign gates on. Two things make it not a
division of the process wall:

* the two trajectories do not start together. `B` waits for `A` to clear its compile round
  and then pays its own, so the first stretch of the run has one trajectory on the card and
  the second has none. Counting those seconds would price the interleave against a serial
  arm it was not running in.
* round 1 of each trajectory is its compile and is dropped, here as everywhere in this
  campaign.

So the window is `[latest first warm round start, earliest last round end]` across slots,
and only rounds that lie WHOLLY inside it are counted. On a serial arm the slots do not
overlap at all and the window is empty, which is the correct answer for that arm: its
amortised round is just its median warm round.

    report.py out/<tag> [out/<tag> ...]
"""
import json
import pathlib
import statistics
import sys


def rounds(events, slot):
    """`[(round, t_start, t_end)]` for one slot, in order, last round closed by round_stop."""
    marks = [(e["round"], e["t0"], e["kind"]) for e in events
             if e["kind"] in ("round_start", "round_stop") and e.get("slot") == slot]
    marks.sort(key=lambda m: m[1])
    out = []
    for (n, t0, kind), (_, t1, _k) in zip(marks, marks[1:]):
        if kind == "round_start":
            out.append((n, t0, t1))
    return out


def clock(samples, t0, t1):
    got = [(c, l) for t, c, l in samples if t0 <= t <= t1]
    if not got:
        return {"n": 0}
    clk = sorted(c for c, _ in got)
    return {"n": len(clk), "aiclk_med": clk[len(clk) // 2], "aiclk_min": clk[0],
            "load1": round(sum(l for _, l in got) / len(got), 2)}


def one(path):
    d = json.loads((pathlib.Path(path) / "round_events.json").read_text())
    st, ev = d["stamp"], d["events"]
    slots = sorted({e.get("slot") for e in ev if e["kind"] == "round_start"} - {None})
    per = {s: rounds(ev, s) for s in slots}
    warm = {s: [r for r in per[s] if r[0] >= 2] for s in slots}

    print(f"\n=== {path} | interleave={st.get('interleave')} | commit {st['commit'][:9]} "
          f"| {st['host']} card {st['card']} ===")
    for s in slots:
        walls = [t1 - t0 for _n, t0, t1 in warm[s]]
        if not walls:
            print(f"  slot {s}: no warm rounds")
            continue
        print(f"  slot {s}: {len(walls)} warm rounds, median {statistics.median(walls):.3f} s, "
              f"min {min(walls):.3f}, max {max(walls):.3f}")

    if len(slots) < 2 or any(not warm[s] for s in slots):
        print("  not every slot has a warm round, so there is no window to amortise over")
        return
    start = max(warm[s][0][1] for s in slots)
    end = min(warm[s][-1][2] for s in slots)
    if end <= start:
        print("  no common window: the two trajectories never ran at the same time, so this "
              "arm's amortised round is its median warm round")
        return
    # Rounds are counted PRO RATA: a round that straddles an edge of the window contributes
    # the fraction of itself that is inside it. Counting only whole rounds throws away their
    # seconds while keeping the window that contains them, which on a short arm halves the
    # answer -- duo4 read 17.3 s/round that way against a true 8.7.
    def share(r):
        lo, hi = max(r[1], start), min(r[2], end)
        return max(0.0, (hi - lo)) / (r[2] - r[1])
    inside = {s: sum(share(r) for r in warm[s]) for s in slots}
    n = sum(inside.values())
    wall = end - start
    print(f"  common window {wall:.3f} s, rounds inside (pro rata): "
          f"{ {s: round(v, 2) for s, v in inside.items()} } = {n:.2f}")
    if n:
        print(f"  **amortised {wall / n:.3f} s/round**  "
              f"RATIO {wall / n / 0.6958:.2f}x against the 0.6958 s H200 reference")
    print(f"  AICLK/load in the window: {clock(d.get('aiclk', []), start, end)}")
    g = st.get("gate") or {}
    print(f"  gate waited_s {g.get('waited_s')}  held_s {g.get('held_s')}  "
          f"seams {g.get('seams')}")


for p in sys.argv[1:]:
    one(p)
