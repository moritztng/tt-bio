#!/usr/bin/env python3
"""Read the ceiling walk and the paired round back out of the rungs, with the AICLK each
round was measured at.

The clock comes from `aiclk.tsv`, a 1 Hz log of the card's own sysfs node that runs across the
whole chain: `per_round` carries a min/median/max per round but not how many samples sat under
1200, and on a p150a a round taken under a clamped arbiter is an artifact, not a regression.
"""
import json
import pathlib
import statistics
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"


def clock(path=None):
    """The 1 Hz log that covers these rounds. Each chain writes its own: the pc chains share
    out/aiclk.tsv, the qb2 round keeps one beside its rungs."""
    path = path or (OUT / "aiclk.tsv")
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#"):
            continue
        t, c, _ = ln.split("\t")
        rows.append((float(t), int(c)))
    return rows


def window(rows, t0, t1):
    c = sorted(x for t, x in rows if t0 <= t <= t1)
    if not c:
        return {"n": 0}
    return {"n": len(c), "med": statistics.median(c), "min": c[0], "max": c[-1],
            "under1200": sum(1 for x in c if x < 1200)}


def rounds_of(d: pathlib.Path):
    """Per-round wall seconds, dropping each sitting's first round (compile-laden) and the
    run's last boundary (which `per_round` already drops)."""
    j = json.loads((d / "rung.json").read_text())
    rs = [r for r in j.get("per_round", []) if r["round"] > 1]
    return j, rs


def main():
    clk = clock()
    print("== CEILING (rel012 integration tree, pc card 0, p150a, design entry, 1 trajectory)")
    for d in sorted((OUT / "ceiling").glob("a*/")):
        if not (d / "rung.json").exists():
            print(f"{d.name:>6}  no rung.json (see {d.name}.log)")
            continue
        j, rs = rounds_of(d)
        ax = j.get("evoformer_axis")
        err = (j.get("error") or "")[:120]
        secs = [r["seconds"] for r in j.get("per_round", [])]
        rows = json.loads((d / "rounds.json").read_text()) if (d / "rounds.json").exists() else []
        w = window(clk, rows[0]["t"], rows[-1]["t"]) if len(rows) > 1 else {"n": 0}
        print(f"{d.name:>6}  axis={ax} rounds={j.get('rounds_done')} "
              f"secs={[round(s, 1) for s in secs]} aiclk={w} err={err}")
    print()
    for sub in ("round", "round_p300c"):
        if (OUT / sub).exists():
            print(f"== {sub} (288 tokens, hPDL1 + 146 aa, 3 trajectories, pro rata)")
            own = OUT / sub / "aiclk.tsv"
            round_table(OUT / sub, clock(own) if own.exists() else clk)
    return 0


def round_table(root, clk):
    """A round is PRO RATA: three interleaved slots finish three rounds in one slot-to-slot
    interval, which is how v0.11.0 computed the 6.00 s this release is measured against."""
    for arm in ("tree", "tag011"):
        allr, wins = [], []
        for d in sorted(root.glob(f"{arm}[0-9]")):
            if not (d / "rung.json").exists():
                continue
            j, rs = rounds_of(d)
            slots = len({r["slot"] for r in j.get("per_round", [])}) or 1
            for r in rs:
                allr.append(r["seconds"] / slots)
            rows = json.loads((d / "rounds.json").read_text())
            for a, b in zip(rows, rows[1:]):
                wins.append((a["t"], b["t"]))
        if not allr:
            print(f"  {arm}: no rounds yet")
            continue
        cs = [c for t0, t1 in wins for t, c in clk if t0 <= t <= t1]
        print(f"  {arm}: n={len(allr)} median={statistics.median(allr):.3f}s "
              f"min={min(allr):.3f} max={max(allr):.3f} "
              f"| AICLK med={statistics.median(cs) if cs else '-'} "
              f"min={min(cs) if cs else '-'} under1200={sum(1 for c in cs if c < 1200)} n={len(cs)}")


if __name__ == "__main__":
    sys.exit(main())
