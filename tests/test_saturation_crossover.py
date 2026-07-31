#!/usr/bin/env python3
"""Exercise crossover_block on curves whose answer is known by construction.

Deliverable 4's headline -- whether the best generator depends on the sampling budget --
rests entirely on this detector, and the live shared subset has one target and one leader,
so the multi-generator paths have never run. Synthetic curves here, never in section 7.

The scenarios live inside run_checks() rather than at module level: pytest imports this
file during collection, so anything raising (or exiting) at import aborts the whole
session with INTERNALERROR instead of failing one test.
"""
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "abag_xm_saturation_analysis.py"


def _load_analysis():
    spec = importlib.util.spec_from_file_location("an", SCRIPT)
    an = importlib.util.module_from_spec(spec)
    sys.modules["an"] = an
    spec.loader.exec_module(an)
    return an


def run_checks():
    """Run every scenario and return the names of the checks that failed."""
    an = _load_analysis()
    G = an.GRID
    fails = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        got  {got}\n        want {want}")
            fails.append(name)

    # 1. A real crossover: A leads early, B overtakes and stays ahead. Built so the switch
    #    happens between N=64 and N=100 and nowhere else.
    a = {m: 0.50 + 0.0001 * i for i, m in enumerate(G)}          # nearly flat, starts high
    b = {m: 0.10 + 0.60 * (i / (len(G) - 1)) for i, m in enumerate(G)}  # climbs past it
    r = an.crossover_block({"A": a, "B": b})
    switch = [(f["from"], f["to"], f["between"]) for f in r["crossovers"]]
    print("test 1 — one genuine crossover")
    check("exactly one flip", len(r["crossovers"]), 1)
    check("flip direction A -> B", switch[0][:2] if switch else None, ("A", "B"))
    check("leader at smallest N is A", r["leader_by_n"][G[0]]["leader"], "A")
    check("leader at largest N is B", r["leader_by_n"][G[-1]]["leader"], "B")
    check("verdict mentions a crossover", r["crossover_verdict"].startswith("crossover(s):"), True)

    # 2. No crossover: A dominates everywhere.
    a2 = {m: 0.80 for m in G}
    b2 = {m: 0.20 for m in G}
    r2 = an.crossover_block({"A": a2, "B": b2})
    print("test 2 — clear domination, no crossover")
    check("no flips", r2["crossovers"], [])
    check("verdict says no crossover", r2["crossover_verdict"].startswith("no crossover:"), True)
    check("A named as the leader", "A" in r2["crossover_verdict"], True)

    # 3. Ties must not be reported as leads or as flips. Exactly equal curves everywhere.
    a3 = {m: 0.55 for m in G}
    b3 = {m: 0.55 for m in G}
    r3 = an.crossover_block({"A": a3, "B": b3})
    print("test 3 — exact tie at every N")
    check("no flips", r3["crossovers"], [])
    check("no leader named anywhere",
          [v["leader"] for v in r3["leader_by_n"].values()], [None] * len(G))
    check("tied_among lists both", sorted(r3["leader_by_n"][G[0]]["tied_among"]), ["A", "B"])
    check("verdict says no single generator", "no single generator" in r3["crossover_verdict"], True)

    # 4. A sub-bar lead is a tie, not a crossover: B leads by less than Q2_TIE_PP late on.
    eps = (an.Q2_TIE_PP / 100.0) / 2
    a4 = {m: 0.60 for m in G}
    b4 = {m: (0.10 if i < 8 else 0.60 + eps) for i, m in enumerate(G)}
    r4 = an.crossover_block({"A": a4, "B": b4})
    print(f"test 4 — B ends ahead by {eps * 100:.4f} pp, under the {an.Q2_TIE_PP} pp bar")
    check("sub-bar lead is not a crossover", r4["crossovers"], [])
    check("no leader named at the largest N", r4["leader_by_n"][G[-1]]["leader"], None)

    # 5. A tie in the middle must not break the incumbent's run into two spurious flips.
    a5 = {m: (0.60 if i != 6 else 0.30) for i, m in enumerate(G)}
    b5 = {m: (0.20 if i != 6 else 0.30) for i, m in enumerate(G)}
    r5 = an.crossover_block({"A": a5, "B": b5})
    print("test 5 — momentary tie mid-curve, A otherwise always ahead")
    check("no flips across the tie", r5["crossovers"], [])
    check("A still the leader after the tie", r5["leader_by_n"][G[-1]]["leader"], "A")

    return fails


def test_crossover_block():
    assert run_checks() == []


if __name__ == "__main__":
    failed = run_checks()
    print()
    print("FAILURES:", failed if failed else "none")
    sys.exit(1 if failed else 0)
