#!/usr/bin/env python3
"""Which release-gate arms can the dividing-k lever move, and which are inert by construction?

A default-on lever reddens gates it was never run against, so the flip needs a gate arm. Running
the whole gate to find out which leg moves is the expensive way round: the lever changes the
fused-HiFi k ladder at exactly 20 tile-aligned lengths (perf/bcx_tapedfwd/out/blast.json), and
every rung the gate folds is a declared constant. Cross the two.

Read off `scripts/release_gate.py` itself rather than transcribed, so a rung added later shows up
here instead of silently widening the blast radius.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    import release_gate as G

    changed = {c["n"] for c in
               json.load((REPO / "perf/bcx_tapedfwd/out/blast.json").open())["changed"]}
    measured = {}
    for f in sorted((REPO / "perf/land_standing/out/khole_bh").glob("*.json")):
        for r in json.load(f.open())["rows"]:
            s, d = r["arms"]["shipped"], r["arms"]["dividing"]
            rung = lambda a: [(x["q_chunk"], x["k_chunk"], x["kv_bf"], x["served"])  # noqa: E731
                              for x in a["rungs"]]
            measured[r["n"]] = "identical" if rung(s) == rung(d) else (
                "LEVER SERVES" if d["served"] and not s["served"] else "both decline")

    legs = {"predict/affinity size ladder": set(G.SIZE_LADDER_RUNGS)}
    for m, extra in G.SIZE_LADDER_EXTRA_RUNGS.items():
        legs[f"size ladder extra ({m})"] = set(extra)
    for card, rungs in G.SIZE_LADDER_CARD_RUNGS.items():
        legs[f"size ladder card-only ({card})"] = set(rungs)
    for m, spec in getattr(G, "SIZE_LADDER_DESIGN", {}).items():
        if isinstance(spec, dict) and "rungs" in spec:
            legs[f"design ladder ({m})"] = set(spec["rungs"])
    legs["foldable target 7ROA (117 aa -> padded)"] = {128}

    print(f"{'leg':44s} {'rungs in the changed set':>26}  measured")
    hot = {}
    for name, rungs in sorted(legs.items()):
        hit = sorted(rungs & changed)
        note = ", ".join(f"{n}:{measured.get(n, 'UNMEASURED')}" for n in hit) or "-"
        print(f"{name:44s} {str(hit or '[]'):>26}  {note}")
        for n in hit:
            if measured.get(n) != "identical":
                hot.setdefault(name, []).append(n)

    print(f"\nrungs the gate declares : {sorted(set().union(*legs.values()))}")
    print(f"lengths the lever changes: {sorted(changed)}")
    print(f"\nARMS THAT CAN MOVE: {hot or 'none'}")
    unmeasured = [n for rs in legs.values() for n in (rs & changed) if n not in measured]
    assert not unmeasured, f"a gate rung in the changed set with no firing measurement: {unmeasured}"
    print("checked: every gate rung in the changed set has a measured firing verdict")


if __name__ == "__main__":
    main()
