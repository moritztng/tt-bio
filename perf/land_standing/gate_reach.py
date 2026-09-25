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
import re
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

    # A rung only matters if the model folding it REACHES the fused-HiFi route at all. The route
    # is per-site: `_fused_hifi_on(self.fused_hifi)` guards the only call, and `fused_hifi` comes
    # from `triatt_sdpa_hifi_site(token, default)`. Parse the sites out of the source so the
    # defaults are read, not transcribed.
    sites, on_by_default = {}, set()
    pat = re.compile(r"triatt_sdpa_hifi_site\(\s*[\"'](?P<tok>[\w.]+)[\"']\s*"
                     r"(?:,\s*(?P<dflt>True|False)\s*)?\)")
    for f in sorted((REPO / "tt_bio").rglob("*.py")):
        for m in pat.finditer(f.read_text()):
            tok, dflt = m.group("tok"), m.group("dflt") == "True"
            sites[tok] = dflt
            if dflt:
                on_by_default.add(tok.split(".")[0])
    print("fused-HiFi sites and their shipped defaults:")
    for tok in sorted(sites):
        print(f"  {tok:28s} {sites[tok]}")
    print(f"models reaching the route by default: {sorted(on_by_default) or 'NONE'}\n")

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
    # A leg is only live if its MODEL is one that reaches the route. `rf3` appears in the extra
    # rungs and `rfd3` owns a design ladder, and neither wires a default-on site, so a rung of
    # theirs inside the changed set still cannot move.
    live = {name: ns for name, ns in hot.items()
            if any(m in name for m in on_by_default)}
    print(f"\nrungs in the changed set whose arm also FIRES the lever: {hot or 'none'}")
    print(f"ARMS THAT CAN MOVE (fires AND the model reaches the route): {live or 'NONE'}")
    if not live:
        print("\nSo the release gate cannot see this lever. Every rung it folds for a model that\n"
              "reaches the fused-HiFi route is outside the 20 changed lengths, and every rung\n"
              "inside them belongs to a model that never enters the route. A green full gate\n"
              "would be green by construction; certifying the flip needs a NEW rung, at a changed\n"
              "length, for a model with the site on.")
    unmeasured = [n for rs in legs.values() for n in (rs & changed) if n not in measured]
    assert not unmeasured, f"a gate rung in the changed set with no firing measurement: {unmeasured}"
    print("checked: every gate rung in the changed set has a measured firing verdict")


if __name__ == "__main__":
    main()
