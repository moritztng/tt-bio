#!/usr/bin/env python3
"""Diff this gate report against the post-b2z2 verdict, leg by leg.

The gate already answers "does each leg reproduce its COMMITTED record". This answers the
different question the K10 wave actually raises: "is any leg worse than it was at the last
GO verdict" -- i.e. new since 0cd6c415, not merely non-PASS. A GAP that was a GAP then is
fine; a GAP that was a PASS then is the thing that blocks a GO.

The post-b2z2 baseline is transcribed from state/tt-bio-full-gate-post-b2z2.md section 4,
which is the record the task points at. Nothing here re-derives a verdict.
"""
import json, sys, pathlib

# post-b2z2 (main 0cd6c415): the six legs that were not a bare PASS. Everything else was PASS.
B2Z2_NON_PASS = {
    "boltz2-prot-nomsa": "GAP",
    "boltz2-9ncy-nomsa": "GAP",
    "openfold3-7xi5-notmpl": "GAP",
    "af2ig-trunk-device": "GAP",
    "af2ig-trunk-monomer": "PASS-caveated",
    "protenix-9ncy-msa": "BLOCKED-REF-REGEN-NEEDED",
}

def main(path):
    r = json.load(open(path))
    legs = r["legs"]
    print(f"legs={len(legs)}  tally={r.get(tally)}  scored={r.get(scored)}  "
          f"wall={r.get(total_wall_s)}s  workers={r.get(workers)}")
    regressed, reproduced, improved = [], [], []
    for leg in legs:
        name, v = leg["leg"], leg["verdict"]
        was = B2Z2_NON_PASS.get(name, "PASS")
        if v.startswith("PASS") and was.startswith("PASS"):
            continue
        if v == was or (v.startswith("PASS") and not was.startswith("PASS")):
            (improved if v.startswith("PASS") else reproduced).append((name, was, v, leg.get("detail", "")))
        else:
            regressed.append((name, was, v, leg.get("detail", "")))
    for title, rows in (("REGRESSED vs post-b2z2 (blocks GO)", regressed),
                        ("reproduces a post-b2z2 non-PASS record", reproduced),
                        ("improved since post-b2z2", improved)):
        print(f"\n## {title}: {len(rows)}")
        for n, was, now, d in rows:
            print(f"  {n:34s} was={was:26s} now={now:26s} {d[:70]}")
    return 1 if regressed else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f/report.json"))
