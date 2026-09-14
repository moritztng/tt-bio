#!/usr/bin/env python3
"""Diff this gate report against the post-b2z2 verdict, leg by leg.

The gate already answers "does each leg reproduce its COMMITTED record". This answers the
different question the K10 wave actually raises: "is any leg worse than it was at the last
GO verdict" -- i.e. new since 0cd6c415, not merely non-PASS. A GAP that was a GAP then is
fine; a GAP that was a PASS then is the thing that blocks a GO.

The baseline is read from the post-b2z2 run's own report.json, not transcribed from the
state doc's prose, so a typo in a hand-copied table cannot invent or hide a regression.
If that artifact is missing the transcribed table below is used instead, and the header
says which source answered.
"""
import json, os, sys

B2Z2_REPORT = "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-b2z2/gate-0cd6c415/report.json"

# Fallback: post-b2z2 section 4, the six legs that were not a bare PASS. Everything else PASS.
B2Z2_NON_PASS_FALLBACK = {
    "boltz2-prot-nomsa": "GAP",
    "boltz2-9ncy-nomsa": "GAP",
    "openfold3-7xi5-notmpl": "GAP",
    "af2ig-trunk-device": "GAP",
    "af2ig-trunk-monomer": "PASS-caveated",
    "protenix-9ncy-msa": "BLOCKED-REF-REGEN-NEEDED",
}


def baseline():
    if os.path.exists(B2Z2_REPORT):
        r = json.load(open(B2Z2_REPORT))
        return {l["leg"]: l["verdict"] for l in r["legs"]}, B2Z2_REPORT
    return dict(B2Z2_NON_PASS_FALLBACK), "transcribed table (post-b2z2 report.json absent)"


def main(path):
    r = json.load(open(path))
    legs = r["legs"]
    base, src = baseline()
    print("legs=%d  tally=%s  scored=%s  wall=%ss  workers=%s"
          % (len(legs), r.get("tally"), r.get("scored"), r.get("total_wall_s"), r.get("workers")))
    print("baseline source: %s" % src)

    regressed, reproduced, improved, unknown = [], [], [], []
    for leg in legs:
        name, v = leg["leg"], leg["verdict"]
        if name not in base:
            unknown.append((name, "(absent at post-b2z2)", v, leg.get("detail", "")))
            continue
        was = base[name]
        if v.startswith("PASS") and was.startswith("PASS"):
            continue
        if v == was:
            reproduced.append((name, was, v, leg.get("detail", "")))
        elif v.startswith("PASS"):
            improved.append((name, was, v, leg.get("detail", "")))
        else:
            regressed.append((name, was, v, leg.get("detail", "")))

    for title, rows in (("REGRESSED vs post-b2z2 (blocks GO)", regressed),
                        ("NEW leg, no post-b2z2 record (judge by hand)", unknown),
                        ("reproduces a post-b2z2 non-PASS record", reproduced),
                        ("improved since post-b2z2", improved)):
        print("\n## %s: %d" % (title, len(rows)))
        for n, was, now, d in rows:
            print("  %-34s was=%-26s now=%-26s %s" % (n, was, now, d[:70]))
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  "/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f/report.json"))
