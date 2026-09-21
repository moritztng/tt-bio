#!/usr/bin/env python3
"""Demonstrate, rather than assert, that GO condition 5 is a keyword test on GAP prose.

CPU only. No card. Reads the live gate source and the live defect ledger; writes nothing but a
verdict to stdout and GATE_KEYWORD_TEST.json beside it.

THE CLAIM UNDER TEST
--------------------
`workstreams/_of3t_donecheck.py::_charter_gate` ends with:

    if re.search(r"inference users get today|unfixed|still open|remains? open", gap, re.I):
        fail.append("VERDICT: GO while GAP still names unfixed or user-facing defects -- the "
                    "charter is not met while a defect ships to users")

Its message says it tests whether a defect ships to users. The code tests whether four English
phrases appear in one field of one markdown file. Those are different tests, and this script
shows they come apart on the campaign's ACTUAL defect list.

WHAT IS RUN
-----------
Two GAP texts, both naming all forty-four live UNFIXED defects by number, so that
`audit_evidence.py`'s coverage check -- "every UNFIXED defect must be NAMED in GAP" -- passes on
both:

  HONEST   labels each one "(UNFIXED)", which is the campaign's convention and what the ledger
           actually says.
  REWORDED labels each one "(open)" and says nothing else. No measurement differs. No defect is
           closed. Nothing about the port has changed.

If the gate refuses the first and accepts the second, it is not measuring the port.

I AM NOT GOING TO USE THE SECOND ONE. That is the point of writing this down: the evasion is
available, it is four words wide, and the campaign's record should contain the demonstration
rather than the exploit. The bar is Moritz's to move; ask 9612 asks him to.
"""
import json
import re
import sys
from pathlib import Path

D = Path("/home/moritz/.coworker")
GATE = D / "workstreams" / "_of3t_donecheck.py"
DEFECTS = D / "state" / "of3t" / "DEFECTS.md"
OUT = Path(__file__).with_name("GATE_KEYWORD_TEST.json")

STATUS_RE = re.compile(r"\b(?:UN)?(?:FIXED|WITHDRAWN|REFUTED|CLOSED|RESOLVED|ROOT-CAUSED)\b")


def live_unfixed():
    last = {}
    for m in re.finditer(r"^### (D\d+)\b(.*)$", DEFECTS.read_text(errors="replace").upper(), re.M):
        t = STATUS_RE.findall(m.group(2))
        if t:
            last[m.group(1)] = t[-1]
    return sorted((n for n, s in last.items() if s == "UNFIXED"), key=lambda d: int(d[1:]))


def gap_pattern_from_source():
    """Lift the gate's own regex out of its source, so this cannot drift from what runs."""
    src = GATE.read_text()
    m = re.search(r'if re\.search\(r"([^"]*unfixed[^"]*)", gap, re\.I\)', src)
    if not m:
        print("REFUSING: could not find the GAP keyword test in the gate source. Either it was "
              "changed (good -- re-read it) or this script has gone stale.")
        sys.exit(2)
    return m.group(1)


def main() -> int:
    unfixed = live_unfixed()
    pat = gap_pattern_from_source()

    honest = "GAP: " + " ".join(f"**{d} (UNFIXED)**: one line." for d in unfixed)
    reworded = "GAP: " + " ".join(f"**{d} (open)**: one line." for d in unfixed)

    # audit_evidence.py's coverage check: every UNFIXED defect named by number in GAP.
    named = {k: all(re.search(rf"\b{d}\b", t) for d in unfixed)
             for k, t in (("honest", honest), ("reworded", reworded))}
    # the gate's own test, run verbatim
    refused = {k: bool(re.search(pat, t, re.I))
               for k, t in (("honest", honest), ("reworded", reworded))}

    result = {
        "gate_source": str(GATE),
        "gate_gap_pattern": pat,
        "unfixed_named": len(unfixed),
        "honest": {"names_every_unfixed_defect": named["honest"],
                   "gate_refuses_GO": refused["honest"]},
        "reworded": {"names_every_unfixed_defect": named["reworded"],
                     "gate_refuses_GO": refused["reworded"]},
        "comes_apart": named["honest"] and named["reworded"]
                       and refused["honest"] and not refused["reworded"],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    for k in ("honest", "reworded"):
        print(f"{k:9s} names all {len(unfixed)} defects: {named[k]}   "
              f"gate refuses GO: {refused[k]}")
    print()
    if result["comes_apart"]:
        print("CONFIRMED: both texts name every UNFIXED defect; the gate refuses one and accepts "
              "the other. GO condition 5 is decided by four words of prose, not by the ledger.")
    else:
        print("NOT CONFIRMED on this run -- read the pattern above; the gate may have been fixed.")
    print("\nNot used. The reworded text is a demonstration, not a submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
