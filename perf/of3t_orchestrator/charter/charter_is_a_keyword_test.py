#!/usr/bin/env python3
"""Demonstrate, rather than assert, that charter conditions 1-4 are keyword tests on prose.

CPU only. No card. Reads the live gate source and the live artifacts; writes nothing but a
verdict to stdout and CHARTER_KEYWORD_TEST.json beside it.

Pass 220 did this for condition 5 (`../defecttriage/gate_is_a_keyword_test.py`). The other four
are the same shape and have a worse exposure: condition 5 at least RUNS on every compose, because
GAP exists. THEIR-TEST, GRADIENTS, TRAJECTORY and COVERAGE have never appeared in any orchestrator
document, so those four clauses have never executed against live content at all. They first run on
the pass that ends the campaign, which is the worst possible moment to discover what they test.

WHAT IS RUN
-----------
One synthetic GO document whose four charter fields contain NO MEASUREMENT -- no number, no
artifact, no file name. Only the English the four regexes look for, plus an explicit admission
that nothing was measured. The clauses are then run verbatim, lifted out of the gate source so
they cannot drift from what actually executes.

Beside it, the same four conditions evaluated against the campaign's real artifacts by
`charter_evidence.py`. If the prose passes while the artifacts say the charter is not met, the
condition is not measuring the port.

I AM NOT GOING TO USE THE DOCUMENT. The evasion is written down so the record contains the
demonstration rather than the exploit, which is how pass 220 handled the same finding.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from charter_evidence import GATE, ROOT, evaluate, lift_spec          # noqa: E402

OUT = Path(__file__).with_name("CHARTER_KEYWORD_TEST.json")

# A GO document that measures nothing. Every field says so in plain words.
NO_MEASUREMENT = """VERDICT: GO
THEIR-TEST: their suite ran. I did not check what it did.
GRADIENTS: every parameter was looked at, at whole-model scope. No figure is quoted here.
TRAJECTORY: 20 steps. I am not saying what they showed.
COVERAGE: all of them. Every one. I did not run anything to find out.
GAP: nothing to report.
"""


def clauses_from_source():
    """Lift CHARTER_GO out of the gate, so this tests what runs and not a copy of it."""
    src = GATE.read_text()
    body = src.split("CHARTER_GO = [", 1)[1].split("\n]\n", 1)[0]
    out = []
    for m in re.finditer(r'\(r"([A-Z-]+)",\s*r"((?:[^"\\]|\\.)*)",\s*\n?\s*r"((?:[^"\\]|\\.)*)",',
                         body):
        out.append((m.group(1), m.group(2), m.group(3)))
    if len(out) != 4:
        sys.exit(f"REFUSING: lifted {len(out)} clauses, expected 4. The gate changed (good -- "
                 f"re-read it) or this script has gone stale.")
    return out


def run_clause(field, need, forbid, doc):
    """_charter_gate's own logic for one clause, character for character."""
    m = re.search(rf"^{field}[^\n]*(?:\n(?![A-Z][A-Z0-9.-]*:).*)*", doc, re.M)
    body = m.group(0) if m else ""
    if not body:
        return False, "field absent"
    if not re.search(need, body, re.I) or re.search(forbid, body, re.I):
        return False, "refused"
    return True, "accepted"


def main() -> int:
    clauses = clauses_from_source()
    prose = {f: run_clause(f, n, fb, NO_MEASUREMENT) for f, n, fb in clauses}
    artifacts = {c["field"]: c for c in evaluate(lift_spec(GATE.read_text()), ROOT)}

    rows = []
    for field, need, forbid in clauses:
        accepted, _ = prose[field]
        a = artifacts.get(field, {})
        rows.append({"field": field, "gate_need_regex": need, "gate_forbid_regex": forbid,
                     "prose_with_no_measurement_accepted": accepted,
                     "artifact": a.get("artifact"), "artifact_says_met": a.get("met"),
                     "artifact_misses": a.get("misses", [])})

    comes_apart = [r["field"] for r in rows
                   if r["prose_with_no_measurement_accepted"] and r["artifact_says_met"] is False]

    OUT.write_text(json.dumps(
        {"gate_source": str(GATE), "document_under_test": NO_MEASUREMENT,
         "conditions": rows, "comes_apart_on": comes_apart}, indent=2) + "\n")

    print("A GO document with no measurement in any of its four charter fields:\n")
    for r in rows:
        print(f"  {r['field']:11s} prose clause: "
              f"{'ACCEPTS' if r['prose_with_no_measurement_accepted'] else 'refuses':7s}   "
              f"artifact: {'MET' if r['artifact_says_met'] else 'NOT MET'}")
        for m in r["artifact_misses"][:2]:
            print(f"                 {m[:150]}")
    print()
    if len(comes_apart) == 4:
        print("CONFIRMED: all four conditions accept a document that measures nothing, while all "
              "four\nartifacts say the charter is not met. Conditions 1-4 are decided by the "
              "words in the\nstate doc, not by the port.")
    else:
        print(f"NOT CONFIRMED on all four this run: {comes_apart}. Read the regexes above; the "
              f"gate may have been fixed.")
    print("\nNot used. The document is a demonstration, not a submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
