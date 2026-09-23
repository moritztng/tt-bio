#!/usr/bin/env python3
"""of3t-recutfin job 1, control: the composer's injection stamp, exercised in both directions.

A check that cannot fire in one direction reports that direction forever (A17). This one is
worse than most: today `pairformer_stack` is the ONLY injected scope in the composition, so the
mixed-pool refusal passes on every real invocation whether it works or not. So it is driven
here, on synthetic pools, through the composer's own function rather than a copy of it.

Seven cases. One pool that must be ACCEPTED and six that must be REFUSED, each violating exactly
one rule and nothing else. The digest is stubbed so no file is read; the real invocation computes
it from the correction file itself.

CPU only, no device, no gradient dump touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "of3t_wholemodel"))
import model_scope  # noqa: E402

ARMS = ["renorm:cond=/x/cond.pt", "renorm:pairformer_stack=/x/trunk.pt"]
STUB = (lambda p: "stub-digest-of-" + Path(p).name)

CASES = [
    ("accepted: the shape of the real pool -- one injected scope, five not",
     ["renorm:cond=not_injected", "renorm:pairformer_stack=graph-cut-external"],
     ["renorm:pairformer_stack=/x/cot.pt"], None),
    ("accepted with nothing declared: the artifact is exactly what it was before this option",
     [], [], None),
    ("REFUSED: the injected scopes disagree -- the one this exists for, and the one no real "
     "invocation can fire today",
     ["renorm:cond=legacy-total-cotangent", "renorm:pairformer_stack=graph-cut-external"],
     ["renorm:cond=/x/legacy.pt", "renorm:pairformer_stack=/x/cot.pt"], "disagree on convention"),
    ("REFUSED: a scope left unnamed would take the least alarming value by default",
     ["renorm:pairformer_stack=graph-cut-external"],
     ["renorm:pairformer_stack=/x/cot.pt"], "are unnamed"),
    ("REFUSED: an injected scope with no correction file cannot be reproduced",
     ["renorm:cond=not_injected", "renorm:pairformer_stack=graph-cut-external"],
     [], "no --injection-correction"),
    ("REFUSED: a not_injected scope carrying a correction",
     ["renorm:cond=not_injected", "renorm:pairformer_stack=not_injected"],
     ["renorm:cond=/x/cot.pt"], "has no correction"),
    ("REFUSED: a convention outside the three the contract names",
     ["renorm:cond=not_injected", "renorm:pairformer_stack=graph_cut_correct"],
     ["renorm:pairformer_stack=/x/cot.pt"], "must be one of"),
    ("REFUSED: a scope that is not one of the --arm parts",
     ["renorm:cond=not_injected", "renorm:pairformer_stack=not_injected",
      "renorm:msa=not_injected"], [], "not one of the --arm parts"),
]


def main() -> int:
    rows, bad = [], []
    for what, inj, corr, want in CASES:
        try:
            got = model_scope.injection_map(ARMS, inj, corr, digest=STUB)
            outcome, detail = "accepted", got
        except SystemExit as e:
            outcome, detail = "refused", str(e)
        row = {"case": what, "injection": inj, "correction": corr,
               "expected": "refused" if want else "accepted", "outcome": outcome,
               "detail": detail}
        if want:
            row["refusal_names"] = want
            row["passed"] = outcome == "refused" and want in str(detail)
        else:
            row["passed"] = outcome == "accepted"
        rows.append(row)
        if not row["passed"]:
            bad.append(what)

    accepted = rows[0]["detail"]
    rep = {
        "what": __doc__.strip().splitlines()[0],
        "host": __import__("socket").gethostname(),
        "row": "of3t-recutfin",
        "device_involved": False,
        "why_no_aiclk": "CPU only; this reads no gradient and opens no device",
        "why": "the mixed-pool refusal passes on every real invocation of this composition "
               "whether or not it works, because only one scope is injected. A check that "
               "cannot fire in one direction reports that direction forever.",
        "n_cases": len(rows), "n_failed": len(bad), "cases": rows,
        "the_accepted_pool_reads": accepted,
        "verdict": ("every case behaves as pre-registered" if not bad
                    else "FAILED: " + "; ".join(bad)),
    }
    out = REPO / "perf/of3t_recutfin/INJECTION_CONTROLS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
