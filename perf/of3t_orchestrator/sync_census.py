#!/usr/bin/env python3
"""Rewrite the orchestrator state doc's census figures from their live sources.

WHY THIS EXISTS
---------------
`audit_evidence.py` requires VERDICT and ROWS to state five counts -- total defects, UNFIXED,
dispatched rows, concluded rows, concluded markers -- and checks each against its source. Those
checks exist because ROWS once read "twenty-four dispatched, twenty-one concluded" for seven
passes while the campaign ran 39 rows.

They are right to exist and they are not the problem. The problem is that a dispatch wave concludes
rows WHILE a compose is running, so every pass since 272 has spent three or four edit cycles
hand-chasing numbers that moved between the compose and the fix. That is error-prone in the one
direction that matters: it trains me to edit a count until a checker goes quiet.

So the counts are DERIVED here and the checkers still verify them independently. This script is not
a way to satisfy the check -- it reads the same sources the check reads, and if it ever disagreed
with the check the check wins and this is the thing that is broken.

A30's third consequence says a count of a moving ledger does not belong in prose. This is the
concession the ledger's own gate forces: if prose must carry a count, the prose is generated.

Refuses rather than silently doing nothing if any sentence it rewrites is not found.

CPU only. Reads four sources, writes one file.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

D = pathlib.Path("/home/moritz/.coworker")
DOC = D / "state/of3t-orchestrator.md"
DEFECTS = D / "state/of3t/DEFECTS.md"
TRIAGE = D / "state/of3t/UNFIXED_TRIAGE.json"

WORDS = {n: w for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def word(n):
    """Number words in the house style: 'one hundred fifty-three', 'seventy-three'."""
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
            8: "eighty", 9: "ninety"}
    if n in WORDS:
        return WORDS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return tens[t] + ("-" + WORDS[u] if u else "")
    h, r = divmod(n, 100)
    return WORDS[h] + " hundred" + (" " + word(r) if r else "")


def sources():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from status_vocab import statuses_by_defect        # noqa: E402  the ONE definition
    st = statuses_by_defect(DEFECTS.read_text())
    total = len(st)
    unfixed = sum(1 for v in st.values() if v == "UNFIXED")
    conc = D / "state/concluded"
    n_conc = len([d for d in conc.iterdir()
                  if d.name.startswith("of3t-") and "of3t-orchestrator" not in d.name])
    n_markers = len([d for d in conc.iterdir() if "of3t" in d.name])
    n_briefs = len(list((D / "workstreams").glob("of3t-*.txt")))
    tri = json.loads(TRIAGE.read_text())
    counts = tri.get("counts", {})
    def count(*keys):
        for k in keys:
            if k in counts:
                return counts[k]
        return None
    split = (count("SCOPE-EXCLUDED", "scope_excluded"),
             count("USER-FACING", "user_facing"),
             count("CAMPAIGN-INTERNAL", "campaign_internal"))
    if None in split:
        print("REFUSING: UNFIXED_TRIAGE.json has no counts for the three classes, so the split "
              "would be left stale while everything else was updated", file=sys.stderr)
        raise SystemExit(2)
    return total, unfixed, n_conc, n_markers, n_briefs, split


def main():
    total, unfixed, n_conc, n_markers, n_briefs, split = sources()
    text = DOC.read_text()
    subs = [
        (r"ROWS: \*\*[a-z-]+ dispatched, [a-z-]+ concluded, [a-z-]+ live\*\*",
         "ROWS: **%s dispatched, %s concluded, %s live**"
         % (word(n_briefs), word(n_conc), word(max(n_briefs - n_conc, 0)))),
        (r"[A-Z][a-z-]+ dispatched, [a-z-]+ concluded, [a-z-]+ live",
         "%s dispatched, %s concluded, %s live"
         % (word(n_briefs).capitalize(), word(n_conc), word(max(n_briefs - n_conc, 0)))),
        (r"one hundred [a-z-]+ defects, [a-z-]+ UNFIXED",
         "%s defects, %s UNFIXED" % (word(total), word(unfixed))),
        (r"[a-z-]+ of3t markers in `state/concluded`",
         "%s of3t markers in `state/concluded`" % word(n_markers)),
    ]
    if None not in split:
        # The sentence is hard-wrapped, so the pattern has to span the wrap: "...9 USER-FACING,\n
        # 41 campaign-internal**." was what made the first version of this script refuse.
        subs.append((r"\*\*\d+ scope-excluded, \d+ USER-FACING,\s+\d+ campaign-internal\*\*",
                     "**%d scope-excluded, %d USER-FACING, %d campaign-internal**" % split))

    missing = [pat for pat, _r in subs if not re.search(pat, text)]
    if missing:
        print("REFUSING: these census sentences are not where this script expects them, so it "
              "would rewrite nothing and report success:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 2
    for pat, rep in subs:
        text = re.sub(pat, rep, text, count=1)
    DOC.write_text(text)
    print("census: %s defects, %s UNFIXED; %s briefs, %s concluded, %s markers; split %s"
          % (word(total), word(unfixed), word(n_briefs), word(n_conc), word(n_markers),
             "/".join(str(x) for x in split)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
