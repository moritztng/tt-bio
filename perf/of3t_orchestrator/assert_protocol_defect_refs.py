#!/usr/bin/env python3
"""No PROTOCOL clause may rest on a defect whose status has since changed (D142).

PROTOCOL is read top to bottom by every row, and A25's own addendum states the rule: an amendment
that supersedes another clause's conclusion has to say so INSIDE it, because a row stops reading
when it has what it needs. Two clauses were found in that state at pass 258 -- one conditioned on
"while D19 stays open" 62 passes after D19 closed, and A18's worked example calling a RESOLVED
policy mismatch a "3.2x gradient defect".

The check is mechanical and narrow on purpose: for every `Dn` a clause names, if that defect is no
longer UNFIXED, the clause must acknowledge it -- by naming the closing status, or by carrying an
ADDENDUM. It does NOT check figures. That limit is the point of the message: the figures are what a
row copies, and nothing here verifies them.

CPU only. Reads two files.
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from status_vocab import statuses_by_defect   # noqa: E402  the ONE definition

D = pathlib.Path("/home/moritz/.coworker/state/of3t")
PROTO, DEF = D / "PROTOCOL.md", D / "DEFECTS.md"

#: A clause is excused when it says what happened, in any of these forms.
ACK = re.compile(r"\b(?:CLOSED|RESOLVED|REFUTED|WITHDRAWN|FIXED|RECORDED|ADDENDUM|superseded)\b",
                 re.I)

#: SCOPE, and it is deliberately narrow. The first version flagged every clause that NAMES a
#: closed defect and found 9 of 11 to be ordinary provenance -- "Record: D96", "raised after
#: D17's finding" -- which stay correct forever after the defect closes. A guard wrong 80 % of
#: the time gets ignored, which is worse than no guard (the same argument
#: assert_dispatch_card_token.py makes about English prose). So only a LIVE CONDITION on the
#: defect counts: a clause that will read false once the defect closes, which is what
#: "while D19 stays open" was.
LIVE_COND = re.compile(
    r"\b(?:while|until|so long as|as long as)\s+(D\d+)\b[^.]{0,60}?"
    r"\b(?:stays?|remains?|is)\s+(?:open|unfixed)\b"
    r"|\b(D\d+)\s+(?:stays?|remains?)\s+(?:open|unfixed)\b"
    r"|\b(?:blocked by|pending|awaiting)\s+(D\d+)\b",
    re.I)


def clauses(text: str):
    """(tag, body) per A-clause, split on the `**An — ...**` headings PROTOCOL uses."""
    parts = re.split(r"(?m)^(\*\*A\d+[^\n]*)$", text)
    return [(re.match(r"\*\*(A\d+)", parts[i]).group(1), parts[i] + parts[i + 1])
            for i in range(1, len(parts) - 1, 2)]


def stale(proto: str, status: dict[str, str]):
    out = []
    for tag, body in clauses(proto):
        if ACK.search(body):
            continue                       # the clause says what happened; that is the fix
        for m in LIVE_COND.finditer(body):
            d = next(g for g in m.groups() if g)
            st = status.get(d)
            if st and st != "UNFIXED":
                out.append((tag, d, st, " ".join(m.group(0).split())))
    return out


def main() -> int:
    if not PROTO.is_file() or not DEF.is_file():
        print("cannot read PROTOCOL.md / DEFECTS.md on this host -- check skipped, and saying so")
        return 0
    status = statuses_by_defect(DEF.read_text())

    # Probe: the check must fire on a clause naming a non-UNFIXED defect without acknowledging it.
    probe = stale("**A99 — probe.**\nIt converts X to Y while D19 stays open.\n"
                  "**A98 — probe two.** Record: D96, which is provenance and must NOT fire.\n",
                  {"D19": "CLOSED", "D96": "FIXED"})
    if [p[:3] for p in probe] != [("A99", "D19", "CLOSED")]:
        print(f"REFUSING: the D142 probe did not fire ({probe}); this check reads nothing")
        return 1

    bad = stale(PROTO.read_text(), status)
    if bad:
        print("PROTOCOL clause(s) rest on a defect whose status has changed and do not say so:")
        for tag, d, st, quote in bad:
            print(f"  {tag}: \"{quote}\" -- {d} is {st}. Correct it in place (D142)")
        return 1
    print("ok    no PROTOCOL clause carries a LIVE CONDITION on a defect that has since closed "
          "(probe fires on the condition and not on provenance; FIGURES are NOT checked -- D142)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
