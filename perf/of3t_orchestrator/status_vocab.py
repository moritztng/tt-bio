#!/usr/bin/env python3
"""The ONE definition of a defect's status vocabulary, and the rules around RECORDED.

Pass 241. The vocabulary was written out five times -- `audit_evidence.py` twice, `triage.py`,
`defecttriage/gate_is_a_keyword_test.py` and `~/.coworker/workstreams/_of3t_donecheck.py` -- which
is the shape of half the defects this campaign has filed against its own instruments. Four of the
five live in this tree and now import from here. The fifth is the gate, which runs from a different
checkout and cannot import this file; `audit_evidence.py` asserts its literal regex against
`PATTERN` instead, which is the two-readers pattern the release gate already uses.

WHY A SIXTH WORD (pass 241). Twenty-nine of a hundred and thirty-four entries declared no status on
any heading (D133), and reading them showed why: most are not defects. They are MEASUREMENTS and
methodological self-catches -- "block 47 is worth one per cent of the model's gradient mass", "I
briefed a row to separate two variables that were rank-identical". There is nothing to fix, so
FIXED is false and UNFIXED is false, and their authors reached for words outside the vocabulary
("MEASURED", "RECORDED") exactly as pass 196 reached for SUPERSEDED.

This is NOT the SUPERSEDED mistake, and the difference is the whole argument. SUPERSEDED was refused
because it is genuinely ambiguous: D87 meant its CLAIM was superseded and D119 meant its NUMBERS
were, and one is a closure while the other is not. RECORDED is unambiguous -- it asserts that the
entry is a finding rather than a defect, so there is nothing to repair.

It is also fenced, because a new status word is an escape hatch unless it is: a defect that has
EVER declared UNFIXED on any heading may not later be restated RECORDED. Retiring an open defect
into "it was only ever a measurement" is precisely the move this word would otherwise enable, and
`recorded_escape_hatches()` below is what refuses it.
"""
from __future__ import annotations

import re

#: Every status word the ledger may use, as a regex alternation body.
WORDS = ("FIXED", "WITHDRAWN", "REFUTED", "CLOSED", "RESOLVED", "ROOT-CAUSED", "RECORDED")

#: The canonical pattern. `UN` prefixes FIXED only in practice, but the optional group is kept
#: exactly as every copy had it so this is a drop-in replacement and not a behaviour change.
PATTERN = r"\b(?:UN)?(?:" + "|".join(WORDS) + r")\b"

STATUS_RE = re.compile(PATTERN)

#: Statuses that mean "this entry needs nothing further". UNFIXED is the only live one, and
#: RECORDED joins the dead list because a finding has no repair pending.
DEAD = ("FIXED", "WITHDRAWN", "REFUTED", "CLOSED", "RESOLVED", "ROOT-CAUSED", "RECORDED")

HEADING_RE = re.compile(r"^### (D\d+)\b(.*)$", re.M)


#: A status word immediately preceded by a negation is not a declaration. D116's heading read
#: "...and D8 is NOT closed by it" and the parser stored CLOSED (D134).
NEGATION_RE = re.compile(r"\b(?:NOT|NEVER|NO LONGER)\s+$", re.I)


def declarations(heading_tail: str) -> list[str]:
    """The status words a heading actually DECLARES: upper-case in the source, unnegated.

    Case matters, and pass 241 is why it has to. `.upper()` made an English word a declaration --
    D69 stored FIXED off "a reading fixed before the arm produced output" for seventy-one passes.
    Pass 240 added a guard that REPORTED that, but the parser went on committing it, so the guard
    and the parse disagreed; and the moment RECORDED joined the vocabulary the same hole silently
    re-read four entries, including D3's "UNFIXED, out of scope, recorded so it is not lost",
    whose last lower-case word would have retired it. The house convention is capitals, always.
    """
    return [m.group(0) for m in STATUS_RE.finditer(heading_tail)
            if not NEGATION_RE.search(heading_tail[:m.start()])]


def statuses_by_defect(doc: str) -> dict[str, str]:
    """Defect -> its LATEST declared status. A heading with no declaration keeps the previous one."""
    out: dict[str, str] = {}
    for m in HEADING_RE.finditer(doc):
        hits = declarations(m.group(2))
        if hits:
            out[m.group(1)] = hits[-1]
    return out


def recorded_escape_hatches(doc: str) -> list[str]:
    """Defects now RECORDED that declared UNFIXED on an earlier heading -- always a failure.

    RECORDED says "this was never a defect". A defect that was once UNFIXED was one, so this is
    the retirement-by-relabelling move, and it is refused rather than judged.
    """
    seen_unfixed: set[str] = set()
    out: list[str] = []
    latest: dict[str, str] = {}
    for m in HEADING_RE.finditer(doc):
        n, h = m.group(1), m.group(2)
        hits = declarations(h)
        if "UNFIXED" in hits:
            seen_unfixed.add(n)
        if hits:
            latest[n] = hits[-1]
    for n, st in latest.items():
        if st == "RECORDED" and n in seen_unfixed:
            out.append(n)
    return sorted(out, key=lambda d: int(d[1:]))


if __name__ == "__main__":
    # Self-test, so a change here that breaks the fence is caught by running this file.
    _doc = ("### D1. RECORDED. a measurement.\n"
            "### D2. UNFIXED. a real one.\n"
            "### D2 UPDATE. RECORDED. actually it was only ever a measurement.\n"
            "### D3. UNFIXED, out of scope, recorded so it is not lost.\n"
            "### D4. and D8 is NOT CLOSED by it.\n"
            "### D5. a reading fixed before the arm produced output.\n")
    _by = statuses_by_defect(_doc)
    assert _by == {"D1": "RECORDED", "D2": "RECORDED", "D3": "UNFIXED"}, _by
    assert recorded_escape_hatches(_doc) == ["D2"], recorded_escape_hatches(_doc)
    print("status_vocab self-test OK:", ", ".join(WORDS))
