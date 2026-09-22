#!/usr/bin/env python3
"""Every "what is left" paragraph in the orchestrator's LIVE fields is stamped AND FRESH (A30/D148).

WHY THIS EXISTS
---------------
DIRECTIVE-STATUS closed with "the honest shape of what is left, at pass 199" and was still on the
page at pass 269. Three of its four nouns had stopped being true -- one of them discharged by
`D111 UPDATE 3` INSIDE pass 199, because the summary was composed from the state at the top of the
pass and the row reported before it ended. It was not merely stale; it was wrong on arrival.

WHAT THIS DOES AND DOES NOT CHECK
---------------------------------
It checks that such a paragraph SAYS WHEN IT WAS WRITTEN and that the stamp is no more than ten
passes behind the document's own latest pass. The AGE half is the part that matters: the pass-199
paragraph WAS stamped, so a presence-only check would have passed every one of the seventy passes
it was wrong -- a guard that cannot catch its own motivating defect is ceremony, and this one is
built to fail that test first.

It does NOT check whether the paragraph is true, and that limit is deliberate rather than
laziness:

  * the truth version -- every defect a document names must carry the ledger's current status --
    was built at pass 261 and NOT shipped, at 4 of 5 false positives. A campaign document is mostly
    narration of readings of their day, which is correct frozen;
  * "is left" / "-- OPEN" / "owes one pending check" are not separable from narration by pattern.

So A30 puts the burden on the reader, and this guard bounds how long the reader can be misled: a
summary of what is left either says when it was composed and was re-read within ten passes, or it
goes to PASSLOG where superseded text belongs. Both demands are correct whatever the contents.

SCOPE
-----
The orchestrator state doc's live summary fields only. PASSLOG is excluded -- it is a log, every
entry is stamped by its own heading, and its whole purpose is to hold superseded text verbatim.
DEFECTS.md is excluded too: its entries are stamped by their `### Dn UPDATE (pass N)` headings, and
a paragraph-level scan cannot see a section-level stamp (verified at pass 270 -- the scan's one
DEFECTS hit, D140's "what is still alive is the reference side", is pass-254 narration under a
dated heading and correct frozen).

CPU only. Reads one file.
"""
from __future__ import annotations

import pathlib
import re
import sys

DOC = pathlib.Path("/home/moritz/.coworker/state/of3t-orchestrator.md")

#: A paragraph makes a CURRENT claim about the campaign's remaining work.
LEFT = re.compile(
    r"(what is left|what remains|the honest shape|what is still (?:open|owed|alive|genuinely)|"
    r"still owes|what the campaign owes|left to do)", re.I)

#: Any of these counts as a stamp: an explicit pass number, or a date.
STAMP = re.compile(r"\bpass\s+\d{1,3}\b|\b20\d\d-\d\d-\d\d\b", re.I)

FIELDS = ("VERDICT", "PROVES", "DOESNOT", "GAP", "DIRECTIVE-STATUS", "ROWS")

#: How far behind the document's own latest pass a summary's stamp may fall before it stops being
#: status and becomes history. A30 fixes this at ten.
MAX_AGE = 10

#: The document's current pass, read from PASSLOG's own headings rather than passed in, so the
#: guard cannot be satisfied by lying to it.
PASSLOG_HEAD = re.compile(r"^(?:PASSLOG: )?\*\*Pass (\d{1,3}) ", re.M)


def live_fields(text):
    """Yield (field, body) for the live summary fields, PASSLOG excluded."""
    for name in FIELDS:
        m = re.search(r"^%s:(.*?)(?=^[A-Z][A-Z-]*:)" % re.escape(name), text, re.S | re.M)
        if m:
            yield name, m.group(1)


def current_pass(text):
    """The document's own latest pass, or None if it has no PASSLOG headings."""
    seen = [int(m.group(1)) for m in PASSLOG_HEAD.finditer(text)]
    return max(seen) if seen else None


def offenders(text):
    """(field, why, excerpt) for every summary paragraph that is unstamped or stale."""
    now = current_pass(text)
    out = []
    for field, body in live_fields(text):
        for para in body.split("\n\n"):
            if not LEFT.search(para):
                continue
            # `pass\n269` is the same stamp as `pass 269` -- the paragraphs are hard-wrapped.
            stamps = [int(m.group(1)) for m in re.finditer(r"\bpass\s+(\d{1,3})\b", para, re.I)]
            excerpt = " ".join(para.split())[:110]
            if not stamps and not STAMP.search(para):
                out.append((field, "is not stamped with the pass it was composed in", excerpt))
            elif stamps and now is not None and now - max(stamps) > MAX_AGE:
                out.append((field, "is stamped pass %d, %d passes behind this document's pass %d -- "
                                   "re-audit it against the ledger and re-stamp it, or move it to "
                                   "PASSLOG" % (max(stamps), now - max(stamps), now), excerpt))
    return out


def main():
    text = DOC.read_text()
    bad = offenders(text)

    # Probe 1, and it is the real one: D148 VERBATIM. The pass-199 sentence, stamped exactly as it
    # was, in a document whose PASSLOG has reached pass 269. A presence-only check passes this.
    d148 = ("DIRECTIVE-STATUS: **So the honest shape of what is left, at pass 199**: one located "
            "defect, plus one memory-engineering item, and two questions for Moritz that block "
            "nothing.\n\n"
            "PASSLOG: **Pass 269 - something else entirely.**\n")
    if not offenders(d148):
        print("BROKEN the probe did not fire on D148's own paragraph -- this guard proves nothing",
              file=sys.stderr)
        return 2

    # Probe 2: unstamped fires too.
    if not offenders(d148.replace(", at pass 199", "")):
        print("BROKEN an unstamped summary does not fire", file=sys.stderr)
        return 2

    # Probe 3: it must CLEAR once the paragraph is re-audited and re-stamped, or the guard is a
    # refusal to ever write a summary.
    if offenders(d148.replace("at pass 199", "re-audited at pass 269")):
        print("BROKEN the probe still fires on a FRESHLY stamped summary", file=sys.stderr)
        return 2

    if bad:
        for field, why, para in bad:
            print("  DRIFT %s: a 'what is left' summary %s (A30): %s" % (field, why, para))
        print("FAIL %d summary paragraph(s) a reader would take as current (A30, D148)" % len(bad))
        return 1

    n = sum(1 for _f, b in live_fields(text) for p in b.split("\n\n") if LEFT.search(p))
    print("ok    %d 'what is left' paragraph(s) in the live fields, each stamped and within %d "
          "passes of pass %s (probe fires on D148's own paragraph, clears when re-stamped)"
          % (n, MAX_AGE, current_pass(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
