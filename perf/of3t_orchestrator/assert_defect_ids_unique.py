#!/usr/bin/env python3
"""No two defect entries may claim the same D-number.

Written at pass 308 because I had just done it. Appending a new defect to an 831 KB
`DEFECTS.md` means picking the next free number, and the way I picked it was
`grep -oE '^### D[0-9]+' | sort -n | tail`, which printed `164` and I wrote `D165` --
already taken by an entry about `DEPENDS_ON: none`, filed by an earlier pass. Two entries
under one number is not a cosmetic clash: every closure, ratchet and GAP line in this
campaign addresses a defect BY NUMBER, so a status word written for one of them reads as
the other's, and `STATUSLESS_BACKLOG` / `UNFIXED_TRIAGE` key on the same string.

What it does NOT check: UPDATE headings. `### D164 UPDATE 1` is a legitimate second heading
for one defect and the campaign uses that form deliberately, so only a heading that OPENS a
defect -- `### D<n>.` or `### D<n> ` followed by anything that is not `UPDATE` -- counts.

Negative control is built in and runs every time (`--self-test` runs it alone): the real
pre-correction text, two entries both opening `### D165.`, must be caught.
"""
import re
import sys

OPEN = re.compile(r"^### D(\d+)(?:\.|\s)(?!\s*UPDATE\b)", re.M)


def duplicates(text):
    """D-numbers that more than one entry OPENS, with the count, lowest first."""
    seen = {}
    for m in OPEN.finditer(text):
        seen.setdefault(int(m.group(1)), []).append(m.start())
    return {n: len(v) for n, v in sorted(seen.items()) if len(v) > 1}


NEG = """### D164. something real
### D164 UPDATE 1. still the same defect, must NOT count
### D165. `DEPENDS_ON: none` is taken literally
### D165. `tt_bio.autograd.softmax` is a third softmax-backward site
### D166. fine
"""
POS = NEG.replace("### D165. `tt_bio", "### D166. `tt_bio").replace("### D166. fine",
                                                                    "### D167. fine")


def self_test():
    bad = duplicates(NEG)
    assert bad == {165: 2}, f"negative control did not catch the real collision: {bad}"
    assert duplicates(POS) == {}, f"positive control false-fired: {duplicates(POS)}"
    # the UPDATE form must survive both ways
    assert duplicates("### D1. a\n### D1 UPDATE 3. b\n") == {}, "UPDATE heading counted as an open"
    print("self-test OK: catches the real D165 collision, passes the corrected text, "
          "and does not count an UPDATE heading")


if __name__ == "__main__":
    self_test()
    if "--self-test" in sys.argv:
        raise SystemExit(0)
    path = next((a for a in sys.argv[1:] if not a.startswith("-")),
                "/home/moritz/.coworker/state/of3t/DEFECTS.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    bad = duplicates(text)
    n = len(OPEN.findall(text))
    if bad:
        for num, count in bad.items():
            print(f"FAIL: D{num} is opened by {count} separate entries in {path}")
        raise SystemExit(1)
    print(f"OK: {n} defect entries in {path}, every D-number opened exactly once")
