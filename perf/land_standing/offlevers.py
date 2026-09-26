#!/usr/bin/env python3
"""Every default-off env lever on main, with its claimed number and its OWN verdict lines.

Why this is not a classifier any more. The first version matched a fixed vocabulary of blocking
phrases ("hard stop", "held off", "accuracy regression", ...) and declared anything else
"no stated reason". It then reported `TT_BIO_TRIMUL_TAIL_L1` and `TT_BIO_TRIMUL_OUT_L1` as the
open queue, when their own comments say:

    MEASURED, AND IT LOSES, so it stays off ... Net 1.15-1.17x SLOWER than shipped
    AND THE OP WIN DOES NOT REACH THE FOLD, which is why this stays off

Both closed, in plain English the regex had no word for. A keyword classifier over prose written
by many hands is always going to be incomplete in the direction that MAKES work, which is the
expensive direction for a landing row.

So this prints the lever's own verdict-bearing lines and leaves the judgement to the reader. It
reports "NO VERDICT LINE" only when the comment contains nothing that looks like one, which is a
claim about the text rather than about the lever.
"""
import re
import sys

SRC = sys.argv[1]
lines = open(SRC).read().split("\n")

FLAG = re.compile(r'env_flag\("([A-Z_0-9]+)",\s*False\)')
NUM = re.compile(r"(\d+\.\d+x|\+\d+\.\d+ s|\d+\.\d+ ms|\d+\.\d+ %)")
# Broad: any comment line that reads like a disposition, however it is worded.
VERDICT = re.compile(
    r"\b(off|stays off|default off|loses|slower|does not|never|not reach|no win|worth nothing|"
    r"hard stop|held|closed|gated|regression|depend|diagnostic|NO-GO|why this|so it|"
    r"until|blocked|unmeasured|re-?open)\b", re.I)


def comment_above(i):
    j, com = i - 1, []
    while j >= 0 and (lines[j].lstrip().startswith("#") or not lines[j].strip()):
        if lines[j].strip():
            com.append(lines[j].lstrip("# ").rstrip())
        j -= 1
        if len(com) > 80:
            break
    return list(reversed(com))


rows = []
for i, line in enumerate(lines):
    m = FLAG.search(line)
    if not m:
        continue
    com = comment_above(i)
    rows.append((m.group(1), i + 1, com,
                 NUM.findall("\n".join(com))[:3],
                 [c for c in com if VERDICT.search(c)][:2]))

noverdict = []
for name, ln, com, nums, verdict in rows:
    print("\n%s  (line %d, %d comment lines)" % (name, ln, len(com)))
    print("   number(s): %s" % (", ".join(nums) if nums else "none claimed"))
    if verdict:
        for v in verdict:
            print("   verdict?   %s" % v[:110])
    else:
        print("   verdict?   NO VERDICT LINE IN ITS COMMENT")
        noverdict.append((name, ln, len(com), nums))

print("\n\n%d default-off levers. %d have no verdict-looking line at all:" % (len(rows), len(noverdict)))
for name, ln, nc, nums in noverdict:
    print("    %-32s line %-6d %2d cmt lines  %s"
          % (name, ln, nc, ", ".join(nums) if nums else "no number"))
print("\nA lever with a number and NO verdict line is the only shape that is genuinely untriaged.")
