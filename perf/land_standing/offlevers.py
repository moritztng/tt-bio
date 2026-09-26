#!/usr/bin/env python3
"""Every default-off env lever across a tree, with its claimed number and its OWN verdict lines.

Widened from `tenstorrent.py` alone to every file given on argv, because the first census only
read one file and its conclusion ("the queue is worked through") was therefore a claim about one
file rather than about the engine.

It does not classify. An earlier version matched a fixed vocabulary of blocking phrases and
reported two levers as open whose own comments said "MEASURED, AND IT LOSES" and "THE OP WIN DOES
NOT REACH THE FOLD". A keyword classifier over prose written by many hands is always incomplete in
the direction that MAKES work, which is the expensive direction for a landing row. So it prints
the verdict-bearing lines and leaves the judgement to the reader, and says NO VERDICT LINE only as
a claim about the text.
"""
import pathlib
import re
import sys

FLAG = re.compile(r'env_flag\(\s*"([A-Z_0-9]+)"\s*,\s*False\s*\)')
NUM = re.compile(r"(\d+\.\d+x|\+\d+\.\d+ s|\d+\.\d+ ms|\d+\.\d+ %)")
VERDICT = re.compile(
    r"\b(off|stays off|default off|loses|slower|does not|never|not reach|no win|worth nothing|"
    r"hard stop|held|closed|gated|regression|depend|diagnostic|NO-GO|why this|so it|"
    r"until|blocked|unmeasured|re-?open|screen|dead-?end)\b", re.I)


# The scraper reads UPWARD only, and that is a known blind spot: RFD3_BLOCK_SPARSE states its
# reason ("per-target tuning", +3.787 in sample vs +3.744 out, 1.2 % overfit) in the module
# docstring and in comments BELOW the flag, so this file reported it as having no verdict. Treat
# a NO VERDICT LINE result as "look at the file", never as "this lever is untriaged".


def comment_below(lines, i, n=12):
    out = []
    for line in lines[i + 1:i + 1 + n]:
        t = line.lstrip()
        if t.startswith("#") or t.startswith("#:"):
            out.append(t.lstrip("#: ").rstrip())
    return out


def comment_above(lines, i):
    j, com = i - 1, []
    while j >= 0 and (lines[j].lstrip().startswith("#") or not lines[j].strip()):
        if lines[j].strip():
            com.append(lines[j].lstrip("# ").rstrip())
        j -= 1
        if len(com) > 80:
            break
    return list(reversed(com))


total, noverdict, withnum = 0, [], []
for path in sorted(sys.argv[1:]):
    lines = pathlib.Path(path).read_text(errors="replace").split("\n")
    hits = []
    for i, line in enumerate(lines):
        m = FLAG.search(line)
        if not m:
            continue
        com = comment_above(lines, i)
        nums = NUM.findall("\n".join(com))[:3]
        below = comment_below(lines, i)
        verdict = [c for c in com + below if VERDICT.search(c)][:1]
        nums = nums or NUM.findall(chr(10).join(below))[:3]
        hits.append((m.group(1), i + 1, len(com), nums, verdict))
    if not hits:
        continue
    print("\n=== %s ===" % path)
    for name, ln, nc, nums, verdict in hits:
        total += 1
        tag = "NO VERDICT LINE" if not verdict else verdict[0][:82]
        print("  %-34s L%-6d %2dc  %-26s %s"
              % (name, ln, nc, ",".join(nums) if nums else "-", tag))
        if nums:
            withnum.append((path, name, ln, nums, bool(verdict)))
        if not verdict:
            noverdict.append((path, name, ln, nc, nums))

print("\n\n%d default-off levers across %d files." % (total, len(sys.argv) - 1))
print("\nWith a NUMBER and NO verdict line -- the only genuinely untriaged shape:")
any_open = False
for path, name, ln, nums, hasv in withnum:
    if not hasv:
        any_open = True
        print("    %-34s %s:%d  %s" % (name, pathlib.Path(path).name, ln, ",".join(nums)))
if not any_open:
    print("    (none)")
print("\nNo verdict line at all (%d), most of which are plumbing with no number:" % len(noverdict))
for path, name, ln, nc, nums in noverdict:
    print("    %-34s %s:%-6d %2dc  %s"
          % (name, pathlib.Path(path).name, ln, nc, ",".join(nums) if nums else "no number"))
