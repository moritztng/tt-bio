#!/usr/bin/env python3
"""Every default-off env lever on main, with the number its own comment claims and the reason.

This row's audit concluded that its charter's named backlog is exhausted and that candidates have
to come from campaign handovers. That is only half true: main itself carries default-off levers
with measured numbers attached, and `merged-lever-defaults-off-is-not-a-landed-win` makes every
one of them a candidate until something says otherwise. `TT_BIO_SDPA_WIDE_K_UP` was found this
way and rejected on measurement in one pass.

For each flag this prints the number its comment claims and whether the comment already states a
blocking reason, so the row can triage instead of re-deriving. It reads main, not the worktree.
"""
import re
import sys

SRC = sys.argv[1]
lines = open(SRC).read().split("\n")

FLAG = re.compile(r'env_flag\("([A-Z_0-9]+)",\s*False\)')
# A claimed speed number: "1.234x", "+0.123 s", "123 ms", "12.3 %"
NUM = re.compile(r"(\d+\.\d+x|\+\d+\.\d+ s|\d+\.\d+ ms|\d+\.\d+ %)")
BLOCK = re.compile(r"hard stop|held (off|there)|accuracy regression|card-depend|diagnostic|"
                   r"NO-GO|release-gated|not bit-exact|Moritz", re.I)

rows = []
for i, line in enumerate(lines):
    m = FLAG.search(line)
    if not m:
        continue
    # Walk back over the contiguous comment block that documents this flag.
    j = i - 1
    com = []
    while j >= 0 and (lines[j].lstrip().startswith("#") or not lines[j].strip()):
        if lines[j].strip():
            com.append(lines[j])
        j -= 1
        if len(com) > 60:
            break
    com = "\n".join(reversed(com))
    nums = NUM.findall(com)
    blocks = sorted(set(b[0] if isinstance(b, tuple) else b for b in BLOCK.findall(com)))
    rows.append((m.group(1), i + 1, len(com.split("\n")) if com else 0,
                 nums[:3], blocks[:3]))

print("%-34s %6s %5s  %-28s %s" % ("flag", "line", "cmt", "claimed number(s)", "stated reason"))
for name, ln, nc, nums, blocks in rows:
    print("%-34s %6d %5d  %-28s %s"
          % (name, ln, nc, ",".join(nums) if nums else "-", ",".join(blocks) if blocks else "-"))

undocumented = [r for r in rows if r[2] < 3]
unexplained = [r for r in rows if r[3] and not r[4]]
print("\n%d default-off levers." % len(rows))
print("%d have a claimed number and NO stated blocking reason -- the triage queue:" % len(unexplained))
for name, ln, _, nums, _ in unexplained:
    print("    %-32s line %-6d %s" % (name, ln, ",".join(nums)))
print("%d carry fewer than 3 comment lines, so their default is undocumented:" % len(undocumented))
for name, ln, *_ in undocumented:
    print("    %-32s line %d" % (name, ln))
