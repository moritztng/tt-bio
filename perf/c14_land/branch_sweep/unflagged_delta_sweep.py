#!/usr/bin/env python3
"""The other half: a branch can change shipped behaviour without touching an env_flag.

defaultsweep2 proves no unshipped branch proposes a FLAG default change. That covers this
campaign's levers, which are flag-gated by its own one-lever-one-flag discipline, but it does not
cover a changed constant, a new unconditional fast path or a dispatch-table row. Those show up as
a tt_bio/ delta against the branch's own merge-base.

A purely ADDITIVE delta (insertions only) is almost always a new gated path or a new helper: it
cannot change what today's default execution does unless something calls it, and nothing on main
does. A delta with DELETIONS modifies or removes shipped lines, which is where an unflagged
behaviour change lives. So rank by deletions and name every branch with any.
"""
import subprocess
from pathlib import Path

REPO = "/home/ttuser/.coworker/wt/c14-land-tail"


def git(*a, timeout=60):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                          text=True, timeout=timeout).stdout


rows = []
for b in Path("/tmp/unshipped_branches.txt").read_text().split():
    ref = "origin/" + b
    num = git("diff", "--numstat", f"origin/main...{ref}", "--", "tt_bio/")
    ins = dele = files = 0
    for line in num.splitlines():
        p = line.split("\t")
        if len(p) == 3 and p[0].isdigit() and p[1].isdigit():
            ins += int(p[0]); dele += int(p[1]); files += 1
    if files:
        rows.append((dele, ins, files, b))

rows.sort(reverse=True)
withdel = [r for r in rows if r[0] > 0]
print(f"unshipped branches with any tt_bio/ delta vs their merge-base : {len(rows)}")
print(f"  of those, purely ADDITIVE (zero deletions)                  : {len(rows) - len(withdel)}")
print(f"  of those, containing DELETIONS (can change shipped lines)   : {len(withdel)}")
print()
print(f"{'deletions':>10}{'insertions':>12}{'files':>7}  branch")
for dele, ins, files, b in withdel[:25]:
    print(f"{dele:>10}{ins:>12}{files:>7}  {b}")
