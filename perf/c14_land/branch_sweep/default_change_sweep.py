#!/usr/bin/env python3
"""Which unshipped branches PROPOSE a default change, as opposed to merely predating main's?

The first version of this compared each branch's env_flag defaults against origin/main's TIP and
got a wall of `TT_BIO_DIT_COND_HOIST: main=True -> branch=False`. That is not 30 branches asking to
turn cond-hoist off. It is 30 branches that were cut before main flipped it at 15f1ab0ac, and the
comparison could not tell "proposes a change" from "is old". Same shape as
`merge-must-verify-local-main-matches-origin-before-merging`: a contrast against the wrong
reference reads as a finding.

The right reference is each branch's OWN merge-base with main. A branch that proposes a default
change differs from the tree it was cut from; a stale branch does not.
"""
import re
import subprocess
from functools import lru_cache
from pathlib import Path

REPO = "/home/ttuser/.coworker/wt/c14-land-tail"
PAT = re.compile(r'env_flag\(\s*"([A-Z_0-9]+)"\s*,\s*(True|False)\s*\)')
FILES = ["tt_bio/tenstorrent.py", "tt_bio/boltz2.py", "tt_bio/trimul_tail.py",
         "tt_bio/sdpa_generic.py", "tt_bio/mm_generic.py", "tt_bio/triatt_sdpa.py"]


def git(*a, timeout=60):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True,
                          text=True, timeout=timeout).stdout


@lru_cache(maxsize=None)
def flags(ref):
    out = {}
    for f in FILES:
        for name, val in PAT.findall(git("show", f"{ref}:{f}")):
            out[name] = val
    return tuple(sorted(out.items()))


main_tip = dict(flags("origin/main"))
print(f"origin/main tip: {len(main_tip)} env_flag defaults\n")

proposals, stale, unreadable = [], 0, 0
for b in Path("/tmp/unshipped_branches.txt").read_text().split():
    ref = "origin/" + b
    base = git("merge-base", "origin/main", ref).strip()
    if not base:
        unreadable += 1
        continue
    bf, basef = dict(flags(ref)), dict(flags(base))
    if not bf:
        unreadable += 1
        continue
    changed = {k: (basef[k], v) for k, v in bf.items()
               if k in basef and basef[k] != v}
    added_on = sorted(k for k, v in bf.items() if k not in basef and v == "True")
    added_off = sorted(k for k, v in bf.items() if k not in basef and v == "False")
    if changed or added_on:
        proposals.append((b, changed, added_on, added_off))
    else:
        stale += 1

print("BRANCHES THAT PROPOSE A DEFAULT CHANGE (against their own merge-base)")
print("=" * 72)
if not proposals:
    print("  none")
for b, changed, added_on, added_off in proposals:
    print(f"{b}")
    for k, (base_v, v) in changed.items():
        live = main_tip.get(k)
        note = "ALREADY ON MAIN" if live == v else f"main now {live}"
        print(f"    FLIPS   {k}: {base_v} -> {v}   [{note}]")
    for k in added_on:
        print(f"    ADDS ON {k}   [{'on main' if k in main_tip else 'NOT on main'}]")

print(f"\nproposing a default change : {len(proposals)}")
print(f"no default change at all   : {stale}")
print(f"unreadable                 : {unreadable}")
