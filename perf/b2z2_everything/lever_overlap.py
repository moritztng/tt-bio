#!/usr/bin/env python3
"""Do two lever branches touch the same code? Answer it before quoting their product.

Host only, read-only, opens nothing. This campaign has charged one site twice four separate times:
a stack that multiplied three levers all attacking the same per-program constant (corrected 1.2507x
-> 1.2058x), a stacking brief that named a branch already containing another, and -- the one that
motivated this script -- `TT_BIO_ATOM_KEY_WINDOW` and `TT_BIO_ATOM_SHIFT_GATHER`, which are two
implementations of the same atom key gather, measured alone on the Blackhole fold by two rows that
never cited each other, and carried on the shipping ladder as if they added.

`git merge-base --is-ancestor` answers the easy half (does one branch already CONTAIN the other).
This answers the half that keeps biting: two independent branches that edit the same function.

    python3 perf/b2z2_orch/lever_overlap.py origin/wk/b2z2-akw-ship origin/wk/b2z2-layout-op-elision
    python3 perf/b2z2_orch/lever_overlap.py --base origin/main --path tt_bio/ A B C

Exit status is 1 when any pair overlaps, so it can gate a stacking brief.
"""
from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr.strip()}")
    return r.stdout


def touched(base: str, ref: str, path: str) -> tuple[set[str], dict[str, set[int]]]:
    """(hunk contexts, file -> changed line numbers on the branch side) for ref vs base."""
    out = git("diff", "-U0", f"{base}...{ref}", "--", path)
    ctx: set[str] = set()
    lines: dict[str, set[int]] = {}
    cur = ""
    for ln in out.splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        m = HUNK.match(ln)
        if m and cur:
            start, count, context = int(m.group(1)), int(m.group(2) or 1), m.group(3).strip()
            if context:
                ctx.add(context)
            lines.setdefault(cur, set()).update(range(start, start + max(count, 1)))
    return ctx, lines


def ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                          capture_output=True).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+", help="two or more branch refs to compare pairwise")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--path", default="tt_bio/", help="limit the diff to this path")
    a = ap.parse_args()
    if len(a.refs) < 2:
        raise SystemExit("give at least two refs")

    info = {r: touched(a.base, r, a.path) for r in a.refs}
    for r, (ctx, lines) in info.items():
        n = sum(len(v) for v in lines.values())
        print(f"{r}: {n} changed lines in {len(lines)} file(s), {len(ctx)} hunk context(s)")

    bad = 0
    for x, y in itertools.combinations(a.refs, 2):
        print(f"\n=== {x}  vs  {y}")
        if ancestor(x, y) or ancestor(y, x):
            print("  CONTAINS: one branch is an ancestor of the other -- their ratios are not "
                  "independent and the marginal one is the only thing you may claim.")
            bad += 1
            continue
        cx, lx = info[x]
        cy, ly = info[y]
        shared_ctx = sorted(cx & cy)
        shared_files = sorted(set(lx) & set(ly))
        overlap = {f: sorted(lx[f] & ly[f]) for f in shared_files if lx[f] & ly[f]}
        if not shared_ctx and not overlap:
            print("  DISJOINT by hunk context and by changed lines. Their ratios may be composed -- "
                  "but compose them and MEASURE the union; disjoint code is not additive cost.")
            continue
        bad += 1
        if overlap:
            print("  CONFIRMED SAME SITE -- overlapping changed lines on the branch side:")
            for f, ls in overlap.items():
                print(f"    {f}: {len(ls)} line(s), e.g. {ls[:8]}")
            print("  => do NOT multiply these ratios. Measure the pair, or pick one.")
        if shared_ctx:
            print("  COLLISION RISK -- both branches edit inside these, at class/def granularity:")
            for c in shared_ctx:
                print(f"    {c}")
            if not overlap:
                print("  => not proof: a class is big and two branches can edit different methods "
                      "of it. READ the two diffs before you compose them, and say in your doc which "
                      "you did. This is where the atom-gather duplicate hid.")

    print(f"\n{bad} of {len(list(itertools.combinations(a.refs, 2)))} pair(s) are not independent.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
