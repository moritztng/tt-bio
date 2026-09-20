#!/usr/bin/env python3
"""What commit every p300c size-ladder baseline entry was recorded at, and how far main has moved.

The 09-16 instance of this fix re-recorded two models because those were the two a flag's census
touched. That is the wrong denominator: the baseline goes stale per CARD ENTRY, not per flag, so
this prints all nine and lets the reader see which ones actually moved instead of assuming.

Two things it checks that a `recorded` date cannot tell you:

  ANCESTRY. An entry names the commit it was recorded AT, and that commit can live on a worker
  branch that never merged -- `boltz2` and `openbind` both name commits on `wk/c14-p300c-256-recell`,
  which is not an ancestor of main. The drift count is still the right number (it counts what is
  reachable from HEAD and not from there), but "78 commits behind" and "78 commits behind a commit
  that is not on main" are different provenance and the second one should be visible.

  tt_bio vs ALL. Only `tt_bio/` can move a lever census or a runtime. A baseline 459 commits
  behind on the whole repo and 104 behind on `tt_bio/` is 104 behind for this arm's purposes.

Usage:  provenance.py [--card p300c]
"""
import argparse
import glob
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="p300c")
    args = ap.parse_args()
    head = sh("git", "rev-parse", "HEAD")
    print(f"HEAD {head[:9]} {sh('git', 'log', '-1', '--format=%cI', head)}   "
          f"origin/main {sh('git', 'rev-parse', '--short', 'origin/main')}   card {args.card}\n")
    for f in sorted(glob.glob(str(ROOT / "docs/size_ladder_baseline.d/*.json"))):
        model = os.path.basename(f)[:-5]
        card = (json.load(open(f)).get("cards") or {}).get(args.card)
        if not card:
            print(f"{model:<14} no {args.card} entry")
            continue
        c = card.get("commit")
        # Against origin/main, not against HEAD. On a worker branch HEAD is the branch, so
        # "ancestor of HEAD" is true for every commit this branch made and answers nothing.
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", c, "origin/main"],
                             cwd=ROOT).returncode == 0
        # An entry recorded on a worker branch names a commit that is not on main, and the
        # question that actually matters is not whether the SHA is reachable but whether the
        # ENGINE it names differs from main's. A data-only branch records at its own HEAD and
        # the tt_bio content is main's, which is checkable rather than a claim.
        engine = "same-as-main" if anc else (
            "engine=main" if not sh("git", "diff", "--stat", f"{c}..origin/main", "--", "tt_bio/")
            else "ENGINE DIFFERS FROM MAIN")
        print(f"{model:<14} commit={c[:9]} recorded={card.get('recorded')} "
              f"host={card.get('host')} on-main={'yes' if anc else 'NO'} {engine} "
              f"behind: tt_bio={sh('git', 'rev-list', '--count', f'{c}..HEAD', '--', 'tt_bio/'):>4} "
              f"all={sh('git', 'rev-list', '--count', f'{c}..HEAD'):>4} "
              f"models={sorted((card.get('models') or {}))}")


if __name__ == "__main__":
    main()
