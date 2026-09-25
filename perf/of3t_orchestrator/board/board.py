#!/usr/bin/env python3
"""Print the OF3T board from git and the filesystem, so no pass has to trust the last one's prose.

K51 and K57 are the same defect twice, four passes apart, both mine: the state doc's BRANCH field
said five branches were "deliberately held, unmerged" and `merge-base --is-ancestor` said three of
them were fully in main. The first time it sent a row to measure on a release-gated branch for a
reason that had expired; the second it kept a closed user-facing defect (D205) open on the board.

Both were re-derivable in under a second. The countermeasure is not another reminder to check --
it is that the checking is one command, so a pass that prints a branch list has no excuse for
transcribing one.

    python3 perf/of3t_orchestrator/board/board.py [--repo PATH]

Reads only. Never writes, never fetches -- a fetch would make the output depend on the network and
this is meant to be run mid-pass without surprises. Run `git fetch origin` yourself first if you
want it fresh, and the output says how stale its refs are so you know whether you did.
"""
from __future__ import annotations
import argparse, subprocess, time
from pathlib import Path

COWORKER = Path("/home/moritz/.coworker")


def git(repo: Path, *a: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
    return r.stdout.strip()


def ok(repo: Path, *a: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), *a],
                          capture_output=True, text=True).returncode == 0


def branches(repo: Path) -> list[dict]:
    names = [l.strip() for l in
             git(repo, "for-each-ref", "--format=%(refname:short)",
                 "refs/remotes/origin/wk/of3t-*").splitlines() if l.strip()]
    out = []
    for full in names:
        b = full[len("origin/"):]
        counts = git(repo, "rev-list", "--left-right", "--count", f"{full}...origin/main").split()
        ahead, behind = (counts + ["?", "?"])[:2]
        # `0 ahead` is conclusive on its own -- every commit is an ancestor of main -- but the
        # ancestor test is what the lesson is about, so it is the field that gets printed.
        merged = ok(repo, "merge-base", "--is-ancestor", full, "origin/main")
        files = [f for f in git(repo, "diff", "--name-only",
                                f"origin/main...{full}").splitlines() if f]
        out.append({
            "branch": b, "ahead": ahead, "behind": behind, "merged": merged,
            "engine": sum(1 for f in files if f.startswith("tt_bio/")),
            "files": len(files),
            # Textual only, and the docstring of every caller should say so: land-d264 showed a
            # merge error in autograd.py is invisible to inference and visible only to a taped arm.
            "clean": ok(repo, "merge-tree", "--write-tree", "origin/main", full),
        })
    return out


def rows() -> list[dict]:
    """The ACTIVE board: queued, live, or concluded-but-still-queued. Not a campaign history.

    A row that concluded and had its TASKS tag closed is gone from queue.tsv and is deliberately
    absent here -- 150 of those would bury the three that can still move, which is the same
    failure the --all default avoids for branches.
    """
    live = subprocess.run(["pgrep", "-af", r"worker\.sh of3t-"],
                          capture_output=True, text=True).stdout
    live_slugs = {l.split("worker.sh ")[1].split()[0]
                  for l in live.splitlines() if "worker.sh " in l}
    queued = {}
    q = COWORKER / "workstreams" / "queue.tsv"
    if q.is_file():
        for line in q.read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 3 and p[0].startswith("of3t-"):
                queued[p[0]] = f"{p[1]}/{p[2]}"
    done = {p.name for p in (COWORKER / "state" / "concluded").glob("of3t-*")}
    slugs = sorted(set(queued) | live_slugs | {s for s in done if s in queued})
    return [{"row": s,
             "state": "LIVE" if s in live_slugs else ("concluded" if s in done else "queued"),
             "where": queued.get(s, "-")} for s in slugs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(COWORKER / "wt" / "of3t-orchestrator"))
    # The campaign has ~150 of3t branches and all but a handful are finished history. Printing
    # them all buries the two that matter, which is how a wall of output becomes as useless as no
    # output. Default to the unmerged ones; --all exists for an audit.
    ap.add_argument("--all", action="store_true",
                    help="include branches already fully in main (default: only unmerged)")
    a = ap.parse_args()
    repo = Path(a.repo)

    age = git(repo, "log", "-1", "--format=%ct", "origin/main")
    stale = (time.time() - int(age)) / 3600 if age.isdigit() else None
    print(f"origin/main {git(repo, 'rev-parse', '--short', 'origin/main')}"
          + (f", newest commit {stale:.1f} h old -- `git fetch origin` if that looks wrong"
             if stale is not None else ""))

    bs = branches(repo)
    shown = bs if a.all else [b for b in bs if not b["merged"]]
    hidden = len(bs) - len(shown)
    print(f"\nBRANCHES with commits main lacks (derived, not transcribed)"
          + (f" -- {hidden} more are fully in main, --all to list" if hidden else ""))
    print(f"  {'branch':26s} {'ahead':>5s} {'behind':>6s}  {'in main':8s} {'engine':>6s} "
          f"{'files':>5s}  merge(textual)")
    for b in shown:
        print(f"  {b['branch']:26s} {b['ahead']:>5s} {b['behind']:>6s}  "
              f"{'YES' if b['merged'] else 'no':8s} {b['engine']:>6d} {b['files']:>5d}  "
              f"{'clean' if b['clean'] else 'CONFLICT'}")
    print("  `in main` = merge-base --is-ancestor. A branch with YES is finished history: it is"
          "\n  not held, whatever any document says about it.")

    print("\nROWS")
    for r in rows():
        print(f"  {r['row']:26s} {r['state']:10s} {r['where']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
