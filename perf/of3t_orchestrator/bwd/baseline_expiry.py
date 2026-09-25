#!/usr/bin/env python3
"""Is a banked perf artifact still describing a tree anyone runs?

LEDGER R209: the OF3T campaign quoted a 466.702 s step for four days after a commit changed it
by ~65x. The artifact was honest -- it recorded its own `env.commit` -- but provenance only says
where a number CAME FROM, and nobody asked whether it had EXPIRED. A campaign that re-reads a
banked baseline every pass is exactly the one that never re-checks it.

The check is two lines of git and it would have fired the first time any row read the artifact
after 2026-09-23:

    an artifact is LIVE  if its env.commit is an ancestor of HEAD and nothing since then
                            touched the files its measurement depends on;
    an artifact is STALE if a commit between env.commit and HEAD touched those files.

Usage:
    baseline_expiry.py ARTIFACT.json [--paths tt_bio/autograd.py tt_bio/taped_ttnn.py ...]

Defaults to the paths a training-step timing depends on. Exit 1 if stale, so it can gate.
"""
import argparse, json, subprocess, sys
from pathlib import Path

DEFAULT_PATHS = ["tt_bio/autograd.py", "tt_bio/taped_ttnn.py", "tt_bio/tenstorrent.py",
                 "tt_bio/train/", "tt_bio/kernels/"]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--paths", nargs="*", default=DEFAULT_PATHS)
    ap.add_argument("--head", default="HEAD")
    a = ap.parse_args()

    d = json.loads(Path(a.artifact).read_text())
    env = d.get("env") or {}
    commit = env.get("commit") or d.get("commit")
    if not commit:
        print(f"UNDATABLE: {a.artifact} records no env.commit. A perf artifact without the "
              f"commit it ran on cannot be checked for expiry and should not be quoted.")
        return 1

    short = commit[:9]
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, a.head]).returncode != 0:
        print(f"STALE: {short} is not an ancestor of {a.head} -- the artifact ran on a tree that "
              f"is not in this history at all.")
        return 1

    since = [l for l in git("log", "--oneline", f"{commit}..{a.head}", "--", *a.paths).split("\n") if l]
    if since:
        print(f"STALE: {len(since)} commit(s) touched the measured paths between {short} and "
              f"{a.head}. Re-take before quoting. Most recent first:")
        for l in since[:10]:
            print("   ", l)
        if len(since) > 10:
            print(f"    ... and {len(since)-10} more")
        return 1

    print(f"LIVE: {short} is an ancestor of {a.head} and nothing since touched the measured paths.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
