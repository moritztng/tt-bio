#!/usr/bin/env python3
"""Decide whether a fold A/B session measured ONE tree, judged on the code that ran.

The obvious check is that every leg records the same `git_head`, and it is wrong in both
directions. Session 3 of the APB A/B spans `59e2adced` and `a28ace13f` because the guard fix
of pass 21 was committed while the session folded, and that commit touches
`perf/c14_land/pair_channel_quiet.py` only -- it cannot reach a fold, because the worker
imports `tt_bio` and nothing else from the repo. A head-equality check discards that session.
Relaxing it to "ignore the heads" is worse: it would also accept a session whose `tt_bio`
changed underneath it, which is the failure this campaign actually hit (pass 18 caught a full
session measured on a tree 110 commits behind main).

So the check is: several heads are fine IFF `git diff <a> <b> -- tt_bio/` is empty. That is
the code the measured process executed. Everything else in the repo is harness, artifacts and
docs, none of which the worker imports.

Limit of the evidence, stated so it is not over-read: this compares committed trees. It cannot
see an uncommitted edit to `tt_bio/` made mid-session. `apb_fold_ab.py` already asserts that
`tt_bio` resolves inside the worktree; a dirty-tree check belongs there, not here.
"""
import subprocess
import sys

CODE = "tt_bio/"


def tt_bio_delta(repo: str, a: str, b: str) -> str:
    """Names of files under tt_bio/ that differ between two commits. '' means identical."""
    r = subprocess.run(["git", "-C", repo, "diff", "--name-only", a, b, "--", CODE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git diff %s..%s failed: %s" % (a[:9], b[:9], r.stderr.strip()))
    return r.stdout.strip()


def check(heads, repo="."):
    """(ok, [lines]) for the distinct git_heads a session recorded, in order of first sight."""
    seen = []
    for h in heads:
        if h not in seen:
            seen.append(h)
    if not seen or seen == [None]:
        return False, ["no git_head recorded on any leg -- provenance unknowable, do not score"]
    if None in seen:
        return False, ["some legs carry no git_head: %r" % (seen,)]
    if len(seen) == 1:
        return True, ["one tree across every leg: %s" % seen[0][:9]]
    lines = ["%d distinct heads: %s" % (len(seen), " ".join(h[:9] for h in seen))]
    ok = True
    ref = seen[0]
    for h in seen[1:]:
        delta = tt_bio_delta(repo, ref, h)
        if delta:
            ok = False
            lines.append("  %s..%s CHANGES %s: %s"
                         % (ref[:9], h[:9], CODE, delta.replace("\n", " ")))
        else:
            lines.append("  %s..%s leaves %s byte-identical" % (ref[:9], h[:9], CODE))
    lines.append("PROVENANCE OK -- every leg ran the same tt_bio" if ok else
                 "PROVENANCE FAIL -- the measured code changed mid-session, do not score")
    return ok, lines


def heads_of(session: dict):
    return [(b.get("result") or {}).get("env", {}).get("git_head") for b in session["blocks"]
            if b.get("returncode") == 0 and b.get("result")]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    ok, lines = check(heads_of(json.loads(Path(a.session).read_text())), a.repo)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)
