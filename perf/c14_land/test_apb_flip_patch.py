#!/usr/bin/env python3
"""Device-free proof that `apb_flip.patch` still applies and still does exactly one thing.

WHY THIS EXISTS. The patch is the whole point of not flipping the default while ask 9441 is open:
the decision costs one command instead of a pass. But it is a frozen diff against a moving branch,
and this row may wait days for an answer. A patch that has silently rotted is worse than no patch,
because it will be run by someone who believes it works -- the same shape as a helper pathed into a
worktree that no longer exists.

So the patch gets a test, and the test asserts the OUTCOME rather than the diff text: a patch that
still applies but flips the wrong thing would pass a `--check` and fail here.

Runs `git apply` in a scratch worktree, never in the live one, so a failure cannot leave the tree
dirty for whatever runs next.

    python3 perf/c14_land/test_apb_flip_patch.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATCH = Path(__file__).resolve().parent / "apb_flip.patch"
FLAG = "TT_BIO_APB_CONCAT_HEADS"
ENV_FLAG = re.compile(r'env_flag\(\s*"([A-Z_0-9]+)"\s*,\s*(True|False)\s*\)')


def git(*a, cwd=REPO, check=True):
    r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
    return r


def defaults(text):
    return dict(ENV_FLAG.findall(text))


def main() -> int:
    fails = []
    if not PATCH.exists():
        print(f"FAIL patch missing: {PATCH}")
        return 1

    # 1. It applies to the live tree as git sees it right now.
    if git("apply", "--check", str(PATCH), check=False).returncode:
        fails.append("1 `git apply --check` refuses the patch against the current tree -- it has "
                     "rotted, regenerate it before anyone runs it")
        for f in fails:
            print("FAIL", f)
        return 1

    # 2. Apply it in a scratch copy and check the OUTCOME, not the diff text.
    tmp = Path(tempfile.mkdtemp(prefix="apbflip-"))
    try:
        work = tmp / "w"
        # A worktree of HEAD, so the scratch copy is exactly what the patch was cut against.
        git("worktree", "add", "--detach", "-q", str(work), "HEAD")
        try:
            tt = work / "tt_bio/tenstorrent.py"
            before = defaults(tt.read_text())
            if before.get(FLAG) != "False":
                fails.append(f"2 precondition: {FLAG} is {before.get(FLAG)} at HEAD, expected False "
                             "-- either the default already moved or the flag was renamed")
            else:
                r = subprocess.run(["git", "-C", str(work), "apply", str(PATCH)],
                                   capture_output=True, text=True)
                if r.returncode:
                    fails.append(f"3 apply in scratch worktree failed: {r.stderr.strip()}")
                else:
                    after = defaults(tt.read_text())
                    if after.get(FLAG) != "True":
                        fails.append(f"4 after the patch {FLAG} is {after.get(FLAG)}, expected True")
                    moved = {k: (before.get(k), v) for k, v in after.items()
                             if k != FLAG and before.get(k) != v}
                    if moved:
                        fails.append(f"5 the patch moved OTHER defaults, which it must never do: "
                                     f"{moved}")
                    if set(after) - set(before):
                        fails.append(f"6 the patch added flags: {sorted(set(after) - set(before))}")
                    doc = (work / "docs/tuning-flags.md").read_text()
                    if f"## `{FLAG}` — on" not in doc:
                        fails.append("7 the docs heading did not move to `on`; code and docs would "
                                     "ship disagreeing about the default")
                    if "It is off because" in doc.split(f"## `{FLAG}`")[1].split("\n## ")[0]:
                        fails.append("8 the docs entry still carries its 'why it is off' sentence "
                                     "after a flip to on")
        finally:
            git("worktree", "remove", "--force", str(work), check=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        git("worktree", "prune", check=False)

    for f in fails:
        print("FAIL", f)
    print(f"{8 - len(fails)} checks passed, {len(fails)} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
