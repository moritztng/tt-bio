#!/usr/bin/env python3
"""Re-verify TRIX's shipped claim against the repository. CPU only, no device, no network by default.

`TOTAL:` in state/trix-orchestrator.md says 0.222 s. That field is only allowed to move when a lever
is on `origin/main` AND on by default, because this fleet has twice recorded a "landed" win that was
either unmerged or merged behind a flag that defaults off (`concluded-marker-ship-verdict-not-merged`,
`merged-lever-defaults-off-is-not-a-landed-win`). Prose cannot establish either half, so this script
reads the repository.

It checks the DEPLOYED artifact -- the file content at `origin/main` -- rather than the diff that was
merged, because a later commit can revert a default without touching the merge
(`verify-the-deployed-artifact-not-your-own-change`).

    python3 perf/trix_ledger/verify_shipped.py            # check against the local origin/main ref
    python3 perf/trix_ledger/verify_shipped.py --fetch     # git fetch first

Exit 0 if the shipped claim holds, 1 if any part of it does not.
"""
import argparse
import subprocess
import sys

REF = "origin/main"

# The two commits that carry `trix-layout`'s lever: the code and its measurement.
COMMITS = {
    "121cb8a2a": "the lever (tt_bio/ +62/-9)",
    "1ec1c306d": "the fold A/B that measured 1.0211x = 0.222 s",
}

# Flags that must be literally `True` in the file at REF, not merely present in some diff.
# A flag read through `env_flag(name, DEFAULT)` is only on by default if DEFAULT is True.
FLAG_DEFAULTS = {
    "tt_bio/tenstorrent.py": {
        "TRIMUL_BACK_ONE_PASS_L1": "True",
        "TRIMUL_GATED_MOVE_L1": "True",
    },
}

# A default that is True but never read is not a landed lever either, so pin the callsites.
CALLSITES = {
    "tt_bio/tenstorrent.py": ["_TRIMUL_GATED_MOVE_L1", "_TRIMUL_BACK_ONE_PASS_L1"],
    # `eligible_back`'s L1-source guard: 256 measured a LOSS (0.9918x), so the lever is only
    # correct if this rejection exists. Its absence would mean shipping a known-negative shape.
    "tt_bio/reblock_permute.py": ["back_l1_src_narrow"],
}


def git(*args):
    return subprocess.run(("git",) + args, capture_output=True, text=True)


def show(path):
    r = git("show", f"{REF}:{path}")
    if r.returncode:
        return None
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="git fetch origin before checking")
    a = ap.parse_args()
    if a.fetch:
        git("fetch", "origin", "--quiet")

    head = git("log", "-1", "--format=%h %ad %s", "--date=short", REF).stdout.strip()
    if not head:
        print(f"FAIL: cannot resolve {REF}")
        return 1
    print(f"{REF} = {head}\n")

    bad = []

    print("merged into origin/main:")
    for sha, what in COMMITS.items():
        ok = git("merge-base", "--is-ancestor", sha, REF).returncode == 0
        print(f"  [{'ok' if ok else 'FAIL'}] {sha}  {what}")
        if not ok:
            bad.append(f"{sha} ({what}) is not an ancestor of {REF}")

    print("\ndefaults in the file at origin/main:")
    for path, flags in FLAG_DEFAULTS.items():
        src = show(path)
        if src is None:
            bad.append(f"{path} missing at {REF}")
            print(f"  [FAIL] {path} missing")
            continue
        for name, want in flags.items():
            hit = [l for l in src.splitlines() if l.startswith(f"{name} = ")]
            got = hit[0].split("=", 1)[1].strip() if hit else None
            ok = got == want
            print(f"  [{'ok' if ok else 'FAIL'}] {path}:{name} = {got} (want {want})")
            if not ok:
                bad.append(f"{path}:{name} is {got!r}, not {want!r} -- default is not on")

    print("\nthe defaults are actually read:")
    for path, names in CALLSITES.items():
        src = show(path) or ""
        for name in names:
            n = src.count(name)
            ok = n > 0
            print(f"  [{'ok' if ok else 'FAIL'}] {path}: {name} x{n}")
            if not ok:
                bad.append(f"{path} does not mention {name} -- the lever is not wired in")

    print()
    if bad:
        print("SHIPPED CLAIM DOES NOT HOLD -- TOTAL: must go back to 0.0000 s:")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("SHIPPED: trix-layout is on origin/main and on by default.")
    print("TOTAL: 0.222 s = 1.0211x at 320 aa, 0.635 % A/A floor, AICLK 1343/1350/1350 MHz")
    print("       sampled during 99 samples, qb1 p150a. Ceiling for the whole module is 1.2425x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
