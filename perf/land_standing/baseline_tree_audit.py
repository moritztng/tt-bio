#!/usr/bin/env python3
"""Why the size-ladder arm fails 148 lever rows, and why no branch can be the cause.

The D10+D24 size-ladder arm fails ~200 rows over eight of nine models. 148 of them are
two lever classes:

    TRIATT_SDPA_HIFI: new lever not in the baseline      (77 rows)
    SDPA_WIDE_K:      resolved 'False' -> 'True'         (71 rows)

Both are properties of main's recorded baseline corpus. The tempting reading is "the
baselines are old", and for TRIATT_SDPA_HIFI that is true: the lever landed 2026-09-21
and eight baselines were recorded on 09-20. For SDPA_WIDE_K it is FALSE and the dates
say so -- the default flipped ON on main at 49080031e, 2026-09-19 17:34, and all eight
failing baselines were committed AFTER that. A date-based reading gets this wrong.

The actual discriminator is a code fact in the tree each baseline was MEASURED on. A
cov-ladder row records a ladder on its own branch tree and merges the file to main
later, so the file's commit date can sit after a default flip the measurement never saw.
This script reads, per baseline: the value recorded in the file, the non-merge commit
that recorded it, and whether THAT COMMIT'S TREE contained the flip.

Expected, and what it prints: perfect separation at n=10. Every baseline whose recording
tree has _SDPA_WIDE_K_DEFAULT = False recorded False and now fails; the two whose tree
has True recorded True and pass. Run inside a tt-bio checkout.
"""
import glob
import json
import os
import re
import subprocess
import sys

# The commit that put TT_BIO_SDPA_WIDE_K on by default, and the merge that put it on main.
FLIP_COMMIT = "65b1c356e"
FLIP_ON_MAIN = "49080031e"


def git(*args):
    r = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    return r.stdout.strip()


def tree_has_flip_true(commit):
    """Did this commit's own tree default TT_BIO_SDPA_WIDE_K to True?"""
    src = git("show", commit + ":tt_bio/tenstorrent.py")
    m = re.search(r"_SDPA_WIDE_K_DEFAULT\s*=\s*(\w+)", src)
    return (m.group(1) if m else "<absent>")


def recorded_values(path):
    txt = open(path).read()
    wide = sorted(set(re.findall(
        r'"SDPA_WIDE_K"[^}]*?"resolved"\s*:\s*"?(\w+)"?', txt)))
    return (",".join(wide) if wide else "ABSENT"), txt.count("TRIATT_SDPA_HIFI")


def main():
    if not glob.glob("docs/size_ladder_baseline.d/*.json"):
        sys.exit("run inside a tt-bio checkout with docs/size_ladder_baseline.d/")
    print("scoring tree: %s" % git("rev-parse", "--short", "HEAD"))
    print("flip commit %s authored %s; reached main at %s %s" % (
        FLIP_COMMIT, git("show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M", FLIP_COMMIT),
        FLIP_ON_MAIN,
        git("show", "-s", "--format=%ad", "--date=format:%Y-%m-%d %H:%M", FLIP_ON_MAIN)))
    print()
    hdr = ("%-18s %-10s %-6s %-18s %-10s %-8s" %
           ("baseline", "recorded", "HIFI", "recorded_by", "rec_date", "tree_flip"))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for f in sorted(glob.glob("docs/size_ladder_baseline.d/*.json")):
        name = os.path.basename(f)[:-5]
        wide, hifi = recorded_values(f)
        commit = git("log", "--no-merges", "-1", "--format=%h", "--", f)
        date = git("log", "--no-merges", "-1", "--format=%ad",
                   "--date=format:%m-%d %H:%M", "--", f)
        flip = tree_has_flip_true(commit) if commit else "?"
        rows.append((name, wide, hifi, flip))
        print("%-18s %-10s %-6s %-18s %-10s %-8s" %
              (name, wide, (hifi or "-"), commit, date, flip))

    # The claim, asserted rather than eyeballed: recorded value tracks the recording
    # tree's default, with no exceptions.
    print()
    bad = [r for r in rows
           if (r[3] == "False" and r[1] != "False")
           or (r[3] == "True" and "True" not in r[1])]
    print("baselines whose recorded SDPA_WIDE_K disagrees with their recording tree's "
          "default: %d" % len(bad))
    for r in bad:
        print("   MISMATCH %s recorded=%s tree=%s" % (r[0], r[1], r[3]))
    pre = [r for r in rows if r[3] == "False"]
    post = [r for r in rows if r[3] == "True"]
    print("pre-flip recording trees: %d (%s)" % (len(pre), ", ".join(r[0] for r in pre)))
    print("post-flip recording trees: %d (%s)" % (len(post), ", ".join(r[0] for r in post)))
    print("baselines carrying TRIATT_SDPA_HIFI at all: %d" %
          len([r for r in rows if r[2]]))


if __name__ == "__main__":
    main()
