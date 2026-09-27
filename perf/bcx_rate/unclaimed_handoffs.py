#!/usr/bin/env python3
"""Concluded rows that handed work to a named owner, and whether anyone picked it up.

Why this exists. Three times on 2026-09-26 a BCX row concluded, named an owner or a set of pages
in its closing text, and nothing delivered it:

  bcx-accept    left three live qb1 arms behind a `[x]` checkbox; a concluded row cannot collect
                its own arms and a checked task has no queue row, so ~20 card-hours of trajectories
                sat unread until an orchestrator pass happened to look.
  bcx-mainstep  named NUMBERS.md, COMPETITIVE.md, RATE.md and CMP-ANSWER.md as "the orchestrator's".
                Six hours later none of the four carried its numbers and all four still carried the
                superseded ones -- the campaign's published gap was ~2x too pessimistic that whole
                time.
  bcx-shipped   left two running reference arms with "they should be assigned a reader so they
                don't go dark". Nobody was assigned. Those arms hold NO card, so not even a
                cardblock auto-clearing signals that a result is ready.

The fleet auto-returns an orphan arm's CARD and has no mechanism at all for its RESULT. This is
the missing half, and it is deliberately a REPORTER, not a guard: it prints candidates for a human
or an orchestrator pass to read. It does not decide that something is unclaimed, because the only
honest test of that is whether a person looked.

  python3 perf/bcx_rate/unclaimed_handoffs.py                 # BCX rows concluded in the last 24h
  python3 perf/bcx_rate/unclaimed_handoffs.py --hours 72 --prefix of3t

The second half of the report has no time window, and that is the correction. The prose scan
above only looks back --hours, which is backwards for the case it was built for: an unclaimed
handoff gets MORE urgent as it ages, and this tool's own 24 h window hid the worst one it had.
`bcx-exactflip` concluded 2026-09-26 17:16 handing its merge to `land-standing`, and 29 h later
`wk/bcx-exactflip` was still not on main, `land-standing.md` had never named it, and main still
shipped the slow `exact=True` default on the morning the competition window opened. A branch
that is not an ancestor of origin/main is a fact, not prose, so it is checked at every age.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

COWORKER = Path("/home/moritz/.coworker")
CONCLUDED = COWORKER / "state" / "concluded"
REPO = Path(__file__).resolve().parents[2]

#: A branch a marker names as carrying its work. Local `wk/` branches only: a marker quoting
#: `origin/main` or a bare sha is not claiming to own an unlanded branch.
BRANCH = re.compile(r"\bwk/([a-z0-9][a-z0-9._-]*[a-z0-9])\b")

#: Phrases a row uses when it is handing something on rather than finishing it. Deliberately
#: narrow: a row that says "I flipped nothing -- that is land-standing's" is handing over, a row
#: that merely mentions another row is not.
HANDOFF = re.compile(
    r"(?i)("
    r"left to whoever|left to the|left for the|should be assigned|assign(?:ed)? a reader|"
    r"pages? (?:that )?(?:have to|must) change|not edited by (?:me|this row)|"
    r"(?:is|are) the orchestrator'?s?\b|that is [`a-z0-9-]+'s\b|"
    r"owe[sd]? to|hand(?:ed|s)? (?:it |them |this )?(?:to|off to)|"
    r"do not (?:re-?run|duplicate)|needs a reader|so (?:they|it) (?:do|does) ?n[o']t go dark"
    r")")

#: Files a handoff commonly names. Used only to date-check, never to decide.
NAMED = re.compile(r"`?(state/[a-z0-9/_.-]+\.md|[A-Z][A-Z0-9-]+\.md)`?")


def named_paths(text):
    out = []
    for m in NAMED.findall(text):
        p = COWORKER / m if m.startswith("state/") else COWORKER / "state" / "bcx" / m
        if p.exists():
            out.append(p)
    return sorted(set(out))


def git(*a):
    r = subprocess.run(("git", "-C", str(REPO)) + a, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def unlanded_branches(prefix):
    """Branches named by a concluded marker that are not ancestors of origin/main.

    Age-independent on purpose. Returns [(branch, tip, marker_name, concluded_at)], oldest
    conclusion first, so the longest-unclaimed merge reads at the top.
    """
    rc, _ = git("rev-parse", "--verify", "--quiet", "origin/main")
    if rc:
        return None
    out, seen = [], set()
    for marker in sorted(CONCLUDED.glob("%s*" % prefix), key=lambda p: p.stat().st_mtime):
        if not marker.is_file():
            continue
        for name in dict.fromkeys(BRANCH.findall(marker.read_text(errors="replace"))):
            branch = "wk/%s" % name
            if branch in seen:
                continue
            rc, tip = git("rev-parse", "--verify", "--quiet", "origin/%s" % branch)
            if rc:
                continue          # branch gone: deleted after landing, or never pushed
            seen.add(branch)
            if git("merge-base", "--is-ancestor", tip, "origin/main")[0] == 0:
                continue
            # Not on main is not the same as unclaimed. A sub-row's branch is normally taken up
            # by an aggregator (every bcx-p10-* feeds wk/bcx-perf10), and a tip contained in
            # another live branch has a carrier, whatever main says. Only a tip nothing else
            # contains is actually sitting with no route.
            carriers = [b.strip().replace("origin/", "", 1)
                        for b in git("branch", "-r", "--contains", tip)[1].splitlines()
                        if b.strip() and "->" not in b and b.strip() != "origin/%s" % branch]
            if carriers:
                continue
            out.append((branch, tip[:9], marker.name, marker.stat().st_mtime))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--prefix", default="bcx", help="slug prefix, or '' for every row")
    args = ap.parse_args(argv)

    if not CONCLUDED.is_dir():
        print("no %s" % CONCLUDED, file=sys.stderr)
        return 2

    cutoff = time.time() - args.hours * 3600
    flagged = 0
    for marker in sorted(CONCLUDED.glob("%s*" % args.prefix), key=lambda p: p.stat().st_mtime):
        if not marker.is_file() or marker.stat().st_mtime < cutoff:
            continue
        text = marker.read_text(errors="replace")
        hits = {m.group(0).lower() for m in HANDOFF.finditer(text)}
        if not hits:
            continue
        flagged += 1
        concluded_at = marker.stat().st_mtime
        print("\n%s  concluded %s" % (
            marker.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(concluded_at))))
        print("   handoff language: %s" % ", ".join(sorted(hits)[:4]))
        paths = named_paths(text)
        if not paths:
            print("   names no state page -- read the marker itself")
        own = {COWORKER / "state" / ("%s.md" % marker.name.split(".")[0])}
        for p in paths:
            touched = p.stat().st_mtime
            if p in own:
                mark = "(the row's own state doc -- ignore)"
            elif touched > concluded_at:
                mark = "TOUCHED AFTER"
            else:
                mark = "**NOT touched since**"
            print("   %-58s %s (%s)" % (
                p.relative_to(COWORKER), mark,
                time.strftime("%m-%d %H:%M", time.localtime(touched))))

    print("\n%d concluded row(s) in the last %gh carry handoff language." % (flagged, args.hours))
    print("'NOT touched since' is a CANDIDATE, not a verdict: a page can be correct already, and a")
    print("handoff can be to a row rather than to a page. Read the marker before acting.")

    unlanded = unlanded_branches(args.prefix)
    if unlanded is None:
        print("\nno origin/main in %s -- branch check skipped" % REPO)
        return 0
    print("\nBranches named by a concluded %s* row that are NOT on origin/main (every age):"
          % (args.prefix or "any"))
    if not unlanded:
        print("   none")
        return 0
    now = time.time()
    for branch, tip, marker, at in unlanded:
        print("   %-26s %s  unlanded %5.1f h  (%s)" % (branch, tip, (now - at) / 3600.0, marker))
    print("Held deliberately (release-gated, superseded) or dropped -- the marker says which.")
    print("Age is the signal: a gate nobody has opened in a day is usually a gate nobody owns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
