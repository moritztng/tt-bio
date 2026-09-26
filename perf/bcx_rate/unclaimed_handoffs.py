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
"""
import argparse
import re
import sys
import time
from pathlib import Path

COWORKER = Path("/home/moritz/.coworker")
CONCLUDED = COWORKER / "state" / "concluded"

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
