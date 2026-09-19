#!/usr/bin/env python3
"""Has main moved underneath a running fold A/B in a way that can reach THIS fold?

Pass 18 threw away a finished 12-block session because it had been measured on a tree 110
commits behind main, and the campaign's own rule is that a lever measured underneath a
different set of shipped levers is not a landable number. That rule is right and it is also
expensive, so it should be applied to the code that can execute rather than to the commit
count. A commit to main that no fold of this model at this size can reach does not invalidate
a session, and saying so needs evidence, not an argument from file names.

Two screens, in order of strength:

  1. FILES. `git diff <session tree> origin/main -- tt_bio/`. Empty means there is nothing to
     decide -- the measured code IS current main's.
  2. DECLINES. If tt_bio/ did move, the only class of change this screen can clear is a change
     that fills a lookup table. A fold that DECLINES zero lookups on a path cannot be changed
     by adding entries to that path's table, whatever the entries are. So this reads the
     declined counters out of a committed firing census for the same model and size.

Screen 2 is deliberately narrow. It clears "keys were added to a table this fold never misses"
and NOTHING else. Any other diff under tt_bio/ comes back UNCLEARED and the session is stale,
which is the conservative direction.
"""
import argparse
import json
import subprocess
from pathlib import Path

# Counters whose [served, declined] pair covers the _MM_BLOCK lookups on the fused qkv path.
QKV_COUNTERS = ("triatt_qkv.QKVG_STATS", "triatt_qkv.QKVGB_STATS", "triatt_qkv.STATS")


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="the A/B session json")
    ap.add_argument("--against", default="origin/main")
    ap.add_argument("--census", default="perf/pvx_eligibility/out/b2512.json",
                    help="committed firing census for the SAME model and size")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()

    d = json.loads(Path(a.session).read_text())
    heads = sorted({(b.get("result") or {}).get("env", {}).get("git_head")
                    for b in d["blocks"] if b.get("result")} - {None})
    print("session heads: %s" % " ".join(h[:9] for h in heads))
    target = sh("git", "-C", a.repo, "rev-parse", a.against)
    print("against %s = %s" % (a.against, target[:9]))

    # Compare from the MERGE BASE, not from the session head. A diff head..main also shows
    # this branch's own production delta (the lever under test, and MM_SHORT_M_BW which is off)
    # as if main had added it. What matters is only what main gained since the session forked.
    bases = sorted({sh("git", "-C", a.repo, "merge-base", h, target) for h in heads})
    print("merge base(s): %s" % " ".join(b[:9] for b in bases))
    moved = set()
    for b in bases:
        out = sh("git", "-C", a.repo, "diff", "--name-only", b, target, "--", "tt_bio/")
        moved |= set(out.split()) if out else set()
    if not moved:
        print("SCREEN 1 CLEAR: %s added nothing under tt_bio/ since the session forked"
              % a.against)
        return 0
    print("SCREEN 1: tt_bio/ moved in %s" % " ".join(sorted(moved)))

    added = [l[1:].strip() for b in bases
             for l in sh("git", "-C", a.repo, "diff", "-U0", b, target, "--", "tt_bio/").splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    body = [l for l in added if l and not l.startswith("#")]
    table_only = all(l.startswith("(") and "):" in l for l in body)
    print("SCREEN 2: %d added source lines, %d non-comment; table-entry shape: %s"
          % (len(added), len(body), table_only))
    if not table_only:
        print("UNCLEARED: the delta is not table entries. Treat the session as stale.")
        return 1

    # Read the census out of the target ref: it ships with the commits being screened and
    # need not exist in the worktree, which is by construction behind them.
    cen = json.loads(sh("git", "-C", a.repo, "show", "%s:%s" % (target, a.census)))
    counters = cen["folds"][0]["counters"]
    tot = 0
    for name in QKV_COUNTERS:
        served, declined = counters[name]
        print("  %-26s served %4d  declined %4d" % (name, served, declined))
        tot += declined
    print("census: %s at %s aa, %s sampling steps / %s recycles, clock %s"
          % (cen["model"], cen["size"], cen["sampling_steps"], cen["recycling_steps"],
             cen["folds"][0]["clock_line"].split("CLOCK: ")[-1]))
    if tot:
        print("UNCLEARED: %d declined lookups -- an added key could be one of them." % tot)
        return 1
    print("SCREEN 2 CLEAR: 0 declined lookups on the qkv path, so no table entry added to it "
          "can fire on this fold. The session's tree is production-equivalent to %s FOR THIS "
          "MODEL AND SIZE." % a.against)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
