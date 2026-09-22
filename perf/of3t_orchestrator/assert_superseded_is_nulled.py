#!/usr/bin/env python3
"""A superseded artifact must be NULLED, not merely stamped.

Standing lesson, found unapplied to four of my own artifacts at pass 180: a `SUPERSEDED` /
`RETIRED` / `REFUTED` field stops nobody reading the numbers out of the *other* fields. A
reader scanning keys sees `scoreboard` and reads it; a script asking for `agreement_results`
gets them. The stamp is documentation, not an interlock.

So: if an artifact carries such a stamp, every structured (list/dict) data field must be
renamed with a `_SUPERSEDED` suffix. Content is kept verbatim for audit -- the point is that
it cannot be picked up BY NAME.

usage: assert_superseded_is_nulled.py [file.json ...]   (default: every of3t_orchestrator json)
"""
import glob
import json
import os
import re as _re
import sys

# THE CONVENTION, made explicit because the first version of this guard conflated two things
# and fired on a live artifact (fourth time one of my guards defaulted too wide; the live sweep
# caught it again, which is why a guard's first real run must not be inside a gate):
#
#   WHOLE-ARTIFACT status  -> a top-level key EXACTLY `SUPERSEDED` / `RETIRED` / `REFUTED` /
#                             `WITHDRAWN`, optionally `_AT_PASS_<n>`. The artifact's central
#                             claim is dead and its data fields must be nulled.
#   PARTIAL correction     -> any other name (`SUPERSEDED_BY_MEASUREMENT`, `CORRECTED_pass180`,
#                             `headline_RETIRED`). ONE figure or clause is superseded inside a
#                             LIVE artifact whose remaining content still stands. Not nulled.
#
# THE_REVISION_ARM_RAN.json is the worked example of the second kind: its
# `SUPERSEDED_BY_MEASUREMENT` retires an 85.6 % estimate while the control reading exactly 0.0
# and the branch that fired remain current.
STAMP_RE = _re.compile(r"^(SUPERSEDED|RETIRED|REFUTED|WITHDRAWN)(_AT_PASS_\d+)?$")
HERE = os.path.dirname(os.path.abspath(__file__))


def check(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        return [f"{path}: unreadable ({e})"]
    if not isinstance(d, dict):
        return []
    stamped = [k for k in d if STAMP_RE.match(k.upper())]
    if not stamped:
        return []
    live = [k for k in d
            if k not in stamped
            and not k.endswith("_SUPERSEDED")
            and isinstance(d[k], (list, dict))]
    if live:
        return [f"{os.path.basename(path)}: carries {stamped[0]} but still exposes live "
                f"structured field(s) {live} -- a stamp does not stop a number being read out "
                f"of another field; rename them with a _SUPERSEDED suffix"]
    return []


def main(argv):
    paths = argv[1:] or sorted(glob.glob(os.path.join(HERE, "*.json")))
    bad = [b for p in paths for b in check(p)]
    for b in bad:
        print("FAIL", b)
    if bad:
        return 1
    print(f"OK {len(paths)} artifact(s): every superseded one has its data fields nulled")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
