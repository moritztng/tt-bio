#!/usr/bin/env python3
"""Print the last structure gpu5_bench.py wrote, or nothing if the run produced none.

Reads $OUTJSON. Its own file rather than an inline `python3 -c` in gpu5_session.sh: the
session script is itself generated through a heredoc, and the nested quoting collapsed
once already, leaving the gate to be handed an empty path and report "no structure at "
for a run that had in fact succeeded.

The LAST prediction is the last warm fold, not the discarded cold one.
"""
import json
import os
import sys

path = os.environ.get("OUTJSON", "")
try:
    with open(path) as fh:
        preds = (json.load(fh).get("result") or {}).get("predictions") or []
except Exception:
    sys.exit(0)          # no json, no structure: the gate reports it, this is not the place
if preds:
    print(preds[-1])
