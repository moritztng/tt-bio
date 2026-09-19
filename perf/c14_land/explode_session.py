#!/usr/bin/env python3
"""Explode a driver session JSON back into the per-leg files score_bracket.py reads.

apb_fold_ab.py writes its per-leg files into a scratch directory that does not survive the
run, and the aggregate <name>_ab.json it leaves behind holds exactly the same records. This
re-materialises them so the PRE-REGISTERED scorer runs UNCHANGED against the session, rather
than the scorer being edited to reach the data it is judging.

Naming is score_bracket.load_blocks()'s own: <size>_<arm>_<block>_<leg>.json, with the leg
index taken from the record's own "pos" field.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import provenance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
a = ap.parse_args()

d = json.loads(Path(a.session).read_text())

# Refuse to hand the scorer legs that did not all run the same tt_bio. Heads may differ --
# a harness or docs commit cannot reach a fold -- but the imported code may not.
ok, lines = provenance.check(provenance.heads_of(d), a.repo)
print("\n".join(lines))
if not ok:
    sys.exit(1)

out = Path(a.dir)
out.mkdir(parents=True, exist_ok=True)
n = 0
for b in d["blocks"]:
    if b.get("returncode") != 0 or not b.get("result"):
        continue
    name = "%s_%s_%d_%d.json" % (b["size"], b["arm"], b["block"], b["pos"])
    (out / name).write_text(json.dumps(b["result"], indent=1) + "\n")
    n += 1
print("%d legs -> %s" % (n, out))
