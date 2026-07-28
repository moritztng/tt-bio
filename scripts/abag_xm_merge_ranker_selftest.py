#!/usr/bin/env python3
"""Test the ranker_scores.csv merge: dedup, local-wins, append, and the dry-run path.

The peer path is fixed at ~/abag_xm/tier_a/ranker_scores.csv on the peer host, so exercising the
merging branch for real would mean writing into qb2's live campaign directory. Instead stub the ssh
read with a synthetic peer CSV and redirect TIERA to scratch, which tests the logic that actually
decides which rows land.
"""
import csv
import importlib.util
import io
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

WT = Path("/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4")
spec = importlib.util.spec_from_file_location("mh", WT / "scripts" / "abag_xm_merge_hosts.py")
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)

SCRATCH = Path("/tmp/ranker_merge_test")
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
mh.TIERA = SCRATCH
# The merge refuses to run without a local progress.jsonl. An empty one is the right seed here:
# it means "this host has folded nothing", so the coordinate/label/progress steps are all no-ops
# and only the ranker branch does work.
(SCRATCH / "progress.jsonl").write_text("")

HEADER = ["target", "gen", "rank", "iptm", "dockq", "deeprank_ab"]


def row(t, g, k, dr):
    return {"target": t, "gen": g, "rank": str(k), "iptm": "0.5", "dockq": "0.4",
            "deeprank_ab": dr}


# Local CSV: one pair, already scored, with a recognisable deeprank value.
local = [row("AAAA", "boltz2", k, "LOCAL") for k in range(3)]
with open(SCRATCH / mh.RANKER_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    for r in local:
        w.writerow(r)

# Peer CSV: the same pair (must be REJECTED, local wins) plus two new pairs (must be appended).
peer = ([row("AAAA", "boltz2", k, "PEER") for k in range(3)]
        + [row("BBBB", "protenix-v2", k, "PEER") for k in range(3)]
        + [row("CCCC", "opendde-abag", k, "PEER") for k in range(3)])
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=HEADER)
w.writeheader()
for r in peer:
    w.writerow(r)
peer_text = buf.getvalue()

# Stub the peer read. Everything else in main() is skipped by calling the branch directly is not
# possible (it is inline), so stub _ssh and drive main() with nothing to merge on the fold side:
# progress on both sides is empty, so the coordinate/label/progress steps are all no-ops.
calls = []


def fake_ssh(peer_host, cmd, timeout=120):
    calls.append(cmd)
    if "ranker_scores.csv" in cmd:
        return SimpleNamespace(returncode=0, stdout=peer_text, stderr="")
    if "progress.jsonl" in cmd:
        return SimpleNamespace(returncode=0, stdout="", stderr="")   # peer has no ok folds
    return SimpleNamespace(returncode=0, stdout="", stderr="")


mh._ssh = fake_ssh

print("--- dry run (must not modify the CSV) ---")
sys.argv = ["merge", "--peer", "fakepeer", "--dry-run"]
try:
    mh.main()
except SystemExit as e:
    if e.code:
        sys.exit(f"dry run exited {e.code}")
after_dry = list(csv.DictReader(open(SCRATCH / mh.RANKER_CSV)))
print(f"rows after dry run: {len(after_dry)} (expected 3)")

print("\n--- real run ---")
sys.argv = ["merge", "--peer", "fakepeer"]
try:
    mh.main()
except SystemExit as e:
    if e.code:
        sys.exit(f"real run exited {e.code}")
rows = list(csv.DictReader(open(SCRATCH / mh.RANKER_CSV)))
pairs = {}
for r in rows:
    pairs.setdefault((r["target"], r["gen"]), set()).add(r["deeprank_ab"])
print(f"rows after merge: {len(rows)}")
for k in sorted(pairs):
    print(f"  {k}: deeprank_ab={sorted(pairs[k])}")

fails = []
if len(after_dry) != 3:
    fails.append(f"dry run modified the CSV: {len(after_dry)} rows")
if len(rows) != 9:
    fails.append(f"expected 9 rows after merge (3 local + 6 new peer), got {len(rows)}")
if pairs.get(("AAAA", "boltz2")) != {"LOCAL"}:
    fails.append(f"local-wins violated for AAAA/boltz2: {pairs.get(('AAAA','boltz2'))}")
for k in (("BBBB", "protenix-v2"), ("CCCC", "opendde-abag")):
    if pairs.get(k) != {"PEER"}:
        fails.append(f"{k} not merged from peer: {pairs.get(k)}")

print()
if fails:
    for x in fails:
        print("FAIL:", x)
    sys.exit(1)
print("PASS: dry run inert; duplicate pair kept the LOCAL row; both new peer pairs appended")
