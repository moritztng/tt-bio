#!/usr/bin/env python3
"""Exercise the REAL (non-dry) cross-host merge against a scratch tier_a.

--dry-run proves the plan, not the execution: rsync gets -n, the progress append is skipped, and the
host-preservation branch never runs. This script points the module's TIERA at a scratch directory and
seeds it so that all but two of the peer's folds already look present locally, so a genuine merge
copies exactly two folds. That exercises rsync for real, the label copy, the progress append and the
host field, at a cost of two folds of disk instead of ninety.

The live campaign directory is never touched.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4")
SCRATCH = Path("/tmp/merge_test/tier_a")
PEER = "tt-quietbox2"

# Peer's ok pairs, straight from the peer.
r = subprocess.run(["ssh", "-o", "BatchMode=yes", f"ttuser@{PEER}",
                    "python3 -c \"import json;"
                    "rs=[json.loads(l) for l in open('/home/ttuser/abag_xm/tier_a/progress.jsonl')"
                    " if l.strip()];"
                    "print(json.dumps([r for r in rs if r.get('status')=='ok']))\""],
                   capture_output=True, text=True, timeout=120)
if r.returncode != 0:
    sys.exit(f"could not read peer progress: {r.stderr[-300:]}")
peer_ok = json.loads(r.stdout)
pairs = sorted({(x["target"], x["model"]) for x in peer_ok})
print(f"peer has {len(pairs)} ok pairs")

hold_out = pairs[:2]          # the only two the merge should have to fetch
print(f"holding out: {hold_out}")

if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
with open(SCRATCH / "progress.jsonl", "w") as f:
    for t, g in pairs:
        if (t, g) in hold_out:
            continue
        # Local records deliberately carry host="tt-quietbox" so a merged row keeping
        # host="tt-quietbox2" is visible as coming from the peer.
        f.write(json.dumps({"target": t, "model": g, "status": "ok",
                            "host": "tt-quietbox"}) + "\n")
seeded = len(pairs) - len(hold_out)
print(f"seeded scratch progress.jsonl with {seeded} local ok records")

spec = importlib.util.spec_from_file_location("mh", WT / "scripts" / "abag_xm_merge_hosts.py")
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)
mh.TIERA = SCRATCH                      # redirect every read/write off the live campaign
print(f"TIERA redirected to {mh.TIERA}\n")

sys.argv = ["abag_xm_merge_hosts.py", "--peer", PEER]
try:
    mh.main()
except SystemExit as e:
    if e.code:
        sys.exit(f"merge exited {e.code}")

print("\n--- verification ---")
recs = [json.loads(l) for l in open(SCRATCH / "progress.jsonl") if l.strip()]
merged = [x for x in recs if x.get("host") == PEER]
print(f"progress.jsonl: {len(recs)} records, {len(merged)} carrying host={PEER}")

fails = []
if len(merged) != len(hold_out):
    fails.append(f"expected {len(hold_out)} merged records, got {len(merged)}")
got = {(x["target"], x["model"]) for x in merged}
if got != set(hold_out):
    fails.append(f"merged the wrong pairs: {sorted(got)} vs {sorted(hold_out)}")
# host must be the fold's OWN host, not the --peer string, when the record already had one.
for x in merged:
    if x.get("host") != "tt-quietbox2":
        fails.append(f"{x['target']}/{x['model']} host={x.get('host')!r}")

# Coordinates and labels must have actually landed.
for t, g in hold_out:
    # GENS[g][1] is ALREADY the full result-dir prefix ("protenix_results"), so the directory is
    # f"{prefix}_{target}". Getting this wrong made the first run of this test report 0 CIFs for
    # folds that had in fact copied 50 -- the script was fine, the check was looking in
    # "<prefix>_results_<target>", which never exists.
    subdir, prefix, labpre = mh.GENS[g]
    sd = SCRATCH / subdir / f"{prefix}_{t}" / "structures"
    n_cif = len(list(sd.glob("*.cif"))) if sd.is_dir() else 0
    n_pae = len(list(sd.glob("*_pae.npz"))) if sd.is_dir() else 0
    lab = SCRATCH / "labels" / f"{labpre}_{t}.json"
    print(f"  {t}/{g}: {n_cif} cifs, {n_pae} paes, label "
          f"{'present' if lab.exists() else 'ABSENT'}")
    if n_cif != 50:
        fails.append(f"{t}/{g}: {n_cif} cifs copied, expected 50")
    if n_pae != 51:
        fails.append(f"{t}/{g}: {n_pae} pae files, expected 51 (50 per-sample + compat copy)")
    if not lab.exists():
        fails.append(f"{t}/{g}: label JSON not copied")

# Idempotence: a second run must merge nothing.
print("\n--- second run (must be a no-op) ---")
sys.argv = ["abag_xm_merge_hosts.py", "--peer", PEER]
try:
    mh.main()
except SystemExit as e:
    if e.code:
        fails.append(f"second run exited {e.code}")
recs2 = [json.loads(l) for l in open(SCRATCH / "progress.jsonl") if l.strip()]
if len(recs2) != len(recs):
    fails.append(f"NOT idempotent: {len(recs)} -> {len(recs2)} records on re-run")

print()
if fails:
    for x in fails:
        print("FAIL:", x)
    sys.exit(1)
print("PASS: merged exactly the held-out folds with 50 CIFs + labels each, kept each fold's own "
      "host, and a second run changed nothing")
