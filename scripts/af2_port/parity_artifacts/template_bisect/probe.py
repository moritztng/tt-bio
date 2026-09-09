"""One af2ig-trunk-device probe out of an arbitrary checkout, on an arbitrary card.

Extends scripts/af2_port/parity_artifacts/grid_control/armrun2.py (the section-71 arm runner)
with a card argument and the GOOD/BAD call, scored by a PINNED copy of HEAD's device_floor.py
against a PINNED copy of HEAD's committed floor, so the instrument does not move with the
checkout under test.

    python3 probe.py <worktree> <card> <name>
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
PARAMS = "/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz"
OUT = "/home/ttuser/scratch/gate-p18"
HOLDER = "worker:tt-bio-release-bh-0-8-p4"

spec = importlib.util.spec_from_file_location("dfp", os.path.join(OUT, "device_floor_pinned.py"))
dfp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfp)
FLOOR = json.load(open(os.path.join(OUT, "floor_pinned.json")))

wt, card, name = sys.argv[1], sys.argv[2], sys.argv[3]
rep_path = os.path.join(OUT, name + ".json")
if os.path.exists(rep_path):
    print(name + ": already have " + rep_path)
    sys.exit(0)

env = {k: v for k, v in os.environ.items() if k != "TT_MESH_GRAPH_DESC_PATH"}
env.update({"PYTHONPATH": wt, "TT_VISIBLE_DEVICES": card,
            "TT_BIO_LEASE_CARDS": card, "TT_BIO_LEASE_HOLDER": HOLDER})
argv = [PY, "scripts/af2_port/tap_gate.py", "--params", PARAMS, "--stage", "complex", "--device"]
t0 = time.monotonic()
proc = subprocess.run(argv, cwd=wt, capture_output=True, text=True, env=env)
wall = time.monotonic() - t0
with open(os.path.join(OUT, name + ".log"), "w") as f:
    f.write(proc.stdout)
    f.write("\n===== stderr =====\n")
    f.write(proc.stderr)

dec, found, i = json.JSONDecoder(), None, proc.stdout.find("{")
while i != -1:
    try:
        obj, end = dec.raw_decode(proc.stdout, i)
    except json.JSONDecodeError:
        i = proc.stdout.find("{", i + 1)
        continue
    if isinstance(obj, dict):
        found = obj
    i = proc.stdout.find("{", max(end, i + 1))
if found is None:
    print(name + ": NO REPORT rc=%d wall=%.0fs" % (proc.returncode, wall))
    print((proc.stderr or proc.stdout)[-3000:])
    sys.exit(1)

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=wt, capture_output=True,
                     text=True).stdout.strip()
verdict, detail = dfp.af2ig_device_floor_verdict(found, FLOOR)
found["_probe"] = {"checkout": wt, "sha": sha, "card": card, "wall_s": round(wall, 1),
                   "rc": proc.returncode, "floor_verdict": verdict, "floor_detail": detail}
json.dump(found, open(rep_path, "w"), indent=2, default=str)
P = ("PASS", "IN-ENVELOPE")
bad = sorted(r["tap"] for r in found.get("rows", []) if r.get("verdict") not in P)
print("%s sha=%s card=%s wall=%.0fs taps_scored=%s taps_failed=%s scalars_failed=%s pcc_min=%r"
      % (name, sha, card, wall, found.get("taps_scored"), found.get("taps_failed"),
         found.get("scalars_failed"), found.get("pcc_min")))
print("   floor_verdict=%s  %s" % (verdict, detail))
print("   sm3_in_failing=%s  failing=%s" % (
    any(t.startswith("structure_module#3") for t in bad), bad))
