"""Run one af2ig-trunk-device arm under a pinned grid and keep its report.

Argv the gate itself builds for the leg (scripts/full_parity_gate.py::run_inprocess), so the arm scores
the same code path the record did; the only delta is TT_BIO_FORCE_GRID.
"""
import json
import os
import subprocess
import sys
import time

WT = "/home/ttuser/.coworker/wt/tt-bio-release-bh-0-8-p3"
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
PARAMS = "/home/ttuser/.boltz/af2/params/params_model_1_ptm.npz"
OUT = "/home/ttuser/scratch/gate-p17"

grid, name = sys.argv[1], sys.argv[2]
rep_path = os.path.join(OUT, f"{name}.json")
if os.path.exists(rep_path):
    print(f"{name}: already have {rep_path}")
    sys.exit(0)

env = {k: v for k, v in os.environ.items() if k != "TT_MESH_GRAPH_DESC_PATH"}
env.update({
    "PYTHONPATH": WT,
    "TT_VISIBLE_DEVICES": "0",
    "TT_BIO_LEASE_CARDS": "0",
    "TT_BIO_LEASE_HOLDER": "worker:tt-bio-release-bh-0-8-p3",
    "TT_BIO_FORCE_GRID": grid,
})
argv = [PY, "scripts/af2_port/tap_gate.py", "--params", PARAMS, "--stage", "complex", "--device"]
t0 = time.monotonic()
proc = subprocess.run(argv, cwd=WT, capture_output=True, text=True, env=env)
wall = time.monotonic() - t0
with open(os.path.join(OUT, f"{name}.log"), "w") as f:
    f.write(proc.stdout)
    f.write("\n===== stderr =====\n")
    f.write(proc.stderr)

# tt-metal logs to stdout, so take the last object that decodes (_report_from_stdout's rule).
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
    print(f"{name}: NO REPORT rc={proc.returncode} wall={wall:.0f}s")
    print((proc.stderr or proc.stdout)[-2000:])
    sys.exit(1)
found["_arm"] = {"grid": grid, "wall_s": round(wall, 1), "rc": proc.returncode}
with open(rep_path, "w") as f:
    json.dump(found, f, indent=2, default=str)
print(f"{name}: grid={grid} rc={proc.returncode} wall={wall:.0f}s verdict={found.get('verdict')} "
      f"taps_scored={found.get('taps_scored')} taps_failed={found.get('taps_failed')} "
      f"pcc_min={found.get('pcc_min')!r} scalars_failed={found.get('scalars_failed')}")
