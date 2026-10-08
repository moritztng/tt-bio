"""Reproduce the live stall: a trajectory folder with no row in the table.

Mirrors proj_host_8traj_288 as it stood at 21:02Z -- rows for trajectories 2/3/4 all terminated by a
per-stage gate, a 3_Ranked acceptance from trajectory 1, and design_l173_06e36dad56ba7091 on disk
with no row because the box died during its gradient design. Before the fix, prune() left that
folder and every resume died in TrajectoryRecorder.
"""
import csv, os, shutil, subprocess, sys, tempfile

REC = sys.argv[1]
FIELDS = ["trajectory", "design", "length", "i_pTM", "pLDDT", "terminated"]
BANKED = [
    {"trajectory": "2", "design": "design_l173_f20e2293b3df8030", "length": "173",
     "i_pTM": "0.12", "pLDDT": "0.50", "terminated": "harden"},
    {"trajectory": "3", "design": "design_l173_72b0c1e76f0cb82a", "length": "173",
     "i_pTM": "0.57", "pLDDT": "0.52", "terminated": "refine"},
    {"trajectory": "4", "design": "design_l173_409b501e1bfd7f26", "length": "173",
     "i_pTM": "0.50", "pLDDT": "0.50", "terminated": "refine"},
]
ORPHAN = "design_l173_06e36dad56ba7091"
INFLIGHT = {"trajectory": "6", "design": "design_l173_deadbeefdeadbeef", "length": "173",
            "i_pTM": "0.80", "pLDDT": "0.75", "terminated": ""}

root = tempfile.mkdtemp(prefix="bci17_fixture_")
proj = os.path.join(root, "proj_host_8traj_288")
traj = os.path.join(proj, "1_Trajectories")
os.makedirs(traj)
os.makedirs(os.path.join(proj, "2_Refolded"))

with open(os.path.join(traj, "!_Trajectories.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(BANKED + [INFLIGHT])

# Every row's folder, plus the orphan that has no row at all.
for row in BANKED + [INFLIGHT]:
    os.makedirs(os.path.join(traj, row["design"]))
    open(os.path.join(traj, row["design"], "trajectory.pdb"), "w").write("x")
os.makedirs(os.path.join(traj, ORPHAN))
open(os.path.join(traj, ORPHAN, "trajectory.pdb"), "w").write("x")

# The phantom claim a mid-trajectory death leaves behind.
open(os.path.join(proj, ".campaign_state.json"), "w").write('{"trajectories": 6, "attempted": []}')

out = subprocess.run([sys.executable, "-I", REC, proj], capture_output=True, text=True)
print(out.stdout, out.stderr)
assert out.returncode == 0, out.returncode

left = sorted(d for d in os.listdir(traj) if d.startswith("design_"))
rows = list(csv.DictReader(open(os.path.join(traj, "!_Trajectories.csv"))))

failures = []
if ORPHAN in left:
    failures.append(f"orphan {ORPHAN} survived: this is the live stall")
if INFLIGHT["design"] in left:
    failures.append("in-flight row's folder survived")
for row in BANKED:
    if row["design"] not in left:
        failures.append(f"banked {row['design']} was deleted -- completed work lost")
if [r["trajectory"] for r in rows] != ["2", "3", "4"]:
    failures.append(f"table should keep exactly the 3 terminated rows, got {[r['trajectory'] for r in rows]}")
if os.path.exists(os.path.join(proj, ".campaign_state.json")):
    failures.append("phantom claim file survived")

shutil.rmtree(root)
if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PASS: orphan folder removed, 3 banked trajectories and their folders intact, phantom claim gone")
