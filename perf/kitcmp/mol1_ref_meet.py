"""Hand the reference job list from two B200 boxes to one when they meet, run DETACHED on pc.

    python3 mol1_ref_meet.py <fwd-label> <rev-label> <qb2 user@host:dir>

mol1_ref.py on <fwd-label> walks the job list forward, on <rev-label> backward (REF_REVERSE=1). Each box decides
what to skip only at startup, so past the meeting point each would redo the other's work. Every 5 minutes this
reads both pulled jobs.jsonl; once the gap is <= GAP jobs it
  1. stops the forward box's run and writes MOL1-DONE there, so its session watcher pulls, offloads to qb2, destroys;
  2. copies the forward box's outputs (from qb2) into the reverse box's ref tree, never overwriting;
  3. stops the reverse box's K=4 run, so box_mol1.sh's K=1 pass rescans the tree and runs only what is missing
     (jobs that failed under K-way sharing, the ones cut off in flight).
"""
import json, subprocess, sys, time
from pathlib import Path

GAP = 8
R = Path(__file__).resolve().parent / "results"
fwd, rev, qb2 = sys.argv[1:4]
qhost, qdir = qb2.split(":", 1)
log = lambda m: print(time.strftime("%FT%TZ", time.gmtime()), m, flush=True)
ssh = lambda lab: (R / lab / "ssh").read_text().split()


def run(cmd, t=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
    log(f"rc={p.returncode} {' '.join(cmd)[-160:]} {p.stdout[-300:]} {p.stderr[-300:]}")
    return p.returncode


def positions():
    idx = json.loads(subprocess.run(["ssh", "-o", "BatchMode=yes", qhost, f"cat {qdir}/inputs/INDEX.json"],
                                    capture_output=True, text=True, timeout=300, check=True).stdout)
    head = set(idx["headline_only"])
    paired = [c for c in idx["cases"] if c not in head]
    seeds, selftest, recipes = [42, 66, 101, 2024, 8888], [7, 13, 77, 1234, 9999], list(
        ["default", "use_msa_false", "use_template_true", "use_rna_msa_true"])
    jobs = ([(c, "paired", x) for c in paired for x in seeds] + [(c, "selftest", x) for c in paired for x in selftest]
            + [(c, "headline", x) for c in sorted(head) for x in seeds])  # mol1_ref.py's order
    rec = lambda c: idx["recipes"].get(c, "default")
    jobs.sort(key=lambda j: recipes.index(rec(j[0])) if rec(j[0]) in recipes else len(recipes))
    return {j: i for i, j in enumerate(jobs)}


def seen(lab, pos):
    out, f = [], R / lab / "mol1/ref/jobs.jsonl"
    for line in (f.read_text().splitlines() if f.exists() else []):  # a box still renting has no ledger yet
        try:
            d = json.loads(line)
            out.append(pos[(d["case"], d["set"], d["seed"])])
        except Exception:
            continue
    return out


pos = positions()
while True:
    a, b = seen(fwd, pos), seen(rev, pos)
    gap = (min(b) if b else len(pos)) - (max(a) + 1 if a else 0)
    log(f"{fwd} at {max(a, default=-1)}, {rev} at {min(b, default=len(pos))}, gap {gap} of {len(pos)}")
    if gap <= GAP:
        break
    time.sleep(300)

# The [x] brackets keep pgrep from matching the remote shell that carries this very command line.
stop4 = "for p in $(pgrep -f '[m]ol1_ref\\.py.*--k 4'); do pkill -P $p; kill $p; done"
kill = f"pkill -f '[b]ox_mol1\\.sh'; {stop4}"
run(ssh(fwd) + [f"{kill}; sleep 5; echo done > /root/kc/mol1/MOL1-DONE"])
# The forward watcher offloads on its next 3-minute tick; wait for its SESSION-END so qb2 has every CIF.
while not (R / fwd / "SESSION-END").exists():
    time.sleep(60)
rs = " ".join(ssh(rev)[:-1])
run(["ssh", "-o", "BatchMode=yes", qhost, f"rsync -a --ignore-existing --exclude jobs.jsonl -e '{rs}' "
     f"{qdir}/{fwd}/ref/ {ssh(rev)[-1]}:/root/kc/mol1/ref/"], t=3600)
run(ssh(rev) + [stop4])
log("handed over; the reverse box runs the K=1 remainder")
