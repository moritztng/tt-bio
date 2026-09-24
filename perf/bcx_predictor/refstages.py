"""Parse the reference arms' stage verdicts into a file git will actually keep.

The qb1 reference arms live in `/dev/shm` and are snapshotted into `refsnap/qb1/` every pass,
because volatile memory is the one place this campaign's irreplaceable data sits. But
`.gitignore:71` is `*.log`, and `run.log` is the ONLY place a stage verdict is written -- the
committed `.campaign_state.json` buys resume, not results. So the snapshot has been copying the
verdicts onto disk and git has been dropping them ever since, and they would die with the
worktree. This extracts them into a tracked artifact.

It also happens to be the reference half of the paired per-stage comparison, which is the GO
bar, so it is the table this row needs rather than a backup of a log.
"""
import json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
SNAP = HERE / "refsnap" / "qb1"

STAGE = re.compile(
    r"^\s*(passed|rejected at)\s+(\w+)\s+design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")
TRAJ = re.compile(r"^=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+) ===")

out = {"why": ".gitignore *.log dropped every reference stage verdict; this is the tracked form",
       "arms": {}}
for log in sorted(SNAP.glob("*/run.log")):
    arm = log.parent.name
    traj, rows = None, []
    for line in log.read_text(errors="replace").splitlines():
        m = TRAJ.match(line)
        if m:
            traj = {"index": int(m.group(1)), "name": m.group(2),
                    "accepted": int(m.group(3)), "target": int(m.group(4))}
            continue
        m = STAGE.match(line)
        if m:
            rows.append({"trajectory": traj["name"] if traj else None,
                         "trajectory_index": traj["index"] if traj else None,
                         "stage": m.group(2),
                         "outcome": "passed" if m.group(1) == "passed" else "rejected",
                         "iptm": float(m.group(3)), "plddt": float(m.group(4)),
                         "failed_filter": m.group(5)})
    stamp = {}
    sf = log.parent / "arm_stamp.json"
    if sf.exists():
        s = json.loads(sf.read_text())
        stamp = {k: s.get(k) for k in ("arm", "seed", "length_bucket_size", "host", "commit")}
    out["arms"][arm] = {"stamp": stamp, "stages": rows,
                        "stages_passed": sum(r["outcome"] == "passed" for r in rows),
                        "trajectories_started": max(
                            [r["trajectory_index"] or 0 for r in rows] or [0]),
                        "accepted": traj["accepted"] if traj else None}

(HERE / "refsnap" / "qb1_stages.json").write_text(json.dumps(out, indent=1))
for arm, a in out["arms"].items():
    print(arm, "bucket", a["stamp"].get("length_bucket_size"), "| accepted", a["accepted"])
    for r in a["stages"]:
        print(f"   t{r['trajectory_index']} {r['stage']:8s} {r['outcome']:9s}"
              f" i_pTM {r['iptm']:.2f}  pLDDT {r['plddt']:.2f}"
              f"{'  <- ' + r['failed_filter'] if r['failed_filter'] else ''}")
