"""Time one model's rungs on one pinned chip, for the speed bar (docs/speed-bar.md).

Each fold is the size ladder's own (release_gate._run_census_fold: same fixture, fold settings,
AICLK and load sampling) with the size guard off, so a rung above a model's guarded cap folds
instead of recording a refusal. Rungs below the cap are unaffected by the guard. Like the ladder's
recorder, every rung gets one cold warm-up fold (its kernels compile there) that is logged but not
timed, then `reps` timed folds; the ladder's sigma rung (512) gets three.

One JSON line per fold goes to perf/mgx-speed/runs/<model>.jsonl with the commit, host, chip and
host thread cap, which is the identity the bar requires to be shared by every rung it compares.
A rung above 1024 that fails ends the walk: the rungs above it would allocate more.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> ... python perf/mgx-speed/time_rungs.py \
        <model> <rung>[,<rung>...] [threads] [sigma_reps]
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["TT_BIO_SIZE_LIMIT"] = "0"
import release_gate as rg  # noqa: E402

model, rungs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
rg.HOST_THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
sigma_reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
commit = git("rev-parse", "HEAD")
if git("status", "--porcelain", "--", "tt_bio", "scripts"):
    sys.exit(f"engine tree is dirty at {commit}: a timed rung must name a commit")
ident = {"model": model, "commit": commit, "host": socket.gethostname(),
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "host_threads": rg.HOST_THREADS}
out = ROOT / "perf" / "mgx-speed" / "runs"
out.mkdir(parents=True, exist_ok=True)
work = Path(os.environ.get("RELEASE_GATE_SIZE_WORKDIR", ROOT / "perf" / "mgx-speed" / f"work-{model}"))
for rung in rungs:
    reps = sigma_reps if rung == rg._size_ladder_sigma_rung(model) else 1
    failed = False
    for rep in range(reps + 1):
        tag = "warmup" if rep == 0 else f"rep{rep - 1}"
        t0 = time.time()
        r = rg._run_census_fold(model, rung, work, tag)
        cell = {**ident, "rung": rung, "tag": tag,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "wall_s": round(time.time() - t0, 1)}
        for k in ("error", "refused", "runtime_s", "aiclk", "load", "structure"):
            if r.get(k) is not None:
                cell[k] = r[k]
        with open(out / f"{model}.jsonl", "a") as fp:
            fp.write(json.dumps(cell, default=str) + "\n")
        print(json.dumps({k: v for k, v in cell.items() if k != "structure"}, default=str), flush=True)
        if r.get("error") or r.get("refused"):
            failed = True
            break
    if failed and rung > 1024:
        break
