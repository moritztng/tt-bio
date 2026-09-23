"""Time one model's rungs on one pinned chip, for the speed bar (docs/speed-bar.md).

Each fold is the size ladder's own (release_gate._run_census_fold: same fixture, fold settings,
AICLK and load sampling) with the size guard off, so a rung above a model's guarded cap folds
instead of recording a refusal. Rungs below the cap are unaffected by the guard. Like the ladder's
recorder, every rung gets one cold warm-up fold (its kernels compile there) that is logged but not
timed, then `reps` timed folds; the ladder's sigma rung (512) gets three.

One JSON line per fold goes to perf/mgx-speed/runs/<model>.jsonl with the commit, host, chip and
host thread cap, which is the identity the bar requires to be shared by every rung it compares.
A rung above 1024 that fails ends the walk: the rungs above it would allocate more.

The walk resumes: a rung that already has its warm-up and enough timed folds under the load
ceiling on this chip and engine tree is skipped, and a timed fold whose 1-min load passed the
ceiling (VOID under the bar) is re-run: one pass spends at most reps + EXTRA timed folds on a rung.
A fold the lease refused (another row opened the chip between folds) ran nothing and does not
count against that budget.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> ... python perf/mgx-speed/time_rungs.py \
        <model> <rung>[,<rung>...] [threads] [sigma_reps] [sigma_rung]

sigma_rung defaults to the ladder's; nesso1 takes it inside its clock-matched fit (verdicts.py).
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
from gate_guard import DEFAULT_LOAD_CEILING as LOAD_CEILING  # noqa: E402

EXTRA = 2                           # folds per pass a rung may spend past its reps
CONTENDED = 6                       # lease refusals per rung and pass before the walk moves on

model, rungs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
rg.HOST_THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
sigma_reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
sigma_rung = int(sys.argv[5]) if len(sys.argv) > 5 else rg._size_ladder_sigma_rung(model)
git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
commit = git("rev-parse", "HEAD")
engine = lambda c: git("rev-parse", f"{c}:tt_bio", f"{c}:scripts").replace("\n", " ")
if git("status", "--porcelain", "--", "tt_bio", "scripts"):
    sys.exit(f"engine tree is dirty at {commit}: a timed rung must name a commit")
ident = {"model": model, "commit": commit, "engine": engine(commit), "host": socket.gethostname(),
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "host_threads": rg.HOST_THREADS}
out = ROOT / "perf" / "mgx-speed" / "runs"
out.mkdir(parents=True, exist_ok=True)
log = out / f"{model}.jsonl"
done = [c for c in map(json.loads, log.read_text().splitlines() if log.exists() else [])
        if (c["host"], c["card"], c["host_threads"]) == (ident["host"], ident["card"], rg.HOST_THREADS)
        and (c.get("engine") or engine(c["commit"])) == ident["engine"]]
work = Path(os.environ.get("RELEASE_GATE_SIZE_WORKDIR", ROOT / "perf" / "mgx-speed" / f"work-{model}"))


def fold(rung, tag):
    t0 = time.time()
    r = rg._run_census_fold(model, rung, work, tag)
    cell = {**ident, "rung": rung, "tag": tag,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "wall_s": round(time.time() - t0, 1)}
    for k in ("error", "refused", "runtime_s", "aiclk", "load", "structure"):
        if r.get(k) is not None:
            cell[k] = r[k]
    with open(log, "a") as fp:
        fp.write(json.dumps(cell, default=str) + "\n")
    print(json.dumps({k: v for k, v in cell.items() if k != "structure"}, default=str), flush=True)
    return cell


# exit 75 is the lease refusing a chip another row opened between our folds: nothing ran, so it
# is neither a coverage result nor a timing, just a fold to take again
contended = lambda c: (c.get("error") or "").startswith("census fold exited 75")
failed = lambda c: bool(c.get("error") or c.get("refused")) and not contended(c)
quiet = lambda c: (c.get("load") or {}).get("max", LOAD_CEILING + 1) <= LOAD_CEILING
for rung in rungs:
    mine = [c for c in done if c["rung"] == rung and not contended(c)]
    reps = sigma_reps if rung == sigma_rung else 1
    for _ in range(1 + EXTRA):
        if any(c["tag"] == "warmup" for c in mine) or any(map(failed, mine)):
            break
        if not contended(w := fold(rung, "warmup")):
            mine.append(w)
    timed = [c for c in mine if c["tag"] != "warmup"]
    budget, refused = len(timed) + reps + EXTRA, 0
    while (not any(map(failed, mine)) and sum(map(quiet, timed)) < reps and len(timed) < budget
           and refused < CONTENDED):
        if contended(c := fold(rung, f"rep{len(timed)}")):
            refused += 1
            continue
        timed.append(c)
        mine.append(c)
    if any(map(failed, mine)) and rung > 1024:
        break
