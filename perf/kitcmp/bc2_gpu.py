"""Stock BindCraft 2 on a GPU, timed the way perf/spd/bench.py --model bindcraft2 times it on a TT chip: BindCraft 2's
own campaign on its shipped PD-L1 example (examples/pdl1.json, settings unedited but campaign_seed, max_trajectories and
binder_lengths), one trajectory at a time, JAX on the GPU for everything. design_s = BindCraft 2's run_trajectory call
(every design stage), validate_s = MPNN redesign + validation; seconds per design iteration = design_s / rows of that
trajectory's losses.csv (rows new since its start). Trajectory 1 compiles (cold), the rest are warm.
    python bc2_gpu.py --bc2 BC2_CHECKOUT --out DIR [--seed 101] [--warm 3] [--binder-len 80]   -> DIR/runs.jsonl

BC4's quality reference (state/bc4/QUALITY-REF-REQUEST.json) runs the same harness for a different purpose: whole
campaigns at several seeds, graded by perf/spd/bc2_quality.py rather than timed. That needs another example file, a
trajectory budget the campaign always spends, and the example's own binder lengths, so --example, --max-traj,
--final-designs and --binder-len example exist. Every default is the timing run's, so the protocol command is
unchanged.
"""
import argparse, contextlib, csv, json, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--binder-len", default="80", help="binder_lengths override, or 'example' to keep the example's own")
ap.add_argument("--example", default="pdl1.json", help="file under the BC2 checkout's examples/")
ap.add_argument("--max-traj", type=int, default=None, help="max_trajectories; default 1 + --warm")
ap.add_argument("--final-designs", type=int, default=None,
                help="number_of_final_designs; raise it so a campaign always spends its whole trajectory budget "
                     "instead of stopping early on an accepted design")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.bc2))
import jax  # noqa: E402
import bindcraft.campaign as campaign  # noqa: E402
from bindcraft.campaign_output import TRAJECTORY_STAGE, accepted_table, stage_table, trajectory_metric_rows  # noqa: E402
from bindcraft.model_weights import model_weights  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402

project = a.out / f"campaign_s{a.seed}"
overrides = [f"campaign_seed={a.seed}", f"max_trajectories={a.max_traj or 1 + a.warm}", f"project_folder={project}"]
if a.binder_len != "example":
    overrides.append(f"binder_lengths=[{int(a.binder_len)}]")
if a.final_designs is not None:
    overrides.append(f"number_of_final_designs={a.final_designs}")
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / a.example),
                                                  parse_setting_overrides(overrides)))
sha = subprocess.run(["git", "-C", str(a.bc2), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
LOG = open(a.out / "runs.jsonl", "a")


def log(**kw):
    LOG.write(json.dumps(dict(t=time.time(), **kw), default=str) + "\n"); LOG.flush()


def sm_mhz(t0, t1):
    """SM clock median over [t0, t1] from the 1 Hz nvidia-smi log box_bc.sh keeps beside us."""
    xs = []
    with contextlib.suppress(OSError):
        for ln in open(a.out / "smi.csv"):
            p = [s.strip() for s in ln.split(",")]
            with contextlib.suppress(ValueError, IndexError):
                ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()  # nvidia-smi's local-time stamp
                if t0 <= ts <= t1:
                    xs.append(float(p[1].split()[0]))
    xs.sort()
    return dict(median=xs[len(xs) // 2], min=xs[0], max=xs[-1], n=len(xs)) if xs else None


log(ev="start", bc2_sha=sha, jax=jax.__version__, devices=[str(d) for d in jax.devices()], gpu=gpu,
    settings={k: v for k, v in settings.items() if k != "targets"}, targets=settings.get("targets"), argv=sys.argv)
traj = []
real_design, real_validate = campaign.run_trajectory, campaign.redesign_and_validate_binders


def loss_rows():
    """Rows of every trajectory's losses.csv so far (BindCraft 2 may have zipped a finished trajectory's folder)."""
    with contextlib.suppress(Exception):
        return sum(len(r) for r in trajectory_metric_rows(str(project)))
    return 0


def emit(i, t_end):
    tr = traj[i]
    rows = []
    with contextlib.suppress(OSError):
        rows = list(csv.DictReader(open(stage_table(str(project), TRAJECTORY_STAGE))))
    row = next((r for r in rows if r.get("trajectory") == str(tr["n"])), {})
    acc = 0
    with contextlib.suppress(OSError):
        acc = sum(1 for _ in csv.DictReader(open(accepted_table(str(project)))))
    iters = tr.get("iterations")
    log(ev="rep", trajectory=tr["n"], kind="cold" if i == 0 else "warm", design_s=tr.get("design_s"),
        validate_s=tr.get("validate_s"), traj_s=t_end - tr["t0w"], iterations=iters,
        s_per_iter=(tr["design_s"] / iters) if iters and tr.get("design_s") else None,
        sm=sm_mhz(tr["t0w"], tr["t0w"] + (tr.get("design_s") or 0)), terminated=row.get("terminated") or None,
        accepted_total=acc, csv_row=row, err=tr.get("err"))


def design(*args, **kw):
    if traj:
        emit(len(traj) - 1, time.time())
    traj.append(dict(n=len(traj) + 1, t0w=time.time(), rows0=loss_rows()))
    t0 = time.monotonic()
    try:
        return real_design(*args, **kw)
    finally:
        traj[-1]["design_s"] = time.monotonic() - t0
        traj[-1]["iterations"] = loss_rows() - traj[-1]["rows0"]


def validate(*args, **kw):
    t0 = time.monotonic()
    try:
        return real_validate(*args, **kw)
    finally:
        traj[-1]["validate_s"] = time.monotonic() - t0


campaign.run_trajectory, campaign.redesign_and_validate_binders = design, validate
af2, mpnn = model_weights()
try:
    campaign.run_campaign(settings, str(project), af2_weights=af2, mpnn_weights=mpnn)
except Exception:
    if traj:
        traj[-1]["err"] = traceback.format_exc()[-3000:]
    else:
        log(ev="crash", err=traceback.format_exc()[-3000:])
if traj:
    emit(len(traj) - 1, time.time())
log(ev="end")
