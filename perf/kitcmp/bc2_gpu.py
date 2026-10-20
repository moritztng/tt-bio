"""Stock BindCraft 2 on a GPU, timed the way perf/spd/bench.py --model bindcraft2 times it on a TT chip: BindCraft 2's
own campaign on its shipped PD-L1 example (examples/pdl1.json, settings unedited but campaign_seed, max_trajectories,
binder_lengths and auto_multi_gpu=false), one trajectory at a time in this process, JAX on the GPU for everything.
Left on, auto_multi_gpu packs several design worker processes onto a large card, which neither times one trajectory nor
reaches the hooks here. design_s = BindCraft 2's run_trajectory call
(every design stage), validate_s = MPNN redesign + validation; seconds per design iteration = design_s / rows of that
trajectory's losses.csv (rows new since its start), counted by bc2_traj.py, the code bench.py uses on TT.
Trajectory 1 compiles (cold), the rest are warm.
    python bc2_gpu.py --bc2 BC2_CHECKOUT --out DIR [--seed 101] [--warm 3] [--binder-len 80] [--set K=V ...]
                                                                                              -> DIR/runs.jsonl
--workers N runs the campaign BindCraft 2's own packed way instead: N design worker processes on the one GPU
(auto_multi_gpu, workers_per_gpu=N), drawing trajectories from one file-locked campaign, and the record is one
ev=campaign line with the campaign wall and every trajectory's own design time. That is PROTOCOL rule 5's "the GPU
at its own best concurrency", and its TT twin is bench.py --trajectories N. Compare only campaign iterations per
wall second, and take the N=1 leg with --workers 1 as well, so the driver is the one variable that does not move.

--set takes a BindCraft 2 setting for one GPU arm (use_cueq=true, attention_backend=cudnn); XLA flags go in
BC2_XLA_EXTRA, which BindCraft 2 appends to XLA_FLAGS itself. The start record carries the resolved settings and env.

BC4's quality reference (state/bc4/QUALITY-REF-REQUEST.json) runs the same harness for a different purpose: whole
campaigns at several seeds, graded by perf/spd/bc2_quality.py rather than timed. That needs another example file, a
trajectory budget the campaign always spends, and the example's own binder lengths, so --example, --max-traj,
--final-designs and --binder-len example exist. Every default is the timing run's, so the protocol command is
unchanged.
"""
import argparse, contextlib, json, os, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--binder-len", default="80", help="binder_lengths override, or 'example' to keep the example's own")
ap.add_argument("--set", action="append", default=[], metavar="K=V", help="a BindCraft 2 setting for this arm")
ap.add_argument("--example", default="pdl1.json", help="file under the BC2 checkout's examples/")
ap.add_argument("--max-traj", type=int, default=None, help="max_trajectories; default 1 + --warm")
ap.add_argument("--workers", default=None,
                help="design worker processes packed on the one GPU (PROTOCOL rule 5): N, or 'auto' for BindCraft "
                     "2's own choice (7 on a B200 with the memory for it). Omit for the one-at-a-time protocol "
                     "path. Use --workers 1 for the baseline leg of a concurrency A/B")
ap.add_argument("--final-designs", type=int, default=None,
                help="number_of_final_designs; raise it so a campaign always spends its whole trajectory budget "
                     "instead of stopping early on an accepted design")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.bc2))
import jax  # noqa: E402
import bindcraft.campaign as campaign  # noqa: E402
from bindcraft.model_weights import model_weights  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from bc2_traj import campaign_trajectories, cpu_model, hook, s_per_iter, table_row  # noqa: E402  (beside this file, here and on the box)

project = a.out / f"campaign_s{a.seed}"
packed = a.workers is not None
overrides = [f"campaign_seed={a.seed}", f"max_trajectories={a.max_traj or 1 + a.warm}",
             f"project_folder={project}",
             *([f"auto_multi_gpu=true", f"workers_per_gpu={a.workers}"]
               + ([] if a.workers == "auto" else [f"max_workers_per_gpu={a.workers}"])
               if packed else ["auto_multi_gpu=false"])]
if a.binder_len != "example":
    overrides.append(f"binder_lengths=[{int(a.binder_len)}]")
if a.final_designs is not None:
    overrides.append(f"number_of_final_designs={a.final_designs}")
overrides += a.set
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
    settings={k: v for k, v in settings.items() if k != "targets"}, targets=settings.get("targets"), argv=sys.argv,
    xla_flags=os.environ.get("XLA_FLAGS"), cpu=cpu_model(), ncpu=os.cpu_count())


def emit(i):
    tr = traj[i]
    row, acc = table_row(project, tr["n"])
    log(ev="rep", trajectory=tr["n"], kind="cold" if i == 0 else "warm", design_s=tr.get("design_s"),
        validate_s=tr.get("validate_s"), traj_s=time.time() - tr["t0w"], iterations=tr.get("iterations"),
        s_per_iter=s_per_iter(tr), loadavg=tr.get("loadavg"), other_load=tr.get("other_load"),
        sm=sm_mhz(tr["t0w"], tr["t0w"] + (tr.get("design_s") or 0)), terminated=row.get("terminated") or None,
        accepted_total=acc, csv_row=row, err=tr.get("err"))


if packed:
    # BindCraft 2 packs the card itself, from `launch_campaign` and not from `run_campaign`, so the campaign runs
    # in its own process tree and there is no single trajectory in this one to hook. One worker goes through the
    # same driver as N, so a concurrency A/B moves the worker count and nothing else.
    # BindCraft 2 lowers an explicit worker count that the card's free memory will not hold, so the arm reports the
    # plan it actually ran, not the one it asked for.
    from bindcraft.design_workers import plan_design_workers  # noqa: E402
    from bindcraft.campaign import design_residue_count  # noqa: E402
    plan = plan_design_workers(settings, design_residue_count(settings), int(settings["max_trajectories"]))
    settings_path = a.out / f"packed_settings_s{a.seed}_w{a.workers}.json"
    settings_path.write_text(json.dumps(settings, default=str, sort_keys=True))
    t0, t0w, load0 = time.monotonic(), time.time(), os.getloadavg()[0]
    done = subprocess.run([sys.executable, "-u", "-m", "bindcraft.cli", "design", str(settings_path)],
                          cwd=str(a.bc2), env={**os.environ, "CUDA_VISIBLE_DEVICES":
                                               os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    wall = time.monotonic() - t0
    per = campaign_trajectories(project)
    iters = sum(p["iterations"] or 0 for p in per)
    log(ev="campaign", workers=a.workers, planned_workers=len(plan), plan=plan,
        workers_seen=sorted({p.get("worker") for p in per if p.get("worker") is not None}), wall_s=wall, trajectories=per, iterations=iters,
        wall_per_iter=wall / iters if iters else None, sm=sm_mhz(t0w, time.time()),
        loadavg=[load0, os.getloadavg()[0]], rc=done.returncode,
        err=None if done.returncode == 0 else f"bindcraft.cli design exited {done.returncode}")
    log(ev="end")
    raise SystemExit(done.returncode)

traj = hook(campaign, project, emit)
af2, mpnn = model_weights()
try:
    campaign.run_campaign(settings, str(project), af2_weights=af2, mpnn_weights=mpnn)
except Exception:
    if traj:
        traj[-1]["err"] = traceback.format_exc()[-3000:]
    else:
        log(ev="crash", err=traceback.format_exc()[-3000:])
if traj:
    emit(len(traj) - 1)
log(ev="end")
