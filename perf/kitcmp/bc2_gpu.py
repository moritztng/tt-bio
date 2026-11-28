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
--set takes a BindCraft 2 setting for one GPU arm (use_cueq=true, attention_backend=cudnn); XLA flags go in
BC2_XLA_EXTRA, which BindCraft 2 appends to XLA_FLAGS itself. The start record carries the resolved settings and env.

BC4's quality reference (state/bc4/QUALITY-REF-REQUEST.json) runs the same harness for a different purpose: whole
campaigns at several seeds, graded by perf/spd/bc2_quality.py rather than timed. That needs another example file, a
trajectory budget the campaign always spends, and the example's own binder lengths, so --example, --max-traj,
--final-designs and --binder-len example exist. Every default is the timing run's, so the protocol command is
unchanged.

--gate DIR --slot J is perf/kitcmp/bc4_pack.py's: several of these processes share one GPU and DIR/phase ("<name> <n>
<w>") says which run now. A process whose slot is below n runs w trajectories, touches DIR/done.<name>.<J> and holds
(outside every timed window) until the phase changes; "end" emits the last trajectory and exits.
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
ap.add_argument("--final-designs", type=int, default=None,
                help="number_of_final_designs; raise it so a campaign always spends its whole trajectory budget "
                     "instead of stopping early on an accepted design")
ap.add_argument("--gate", type=Path, help="phase directory of bc4_pack.py (see above)")
ap.add_argument("--slot", type=int, default=0)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.bc2))
import jax  # noqa: E402
import bindcraft.campaign as campaign  # noqa: E402
from bindcraft.model_weights import model_weights  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from bc2_traj import cpu_model, hook, s_per_iter, table_row  # noqa: E402  (beside this file, here and on the box)

project = a.out / f"campaign_s{a.seed}"
overrides = [f"campaign_seed={a.seed}", f"max_trajectories={a.max_traj or 1 + a.warm}",
             f"project_folder={project}", "auto_multi_gpu=false"]
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
        t0w=tr["t0w"], phase=tr.get("phase"), s_per_iter=s_per_iter(tr), loadavg=tr.get("loadavg"), other_load=tr.get("other_load"),
        sm=sm_mhz(tr["t0w"], tr["t0w"] + (tr.get("design_s") or 0)), terminated=row.get("terminated") or None,
        accepted_total=acc, csv_row=row, err=tr.get("err"))


def gated(run):
    """Hold between trajectories until the current phase gives this slot work. Wraps the hook's timer, so no wait is
    ever inside design_s."""
    st = dict(phase=None, left=0)

    def go(*args, **kw):
        while not st["left"]:
            if st["phase"]:
                (a.gate / f"done.{st['phase']}.{a.slot}").touch()
            while True:
                with contextlib.suppress(OSError, ValueError):
                    name, n, w = (a.gate / "phase").read_text().split()
                    if name != st["phase"]:
                        break
                time.sleep(0.2)
            if name == "end":
                if traj:
                    emit(len(traj) - 1)
                log(ev="end")
                os._exit(0)
            st.update(phase=name, left=int(w) if a.slot < int(n) else 0)
        st["left"] -= 1
        try:
            return run(*args, **kw)
        finally:
            traj[-1]["phase"] = st["phase"]
    return go


traj = hook(campaign, project, emit)
if a.gate:
    campaign.run_trajectory = gated(campaign.run_trajectory)
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
