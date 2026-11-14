#!/usr/bin/env python3
"""Seconds per BindCraft 2 design stage, timed at the stage seam rather than read off losses.csv.

BindCraft 2 writes a trajectory's losses.csv in one burst, so a row's arrival time says nothing about when
its iteration ran. This times the seams instead: `build_stage_plan` gives the stage order and each stage's
iteration budget, every `sequence_gradients` call is stamped, and `run_mutation_polish` is stamped whole.
A gradient call belongs to the stage whose budget covers its index, which is exact because a stage gate ends
the trajectory rather than shortening a stage.

    TT_VISIBLE_DEVICES=0 python perf/spd/bc2_stage_time.py --out RUN/stage --bc2 ~/bcx_shipped/bc2 --trajectories 3

Output: <out>/stage_time.json — per stage: calls, total s, median s/it; per trajectory: the stage it ended in.
Runs trajectory-only (no ProteinMPNN weights), because validation is not part of the design stages.
"""
import argparse, collections, json, os, socket, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_shipped/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--trajectories", type=int, default=3)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(a.bc2))
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
import bindcraft.trajectory as T  # noqa: E402
import tt_bio  # noqa: E402
from tt_bio import bindcraft2 as bc2  # noqa: E402


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", f"max_trajectories={a.trajectories}", f"project_folder={project}",
     f"binder_lengths=[{a.binder}]"])))

calls, plan, traj, cur = [], [], [], {"n": 0}

real_plan = T.build_stage_plan


def build_stage_plan(*args, **kw):
    sp = real_plan(*args, **kw)
    if not plan:
        plan.extend(dict(name=s.name, iterations=int(s.sequence_optimizer.iterations)) for s in sp)
    cur["n"] += 1
    cur["call0"] = len(calls)
    cur["t0"] = time.monotonic()
    return sp


real_mutate = T.run_mutation_polish


def run_mutation_polish(*a_, **kw):
    t0 = time.monotonic()
    try:
        return real_mutate(*a_, **kw)
    finally:
        traj[-1]["mutate_s"] = time.monotonic() - t0


cls = bc2.design_model_class()
real_grads = cls.sequence_gradients


def sequence_gradients(self, *a_, **kw):
    if not traj or traj[-1]["n"] != cur["n"]:
        traj.append(dict(n=cur["n"], call0=cur["call0"], t0=cur["t0"], mutate_s=None))
    t0 = time.monotonic()
    try:
        return real_grads(self, *a_, **kw)
    finally:
        calls.append(dict(traj=cur["n"], s=time.monotonic() - t0, t=time.monotonic()))


T.build_stage_plan, T.run_mutation_polish, cls.sequence_gradients = build_stage_plan, run_mutation_polish, sequence_gradients


def stage_of(i_in_traj):
    """The stage a trajectory's i-th (0-based) gradient call belongs to, from the plan's budgets."""
    k = 0
    for s in plan:
        if i_in_traj < k + s["iterations"]:
            return s["name"]
        k += s["iterations"]
    return "beyond-plan"


def write():
    per = collections.defaultdict(list)
    for t in traj:
        mine = [c for c in calls if c["traj"] == t["n"]]
        for i, c in enumerate(mine):
            per[stage_of(i)].append(c["s"])
    stages = {k: dict(calls=len(v), total_s=round(sum(v), 3), median_s=round(statistics.median(v), 4),
                      min_s=round(min(v), 4), max_s=round(max(v), 4)) for k, v in per.items()}
    mut = [t["mutate_s"] for t in traj if t["mutate_s"]]
    doc = dict(host=socket.gethostname(), sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
               engine=str(Path(tt_bio.__file__).parent), chip=os.environ.get("TT_VISIBLE_DEVICES"), argv=sys.argv,
               binder=a.binder, seed=a.seed, plan=plan, stages=stages,
               mutate=dict(calls=len(mut), total_s=round(sum(mut), 3),
                           median_s=round(statistics.median(mut), 3)) if mut else None,
               trajectories=[dict(n=t["n"], gradient_calls=sum(1 for c in calls if c["traj"] == t["n"]),
                                  mutate_s=t["mutate_s"]) for t in traj],
               gradient_calls=len(calls))
    (a.out / "stage_time.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(plan=plan, stages=stages, mutate=doc["mutate"]), indent=1), flush=True)
    print(f"wrote {a.out / 'stage_time.json'}", flush=True)


print(json.dumps(dict(ev="start", argv=sys.argv))[:400], flush=True)
try:
    af2 = os.environ["JAPANFOLD_BC2_AF2_WEIGHTS"]
    with bc2.campaign_predictor(checkpoints=af2):
        bc2.run_campaign(settings, str(project), trajectories_per_card=1, af2_weights=af2)
finally:
    write()
os._exit(0)
