#!/usr/bin/env python3
"""Where a BindCraft 2 mutate round's seconds go, split at the seams inside the round.

The mutate stage is 15 rounds of `run_sequence_mutation_stage` (bindcraft/trajectory.py:155) and one
trailing refold. A round proposes one single-residue mutation, folds the candidate forward-only through
`design_model.predict`, scores it on the host and keeps the better of the two sequences. This stamps
each of those four seams per round, counts the device trunk callbacks inside the fold, and separates the
loop from the trailing refold and the stage filters.

    TT_VISIBLE_DEVICES=0 python perf/spd/bc2_mutate_split.py --out RUN/msplit --bc2 ~/bcx_shipped/bc2 \
        --trajectories 3

Output: <out>/mutate_split.json — per round propose/fold/loss/select seconds, the device seconds inside
each fold (sum of the three on-card stacks' forward callbacks) and the host remainder, plus the trailing
refold and filter time. Runs trajectory-only, like perf/spd/bc2_stage_time.py.
"""
import argparse, glob, json, os, socket, statistics, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_shipped/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--trajectories", type=int, default=3)
ap.add_argument("--hang-s", type=float, default=600.0, help="kill and reset after this long without progress")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(a.bc2))
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
import bindcraft.campaign as C  # noqa: E402
import bindcraft.trajectory as T  # noqa: E402
import bindcraft.sequence_optimization as SO  # noqa: E402
import tt_bio  # noqa: E402
from tt_bio import bindcraft2 as bc2  # noqa: E402


def git(*args, repo=ROOT):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


# AICLK per node every 0.5 s, as bench.py samples it, so a trajectory here carries the clock it ran at
# (PROTOCOL rule 6). bench.py parses its arguments at import, so its sampler cannot be imported.
NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))
samples = []


def _sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    row[n] = v
            except Exception:
                pass
        samples.append((time.monotonic(), row))
        time.sleep(0.5)


threading.Thread(target=_sampler, daemon=True).start()

# A chip that wedges blocks inside one ttnn call while holding the GIL, so nothing in this process can
# notice. A separate watchdog does: it kills this process when the heartbeat (touched at every gradient
# call and mutate seam) goes quiet for --hang-s, then resets the chip, so a hang costs one trajectory
# instead of a borrowed card for hours. Finished trajectories are already on disk (write() runs after each).
HEARTBEAT = a.out / "heartbeat"
HEARTBEAT.touch()
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
subprocess.Popen([sys.executable, "-c", f"""
import os, signal, sys, time, subprocess
pid, hb, hang = {os.getpid()}, {str(HEARTBEAT)!r}, {a.hang_s}
while os.path.exists(f"/proc/{{pid}}") and open(f"/proc/{{pid}}/stat").read().split()[2] != "Z":
    if time.time() - os.path.getmtime(hb) > hang:
        print(f"watchdog: no progress for {{hang}} s, killing {{pid}} and resetting chip {CHIP}", flush=True)
        os.kill(pid, signal.SIGKILL)
        while os.path.exists(f"/proc/{{pid}}") and open(f"/proc/{{pid}}/stat").read().split()[2] != "Z":
            time.sleep(1)
        subprocess.run([os.path.expanduser("~/.local/bin/tt-smi"), "-r", {CHIP!r}], timeout=300)
        break
    time.sleep(10)
"""], start_new_session=True)


def beat():
    os.utime(HEARTBEAT)


def clock(t0, t1):
    win = [r for ts, r in samples if t0 <= ts <= t1]
    out = {}
    for n in NODES:
        v = sorted(r[n] for r in win if n in r)
        if v:
            out[str(n)] = dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v))
    return out


project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", f"max_trajectories={a.trajectories}", f"project_folder={project}",
     f"binder_lengths=[{a.binder}]"])))

# One row per mutate round; `dev_s` accumulates the forward device callbacks of the fold in that row.
rounds, stages, trajectories = [], [], []
state = {"in_mutate": False, "dev_s": 0.0, "dev_n": 0, "traj": 0, "gradient_calls": 0}


def card(fn):
    """Stamp a device-side forward callback (`_primal` of each on-card stack)."""
    def wrapped(self, *args, **kw):
        t0 = time.monotonic()
        try:
            return fn(self, *args, **kw)
        finally:
            state["dev_s"] += time.monotonic() - t0
            state["dev_n"] += 1
    return wrapped


for cls_name in ("EvoformerOnDevice", "ExtraMsaOnDevice", "TemplateOnDevice"):
    stack = getattr(bc2, cls_name)
    stack._primal = card(stack._primal)

model_cls = bc2.design_model_class()
real_predict, real_propose, real_select = model_cls.predict, SO.SemigreedySequenceSampler.propose_sequence_mutation, \
    SO.SemigreedySequenceSampler.select_best_sequence
real_loss, real_mutate, real_loop = T.weighted_design_loss, T.run_mutation_polish, T.run_sequence_mutation_stage


def stamp(key):
    """Time a call into the current mutate round's row, counting device seconds for folds.

    A round runs propose (every round but the first, which scores the sequence it inherits), fold, loss,
    select. So a proposal always opens a row, and any other seam opens one only when this round already
    has that seam -- which is how the trailing refold, a second fold after the loop's last select, lands
    in a row of its own.
    """
    def decorate(fn):
        def wrapped(*args, **kw):
            beat()
            if not state["in_mutate"]:
                return fn(*args, **kw)
            if key == "propose_s" or not rounds or key in rounds[-1]:
                rounds.append({"traj": state["traj"], "round": len([r for r in rounds if r["traj"] == state["traj"]]) + 1})
            row, dev0, n0, t0 = rounds[-1], state["dev_s"], state["dev_n"], time.monotonic()
            try:
                return fn(*args, **kw)
            finally:
                row[key] = time.monotonic() - t0
                if key == "fold_s":
                    row["fold_device_s"] = state["dev_s"] - dev0
                    row["fold_device_calls"] = state["dev_n"] - n0
        return wrapped
    return decorate


model_cls.predict = stamp("fold_s")(real_predict)
SO.SemigreedySequenceSampler.propose_sequence_mutation = stamp("propose_s")(real_propose)
SO.SemigreedySequenceSampler.select_best_sequence = stamp("select_s")(real_select)
T.weighted_design_loss = stamp("loss_s")(real_loss)


def run_sequence_mutation_stage(*args, **kw):
    t0 = time.monotonic()
    try:
        return real_loop(*args, **kw)
    finally:
        stages[-1]["loop_s"] = time.monotonic() - t0


def run_mutation_polish(*args, **kw):
    state["traj"] += 1
    stages.append({"traj": state["traj"], "loop_s": None})
    state["in_mutate"], t0 = True, time.monotonic()
    try:
        return real_mutate(*args, **kw)
    finally:
        stages[-1]["stage_s"] = time.monotonic() - t0
        state["in_mutate"] = False


# `campaign.py:21` binds its own reference to run_trajectory, so the campaign's name is the one to wrap.
real_trajectory, real_grads = C.run_trajectory, model_cls.sequence_gradients


def sequence_gradients(self, *args, **kw):
    state["gradient_calls"] += 1
    beat()
    return real_grads(self, *args, **kw)


def run_trajectory(*args, **kw):
    """One `design_s` per trajectory, the unit PROTOCOL rule 1 defines, with its clock and host load.

    `iterations` is the rows the trajectory adds to losses.csv: one per gradient call and one per mutate
    round, counted here rather than read back off the file.
    """
    g0, m0, la0, t0 = state["gradient_calls"], len(rounds), os.getloadavg()[0], time.monotonic()
    try:
        return real_trajectory(*args, **kw)
    finally:
        t1 = time.monotonic()
        grads, muts = state["gradient_calls"] - g0, len([r for r in rounds[m0:] if "select_s" in r])
        trajectories.append(dict(n=len(trajectories) + 1, design_s=t1 - t0, gradient_calls=grads,
                                 mutate_rounds=muts, iterations=grads + muts,
                                 s_per_iter=(t1 - t0) / (grads + muts) if grads + muts else None,
                                 aiclk=clock(t0, t1), loadavg=[round(la0, 2), round(os.getloadavg()[0], 2)]))
        write()


T.run_mutation_polish, T.run_sequence_mutation_stage = run_mutation_polish, run_sequence_mutation_stage
C.run_trajectory, model_cls.sequence_gradients = run_trajectory, sequence_gradients


def med(xs):
    return round(statistics.median(xs), 4) if xs else None


def write():
    full = [r for r in rounds if "fold_s" in r]
    # The trailing refold is the last fold of the stage: it is inside `run_mutation_polish` but outside the
    # loop, so it has no propose/select of its own.
    loop_rounds = [r for r in full if "select_s" in r]
    trailing = [r for r in full if "select_s" not in r]
    doc = dict(host=socket.gethostname(), sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
               engine=str(Path(tt_bio.__file__).parent), chip=os.environ.get("TT_VISIBLE_DEVICES"), argv=sys.argv,
               binder=a.binder, seed=a.seed, bc2_sha=git("rev-parse", "HEAD", repo=a.bc2),
               cpu=next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
               ncpu=os.cpu_count(), trajectories=trajectories,
               stage=dict(n=len(stages), stage_s=[s.get("stage_s") for s in stages],
                          loop_s=[s.get("loop_s") for s in stages]),
               summary=dict(
                   rounds=len(loop_rounds), trailing_folds=len(trailing),
                   round_fold_s=med([r["fold_s"] for r in loop_rounds]),
                   round_fold_device_s=med([r.get("fold_device_s", 0.0) for r in loop_rounds]),
                   round_fold_device_calls=med([r.get("fold_device_calls", 0) for r in loop_rounds]),
                   round_propose_s=med([r.get("propose_s", 0.0) for r in loop_rounds]),
                   round_loss_s=med([r.get("loss_s", 0.0) for r in loop_rounds]),
                   round_select_s=med([r.get("select_s", 0.0) for r in loop_rounds]),
                   trailing_fold_s=med([r["fold_s"] for r in trailing])),
               rounds=full)
    (a.out / "mutate_split.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(dict(stage=doc["stage"], summary=doc["summary"]), indent=1), flush=True)
    print(f"wrote {a.out / 'mutate_split.json'}", flush=True)


print(json.dumps(dict(ev="start", argv=sys.argv))[:400], flush=True)
try:
    af2 = os.environ["JAPANFOLD_BC2_AF2_WEIGHTS"]
    with bc2.campaign_predictor(checkpoints=af2):
        bc2.run_campaign(settings, str(project), trajectories_per_card=1, af2_weights=af2)
finally:
    write()
os._exit(0)
