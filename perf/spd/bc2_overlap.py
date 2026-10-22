#!/usr/bin/env python3
"""Where N interleaved BindCraft 2 trajectories on one chip spend their time, per thread.

Runs BindCraft 2's shipped PD-L1 campaign trajectory-only (no MPNN weights) with
`trajectories_per_card=N` and `max_trajectories=N`, so every trajectory runs concurrently.
Records, uninstrumented apart from timestamps:
  rounds.jsonl  one row per `sequence_gradients` call: slot, thread, t0, t1
  card.jsonl    one row per outermost `duotraj.card` region: slot, tag, thread, t_req, t_acq, t_rel
                (t_acq - t_req is time standing at the card door; N=1 has no gate and waits 0)
  aiclk.jsonl   every node's AICLK and the 1-min load, every 0.5 s, for the whole run
Once every slot has `--steady` rounds behind it, attaches py-spy twice (sudo, the box has
ptrace_scope=1): `--idle --native` for `--spy` s, then `--gil --native` for `--spy` s, raw
collapsed stacks per thread, so states can be attributed (GIL wait, card-door wait, ttnn
dispatch, device wait, XLA:CPU). The spy windows are written to spy.json and excluded from
the timing analysis, since a native sample stops the process.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 python perf/spd/bc2_overlap.py --out RUN --n 2
"""
import argparse, contextlib, glob, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_e2e/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--steady", type=int, default=20, help="rounds per slot before py-spy attaches")
ap.add_argument("--spy", type=float, default=60.0, help="seconds per py-spy window (0 = no py-spy)")
ap.add_argument("--rate", type=int, default=25)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.bc2))

from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from tt_bio import bindcraft2 as bc2, duotraj  # noqa: E402

T0 = time.monotonic()
now = lambda: time.monotonic() - T0  # noqa: E731
logs = {k: open(a.out / f"{k}.jsonl", "w", buffering=1) for k in ("rounds", "card", "aiclk")}
lock = threading.Lock()


def emit(k, **row):
    with lock:
        logs[k].write(json.dumps(row) + "\n")


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in glob.glob("/sys/class/tenstorrent/tenstorrent!*"))


def sampler():
    while True:
        row = {}
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    row[n] = v
            except Exception:
                pass
        emit("aiclk", t=round(now(), 3), clk=row, load=os.getloadavg()[0])
        time.sleep(0.5)


threading.Thread(target=sampler, daemon=True).start()

depth = threading.local()
real_card = duotraj.card


@contextlib.contextmanager
def card(slot="", tag=""):
    if getattr(depth, "n", 0):
        with real_card(slot, tag):
            yield
        return
    t_req = now()
    with real_card(slot, tag):
        t_acq = now()
        depth.n = 1
        try:
            yield
        finally:
            depth.n = 0
            emit("card", slot=slot, tag=tag, th=threading.current_thread().name, tid=threading.get_native_id(),
                 t_req=round(t_req, 4), t_acq=round(t_acq, 4), t_rel=round(now(), 4))


duotraj.card = card

cls = bc2.design_model_class()
real_grads = cls.sequence_gradients
rounds = {}
spy = {"windows": []}
PYSPY = os.environ.get("PYSPY") or shutil.which("py-spy") or os.path.expanduser("~/.local/bin/py-spy")


def spy_once(flag, name):
    out = a.out / f"spy_{name}.txt"
    t0 = now()
    r = subprocess.run(["sudo", "-n", PYSPY, "record", "--pid", str(os.getpid()), "--native", flag,
                        "--threads", "--format", "raw", "--rate", str(a.rate), "--duration",
                        str(int(a.spy)), "-o", str(out)], capture_output=True, text=True)
    spy["windows"].append(dict(name=name, t0=round(t0, 3), t1=round(now(), 3), rc=r.returncode,
                               err=r.stderr[-2000:]))
    (a.out / "spy.json").write_text(json.dumps(spy, indent=1))


def spy_both():
    spy_once("--idle", "idle")
    spy_once("--gil", "gil")


def sequence_gradients(self, *args, **kwargs):
    slot = duotraj.slot()
    t0 = now()
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        t1 = now()
        emit("rounds", slot=slot, th=threading.current_thread().name, tid=threading.get_native_id(), t0=round(t0, 4), t1=round(t1, 4))
        with lock:
            rounds[slot] = rounds.get(slot, 0) + 1
            go = (a.spy > 0 and not spy.get("started") and len(rounds) >= a.n
                  and min(rounds.values()) >= a.steady)
            if go:
                spy["started"] = True
        if go:
            threading.Thread(target=spy_both, daemon=True).start()


cls.sequence_gradients = sequence_gradients

project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", f"max_trajectories={a.n}", f"project_folder={project}",
     f"binder_lengths=[{a.binder}]"])))
sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
head = dict(ev="start", sha=sha, n=a.n, seed=a.seed, binder=a.binder, pid=os.getpid(), argv=sys.argv,
            env={k: v for k, v in os.environ.items() if k.startswith("TT_")}, load=os.getloadavg(),
            ncpu=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)))
(a.out / "head.json").write_text(json.dumps(head, indent=1))
print(json.dumps(head), flush=True)
weights = os.environ.get("JAPANFOLD_BC2_AF2_WEIGHTS")
with bc2.campaign_predictor(**({"checkpoints": weights} if weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=a.n,
                     **({"af2_weights": weights} if weights else {}))
(a.out / "done.json").write_text(json.dumps(dict(wall=round(now(), 3), rounds=rounds, load=os.getloadavg())))
print("done", rounds, flush=True)
os._exit(0)
