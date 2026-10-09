#!/usr/bin/env python3
"""Per-op census of ONE real BindCraft 2 design iteration, on the chip, at a recorded clock.

The speed unit for BindCraft 2 is seconds per design iteration: one AF2 forward plus backward
issued by `sequence_gradients`. This script runs BindCraft 2's own shipped campaign exactly as
`perf/spd/bench.py --model bindcraft2` does, lets the first few iterations warm and compile, then
instruments iteration `--at` and nothing else.

The instrument is `perf/hallgrad/census.py`'s Recorder, which this reuses rather than copies: it
wraps every `ttnn.<op>` and records only the outermost call, with the autograd verb that issued it.
Each call is timed between two `synchronize_device` calls ("synced" = device time + one dispatch +
one sync round trip). With `--replay R` the call is also replayed R times back to back with one
sync at the end, which hides dispatch and is the honest per-call device time for anything longer
than its enqueue. Replay is off by default: it allocates inside a live trajectory, where the tape
is already holding its retained set, and an OOM here costs the whole run.

The synced sum overstates the iteration (one sync round trip per call). The record carries both it
and the uninstrumented iteration time measured on the same chip in the same process, so the
inflation is a number in the output rather than an assumption.

    TT_VISIBLE_DEVICES=3 python perf/spd/bc2_census.py --out RUN/bc2cen --bc2 ~/bcx_e2e/bc2 --at 4

Output: <out>/census.json (every call record, aggregated by signature, with the clock) and a table
on stdout. BindCraft 2 runs trajectory-only when no MPNN weights are set, which is what we want:
the census is of the part that runs on the card.
"""
import argparse
import collections
import glob
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_e2e/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--at", type=int, default=4,
                help="instrument this sequence_gradients call (1-based); earlier ones warm and compile")
ap.add_argument("--replay", type=int, default=0, help="extra back-to-back replays per call (0 = off)")
ap.add_argument("--wrap", choices=("census", "all"), default="all",
                help="census: census.py's op list only; all: every public ttnn callable, including "
                     "ttnn.experimental, ttnn.transformer and the host<->device transfers")
ap.add_argument("--profile", type=int, default=3,
                help="run this (uninstrumented) gradient call under cProfile; 0 = off")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


# A tree rsynced to a remote box has no .git, so the launcher names the sha it copied.
SHA = git("rev-parse", "HEAD") or os.environ.get("SPD_TREE_SHA", "")
DIRTY = bool(git("status", "--porcelain", "--untracked-files=no"))
ENV = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("TT_BIO_", "PROTENIX_", "TT_METAL_"))}

# AICLK of every node, from sysfs, every 0.5 s. A dead ARC answers 0xFFFFFFFF without raising, so
# anything outside 100..3000 MHz is dropped as a non-reading (bench.py does the same).
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


def clock(t0, t1):
    win = [r for ts, r in samples if t0 <= ts <= t1]
    out = {}
    for n in NODES:
        v = sorted(r[n] for r in win if n in r)
        if v:
            out[n] = dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v))
    return out


sys.path.insert(0, str(a.bc2))
from perf.hallgrad.census import CLASS, Recorder  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
import tt_bio  # noqa: E402
from tt_bio import bindcraft2 as bc2, tenstorrent as TT  # noqa: E402


class LazyRecorder(Recorder):
    """Recorder against tt-bio's own open device, which exists only once the campaign has opened it."""

    @property
    def device(self):
        return TT._device

    @device.setter
    def device(self, _v):  # Recorder.__init__ assigns it; the property is the real answer
        pass


project = a.out / f"campaign_s{a.seed}"
example = a.bc2 / "examples" / "pdl1.json"
settings = cleaned_campaign_settings(read_settings(str(example), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={project}",
     f"binder_lengths=[{a.binder}]"])))

HEAD = dict(sha=SHA, dirty=DIRTY, engine=str(Path(tt_bio.__file__).parent), host=socket.gethostname(),
            chip=os.environ.get("TT_VISIBLE_DEVICES"), env=ENV, binder_length=a.binder,
            campaign_seed=a.seed, at=a.at, replay=a.replay, wrap=a.wrap, profile=a.profile, argv=sys.argv,
            nodes=NODES)

rec = LazyRecorder(__import__("ttnn"), None, a.replay)
rec.rec = []

# Ops outside census.py's list (the fused SDPA, ttnn.experimental kernels, to_torch/from_torch) are
# dispatched async; their device time lands inside the NEXT wrapped call's leading sync, before its
# clock starts, so a census that does not wrap them cannot see them at all. The first WH census
# (wrap=census) accounted for 5.0 s of a 12.5 s iteration; this is how the rest gets a name.
SKIP = ("_", "get_", "set_", "open_", "close_", "enable", "disable", "is_", "synchronize", "Read",
        "Dump", "manage", "register", "dump", "load_", "create_", "query", "device", "num_", "list_")


def wrap_everything():
    import ttnn
    done = set()
    for mod_name in ("", "experimental", "transformer"):
        mod = getattr(ttnn, mod_name) if mod_name else ttnn
        for n in dir(mod):
            fn = getattr(mod, n, None)
            if (not callable(fn) or isinstance(fn, type) or n.startswith(SKIP)
                    or getattr(fn, "__module__", "") in ("typing", "builtins") or id(fn) in done):
                continue
            label = f"{mod_name}.{n}" if mod_name else n
            if not mod_name and n in rec.orig:
                continue
            done.add(id(fn))
            rec.orig[label] = fn
            setattr(mod, n, rec._wrap(label, fn))


def profiled(fn):
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    t0 = time.monotonic()
    pr.enable()
    try:
        return fn()
    finally:
        pr.disable()
        wall = time.monotonic() - t0
        for key in ("tottime", "cumulative"):
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(60)
            (a.out / f"profile_{key}.txt").write_text(f"iteration {state['n']} wall {wall:.3f} s\n" + buf.getvalue())
        pr.dump_stats(str(a.out / "profile.pstats"))
cls = bc2.design_model_class()
real_grads = cls.sequence_gradients
state = dict(n=0, uninstrumented=[], instrumented=None, cl=None)


def sequence_gradients(self, *args, **kwargs):
    state["n"] += 1
    if state["n"] != a.at:
        t0 = time.monotonic()
        try:
            if state["n"] == a.profile:
                return profiled(lambda: real_grads(self, *args, **kwargs))
            return real_grads(self, *args, **kwargs)
        finally:
            state["uninstrumented"].append(dict(n=state["n"], s=time.monotonic() - t0,
                                                profiled=state["n"] == a.profile))
    rec.install()
    if a.wrap == "all":
        wrap_everything()
    rec.active = True
    t0 = time.monotonic()
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        t1 = time.monotonic()
        rec.active = False
        state["instrumented"] = t1 - t0
        state["cl"] = clock(t0, t1)
        write()
        os._exit(0)


def write():
    agg = collections.defaultdict(lambda: dict(calls=0, synced_s=0.0, replay_s=0.0, replays=0,
                                               flops=0.0, bytes=0.0))
    for r in rec.rec:
        k = (r["name"], r["origin"], r["sig"], r["out"])
        d = agg[k]
        d["calls"] += 1
        d["synced_s"] += r["synced_s"]
        d["flops"] += r["flops"]
        d["bytes"] += r["bytes"]
        if r["replay_s"] is not None:
            d["replay_s"] += r["replay_s"]
            d["replays"] += 1
    rows = []
    for (name, origin, sig, out), d in agg.items():
        rows.append(dict(name=name, origin=origin, sig=sig, out=out, cls=CLASS.get(name, "other"), **d))
    rows.sort(key=lambda r: -r["synced_s"])
    total = sum(r["synced_s"] for r in rows)
    doc = dict(**HEAD, iteration_s=state["instrumented"], uninstrumented=state["uninstrumented"],
               aiclk=state["cl"], synced_total_s=total, ops=rows, calls=len(rec.rec))
    (a.out / "census.json").write_text(json.dumps(doc, indent=1))

    warm = [u["s"] for u in state["uninstrumented"]]
    print(f"\nBindCraft 2 census: iteration {a.at}, {len(rec.rec)} outermost ttnn calls, "
          f"synced sum {total:.3f} s over an instrumented iteration of {state['instrumented']:.3f} s")
    print(f"uninstrumented iterations: {['%.3f' % w for w in warm]} s   AICLK {state['cl']}")
    print(f"{'op':34} {'origin':22} {'calls':>6} {'s':>8} {'share':>7}  shape")
    cum = 0.0
    for r in rows[:40]:
        cum += r["synced_s"]
        print(f"{r['name'][:33]:34} {r['origin'][:21]:22} {r['calls']:6d} {r['synced_s']:8.3f} "
              f"{r['synced_s'] / total * 100:6.1f}%  {r['sig'][:70]}")
    by = collections.defaultdict(float)
    for r in rows:
        by[r["cls"]] += r["synced_s"]
    print("\nby class: " + ", ".join(f"{k} {v:.3f}s ({v / total * 100:.1f}%)"
                                     for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
    print(f"wrote {a.out / 'census.json'}", flush=True)


cls.sequence_gradients = sequence_gradients
weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),
                                         ("mpnn_weights", "JAPANFOLD_BC2_MPNN_WEIGHTS")) if os.environ.get(v)}
print(json.dumps(dict(ev="start", **HEAD))[:2000], flush=True)
with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if "af2_weights" in weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
print(f"campaign ended before iteration {a.at} (only {state['n']} gradient calls)", flush=True)
os._exit(1)
