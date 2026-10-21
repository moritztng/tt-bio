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
import gc
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
ap.add_argument("--dump", type=int, default=0,
                help="save the first N gradient calls' results to dump_<n>.npz, so two arms can be compared "
                     "by magnitude when their digests differ")
ap.add_argument("--gc-freeze", type=int, default=0,
                help="gc.freeze() at the start of this call (0 = never): the host objects alive by then, "
                     "trunks and weights, leave the cyclic collector's generation-2 walk")
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
state = dict(n=0, uninstrumented=[], instrumented=None, cl=None, loads=0)

# A trunk loads on the first gradient call that samples its checkpoint (~13 s on a Wormhole Galaxy
# host: npz read, remap, upload), then stays resident. Five design models means five such calls
# per process at random iterations, so every iteration records how many it paid, and the host
# load beside it: on this workload the host is half the iteration and a loaded box doubles it.
real_trunk_init = bc2._Trunk.__init__


def trunk_init(self, *args, **kwargs):
    state["loads"] += 1
    return real_trunk_init(self, *args, **kwargs)


bc2._Trunk.__init__ = trunk_init

# Wall of every tt-bio callback, by stack and verb, so each iteration splits into the time spent
# inside tt-bio (device + dispatch + tape) and the rest, which is BindCraft 2's own JAX on the host.
cb = collections.Counter()


# Wall spent blocked on the card inside callbacks (readbacks and syncs). A callback minus this is
# the host's enqueue time: exact when the command queue never fills, an upper bound when it does.
blocked = collections.Counter()


def blocking(fn):
    def run(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            blocked["s"] += time.monotonic() - t0
    return run


# Wall of Python's cyclic collector, by generation. A live tape is millions of host objects, so a
# generation-2 pass can stall the enqueue of whatever seam it lands in.
gcwall = collections.Counter()
_gc_t0 = [0.0]


def _gc_cb(phase, info):
    if phase == "start":
        _gc_t0[0] = time.monotonic()
    else:
        gcwall[f"gen{info['generation']}_s"] += time.monotonic() - _gc_t0[0]
        gcwall[f"gen{info['generation']}_n"] += 1


gc.callbacks.append(_gc_cb)

_ttnn = __import__("ttnn")
for _name in ("to_torch", "from_device", "synchronize_device"):
    setattr(_ttnn, _name, blocking(getattr(_ttnn, _name)))


def timed(name, fn):
    def run(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            cb[name] += time.monotonic() - t0
    return run


for _cls in (bc2.EvoformerOnDevice, bc2.ExtraMsaOnDevice, bc2.TemplateOnDevice):
    for _verb in ("_primal", "_taped", "_backward"):
        setattr(_cls, _verb, timed(f"{_cls.__name__}.{_verb}", getattr(_cls, _verb)))


def loadavg():
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


def arrays(out):
    """Every numeric array in a round's result, in a fixed order.

    BC2 returns its `StructurePredictions` as a plain object, so `tree_leaves` hands it over
    whole and `np.asarray` makes it an object array whose bytes are pointers: the old digest
    read different on two bit-identical runs, and the dump skipped the predictions entirely.
    Non-array leaves are opened through their attributes instead.
    """
    import jax
    import numpy as np

    def walk(x, depth=0):
        if x is None or depth > 4:
            return
        try:
            a = np.asarray(x)
        except Exception:                                                     # noqa: BLE001
            a = None
        if a is not None and a.dtype.kind in "biufc":
            yield a
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from walk(y, depth + 1)
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                yield from walk(x[k], depth + 1)
        elif hasattr(x, "__dict__"):
            for k in sorted(vars(x)):
                yield from walk(vars(x)[k], depth + 1)

    for leaf in jax.tree_util.tree_leaves(out):
        yield from walk(leaf)


def gsha(out):
    """sha256[:16] over every numeric array of the round's result: the eager/trace parity check."""
    if out is None:
        return None
    import hashlib
    h = hashlib.sha256()
    for a in arrays(out):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def dump(out, path):
    """Every numeric array of a round's result, in `arrays` order, as float64 (leaf_<i> in an npz)."""
    import numpy as np
    np.savez(path, **{f"leaf_{i}": np.asarray(a, dtype=np.float64) for i, a in enumerate(arrays(out))})


def sequence_gradients(self, *args, **kwargs):
    state["n"] += 1
    if state["n"] == a.gc_freeze:
        gc.freeze()
    loads0, load1 = state["loads"], loadavg()
    if state["n"] != a.at:
        cb.clear()
        blocked.clear()
        gcwall.clear()
        t0 = time.monotonic()
        out = None
        try:
            if state["n"] == a.profile:
                out = profiled(lambda: real_grads(self, *args, **kwargs))
            else:
                out = real_grads(self, *args, **kwargs)
            return out
        finally:
            if out is not None and state["n"] <= a.dump:
                dump(out, a.out / f"dump_{state['n']}.npz")
            wall = time.monotonic() - t0
            state["uninstrumented"].append(dict(n=state["n"], s=wall, gsha=gsha(out),
                                                callbacks={k: round(v, 4) for k, v in cb.items()},
                                                host_jax_s=round(wall - sum(cb.values()), 4),
                                                blocked_s=round(blocked["s"], 4),
                                                gc={k: round(v, 4) for k, v in gcwall.items()},
                                                profiled=state["n"] == a.profile,
                                                trunk_loads=state["loads"] - loads0,
                                                load1=load1, ncpu=os.cpu_count()))
    if a.wrap == "all":
        rec.install_all()
    else:
        rec.install()
    rec.active = True
    t0 = time.monotonic()
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        t1 = time.monotonic()
        rec.active = False
        state["instrumented"] = t1 - t0
        state["instrumented_loads"] = state["loads"] - loads0
        state["instrumented_load1"] = load1
        state["cl"] = clock(t0, t1)
        write()
        os._exit(0)


def write(partial=False):
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
               iteration_trunk_loads=state.get("instrumented_loads"), iteration_load1=state.get("instrumented_load1"),
               aiclk=state["cl"], synced_total_s=total, ops=rows, calls=len(rec.rec), partial=partial)
    ag = sys.modules.get("tt_bio.autograd")
    if ag is not None:
        doc.update(rtrace_stats=dict(ag.RTRACE_STATS), rtrace_checks=list(ag.RTRACE_CHECKS))
    (a.out / "census.json").write_text(json.dumps(doc, indent=1))

    warm = [u["s"] for u in state["uninstrumented"]]
    if partial:
        # The campaign ended before iteration `at` (a stage gate rejected the trajectory). There is no
        # instrumented iteration, but the per-call times and gradient digests are the point of an A/B arm.
        print(f"\nBindCraft 2 census (PARTIAL): campaign ended after {state['n']} gradient calls, "
              f"no iteration {a.at}")
        print(f"uninstrumented iterations: {['%.3f' % w for w in warm]} s", flush=True)
        return
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
write(partial=True)
os._exit(1)
