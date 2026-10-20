#!/usr/bin/env python3
"""On-card timeline of warm BindCraft 2 iterations: every device program's start and end.

Every per-kernel number BC4 has planned from is host-synced per call. This reads the card's own
clock instead, through tt-metal's device profiler on a Tracy-enabled build of the same ttnn the
campaign runs (v0.68.0), with the in-process C++ post-processor and no CSV: after every few block
replays `ReadDeviceProfiler` drains the on-card marker buffers and `get_latest_programs_perf_data`
hands back one record per program (firmware and kernel start/end in cycles). Nothing goes to disk
but the compact records this script keeps.

Program names come from Python: every outermost public ttnn call records the device-operation ids
it consumed, and a program's `runtime_id` is that id. A traced block keeps the ids of its capture,
so its replayed programs carry the names of the eager calls that captured it.

Run it on a Tracy build (the launcher `perf/bc4_devgap/run.sh` sets the environment):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \\
        python perf/bc4_devgap/devprof.py --out RUN --at 4 --keep 2

A read stalls the host until the card drains, so a gap that spans a read is the instrument's and
is never counted: the analysis (`analyze.py`) measures gaps only inside one read's batch.
"""
import argparse
import gzip
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
ap.add_argument("--at", type=int, default=4, help="first kept sequence_gradients call (1-based)")
ap.add_argument("--keep", type=int, default=2, help="iterations kept from --at on; the process exits after")
ap.add_argument("--every", type=int, default=4, help="trace replays between profiler reads")
ap.add_argument("--ops", type=int, default=2500, help="eager programs between profiler reads")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


NODES = sorted(int(p.rsplit("!", 1)[1]) for p in __import__("glob").glob("/sys/class/tenstorrent/tenstorrent!*"))
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
    out = {}
    for n in NODES:
        v = sorted(r[n] for ts, r in samples if t0 <= ts <= t1 and n in r)
        if v:
            out[n] = dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v))
    return out


def loadavg():
    return Path("/proc/loadavg").read_text().split()[:3]


sys.path.insert(0, str(a.bc2))
import ttnn  # noqa: E402
from perf.hallgrad.census import Recorder  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
import tt_bio  # noqa: E402
from tt_bio import bindcraft2 as bc2, tenstorrent as TT  # noqa: E402

prof = ttnn._ttnn.profiler
op_id = ttnn._ttnn.get_device_operation_id
st = dict(n=0, depth=0, capturing=False, last_read_id=op_id(), replays=0, reads=0, read_s=0.0)
spans = []      # (first id, last id + 1, label, origin, iteration): one per outermost call that made programs
programs = []   # one compact row per profiled program of a kept iteration
reads = []      # (iteration, read index, host time, programs, callback label)
iters = []      # per sequence_gradients call: host wall, AICLK, load
ORIGIN_SKIP = ("taped_ttnn.py", "devprof.py", "census.py")


def origin():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "tt_bio" in fn and not fn.endswith(ORIGIN_SKIP):
            return f"{Path(fn).name}:{f.f_code.co_name}"
        f = f.f_back
    return "?"


def kept():
    return a.at <= st["n"] < a.at + a.keep


def read(tag):
    dev = TT._device
    if dev is None or st["capturing"]:
        return
    t0 = time.monotonic()
    ttnn.ReadDeviceProfiler(dev)
    data = prof.get_latest_programs_perf_data()
    st["read_s"] += time.monotonic() - t0
    st["last_read_id"] = op_id()
    st["reads"] += 1
    nprog = 0
    for chip, plist in data.items():
        for p in plist:
            nprog += 1
            if not kept():
                continue
            u, r = p.program_execution_uid, p.program_analyses_results
            fw, k = r.get("DEVICE FW DURATION [ns]"), r.get("DEVICE KERNEL DURATION [ns]")
            programs.append([st["n"], st["reads"], int(chip), u.runtime_id, u.trace_id, u.trace_id_counter,
                             p.core_count,
                             fw.start_timestamp if fw else None, fw.end_timestamp if fw else None,
                             fw.duration if fw else None,
                             k.start_timestamp if k else None, k.end_timestamp if k else None,
                             k.duration if k else None])
    reads.append([st["n"], st["reads"], round(t0, 6), nprog, tag])


def wrap(label, fn):
    def run(*args, **kwargs):
        if st["depth"]:
            return fn(*args, **kwargs)
        st["depth"] = 1
        i0 = op_id()
        try:
            return fn(*args, **kwargs)
        finally:
            st["depth"] = 0
            i1 = op_id()
            if i1 != i0:
                spans.append((i0, i1, label, origin(), st["n"]))
                if i1 - st["last_read_id"] >= a.ops:
                    read(f"ops:{label}")
    run.__wrapped__ = fn
    return run


def install():
    done = set()
    for mod_name in ("", "experimental", "transformer"):
        mod = getattr(ttnn, mod_name) if mod_name else ttnn
        for n in dir(mod):
            fn = getattr(mod, n, None)
            if (not callable(fn) or isinstance(fn, type) or n.startswith(Recorder.SKIP) or id(fn) in done
                    or getattr(fn, "__module__", "") in ("typing", "builtins")
                    or n in ("execute_trace", "begin_trace_capture", "end_trace_capture")):
                continue
            done.add(id(fn))
            setattr(mod, n, wrap(f"{mod_name}.{n}" if mod_name else n, fn))
    gi = wrap("getitem", ttnn.Tensor.__getitem__)
    ttnn.Tensor.__getitem__ = lambda self_, key: gi(self_, key)
    real_exec = ttnn.execute_trace

    def execute_trace(*args, **kwargs):
        out = real_exec(*args, **kwargs)
        st["replays"] += 1
        if st["replays"] % a.every == 0:
            read("replay")
        return out
    ttnn.execute_trace = execute_trace
    # A read syncs the device, which a trace capture forbids: none happens between begin and end.
    for verb, flag in (("begin_trace_capture", True), ("end_trace_capture", False)):
        def toggle(*args, _fn=getattr(ttnn, verb), _flag=flag, **kwargs):
            st["capturing"] = _flag
            return _fn(*args, **kwargs)
        setattr(ttnn, verb, toggle)


def at_exit_of(name, fn):
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            read(name)
    return run


install()
for _cls in (bc2.EvoformerOnDevice, bc2.ExtraMsaOnDevice, bc2.TemplateOnDevice):
    for _verb in ("_primal", "_taped", "_backward"):
        setattr(_cls, _verb, at_exit_of(f"{_cls.__name__}.{_verb}", getattr(_cls, _verb)))

HEAD = dict(sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
            host=socket.gethostname(), chip=os.environ.get("TT_VISIBLE_DEVICES"),
            env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("TT_BIO_", "TT_METAL_"))},
            ttnn=ttnn.__file__, binder_length=a.binder, campaign_seed=a.seed, at=a.at, keep=a.keep,
            every=a.every, ops=a.ops, argv=sys.argv, nodes=NODES)
PROG_COLS = ["iteration", "read", "chip", "runtime_id", "trace_id", "trace_id_counter", "core_count",
             "fw_start_cyc", "fw_end_cyc", "fw_ns", "k_start_cyc", "k_end_cyc", "k_ns"]


def write():
    keep_ids = {r[3] for r in programs}
    doc = dict(**HEAD, iterations=iters, reads=reads, program_cols=PROG_COLS, programs=programs,
               spans=[s for s in spans if any(i in keep_ids for i in range(s[0], s[1]))],
               read_s=st["read_s"], profiler_reads=st["reads"], replays=st["replays"])
    with gzip.open(a.out / "devprof.json.gz", "wt") as f:
        json.dump(doc, f)
    print(f"wrote {a.out / 'devprof.json.gz'}: {len(programs)} programs, {len(doc['spans'])} spans", flush=True)


cls = bc2.design_model_class()
real_grads = cls.sequence_gradients


def sequence_gradients(self, *args, **kwargs):
    read("pre")     # drain the host stretch before this call
    st["n"] += 1
    t0, l0 = time.monotonic(), loadavg()
    r0, rs0 = st["reads"], st["read_s"]
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        read("end")
        t1 = time.monotonic()
        iters.append(dict(n=st["n"], wall_s=t1 - t0, read_s=st["read_s"] - rs0, reads=st["reads"] - r0,
                          load_start=l0, load_end=loadavg(), aiclk=clock(t0, t1), kept=kept()))
        print(json.dumps(dict(ev="iter", **iters[-1])), flush=True)
        if st["n"] >= a.at + a.keep - 1:
            write()
            os._exit(0)


cls.sequence_gradients = sequence_gradients
project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={project}", f"binder_lengths=[{a.binder}]"])))
weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),) if os.environ.get(v)}
print(json.dumps(dict(ev="start", **HEAD))[:3000], flush=True)
with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
print(f"campaign ended before iteration {a.at + a.keep - 1} ({st['n']} gradient calls)", flush=True)
write()
os._exit(1)
