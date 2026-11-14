#!/usr/bin/env python3
"""RFD3 device census on the page workload: device kernel time per step, per region and per op, against the wall.

    gap_tracy.py OUT.json [STEPS] [PROFILED_STEPS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json] [--fast]

Needs the Tracy-enabled tt-metal (the pip wheel compiles the device profiler out); run it through
perf/spd_census/run.sh's environment (TT_METAL_DEVICE_PROFILER=1, MID_RUN_DUMP, no push to Tracy).

One warm design at STEPS steps and a plain one timed at the sampler boundary (the page number), then a design whose
first PROFILED_STEPS steps record every ttnn op: its name, the region it ran in (the innermost of the census regions)
and its device-operation id range. The profiler buffer is read every ~1000 programs. Each program's DEVICE KERNEL
DURATION is joined back to its op by runtime id. Device kernel ms/step against the plain fold's wall ms/step is the
share of the step the device is busy; the rest is the host (torch, dispatch, transfers) with the device idle.

Each op is also keyed by its tensor signature (shape and dtype of every tensor in and out), with the bytes those
tensors occupy and, for matmul-like ops, 2*M*K*N. That puts every op on a roofline (perf/spd/rfd3_roof.py) without
re-deriving shapes from the model code.
"""
import collections
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
import ttnn.decorators as D                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FAST = "--fast" in sys.argv
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT = pathlib.Path(args[0])
STEPS = int(args[1]) if len(args) > 1 else 200
PSTEPS = int(args[2]) if len(args) > 2 else 12
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")

REGIONS = [
    ("encoder", M.LocalAtomTransformer, "__call__"),
    ("token_encoder", M.DiffusionTokenEncoder, "run_device"),
    ("dit", M.LocalTokenTransformer, "__call__"),
    ("decoder", M.CompactStreamingDecoder, "run_full_device"),
    ("seq_head", M.LinearSequenceHead, "__call__"),
    ("downcast_q", M.RFD3DiffusionModule, "_downcast_q"),
]
REG = []
ST = {"arm": False, "on": False, "step": 0, "flushed": 0, "depth": 0}
OPID = ttnn._ttnn.get_device_operation_id
OPS = []          # (id0, id1, name, region, step, sig)
SIGS = {}         # sig -> (bytes, flops)
ESZ = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1, "BFLOAT8_B": 1088 / 1024,
       "BFLOAT4_B": 576 / 1024}
KERN = {}         # runtime id >> 10 -> summed kernel ns
SPAN = {}         # runtime id >> 10 -> [first kernel start, last kernel end, summed ns] in device timestamp units
STEP_WALL = []


def region(label, owner, name):
    fn = getattr(owner, name)

    def w(*a, **k):
        REG.append(label)
        try:
            return fn(*a, **k)
        finally:
            REG.pop()
    setattr(owner, name, w)


for r in REGIONS:
    region(*r)


def drain():
    ttnn.synchronize_device(T._device)
    for attempt in range(4):
        try:
            ttnn.ReadDeviceProfiler(T._device)
            break
        except RuntimeError as e:
            if attempt == 3:
                raise
            time.sleep(2)
    for _chip, progs in ttnn.get_latest_programs_perf_data().items():
        for p in progs:
            r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if r is not None:
                rid = p.program_execution_uid.runtime_id >> 10
                KERN[rid] = KERN.get(rid, 0) + r.duration
                sp = SPAN.setdefault(rid, [r.start_timestamp, r.end_timestamp, 0])
                sp[0], sp[1], sp[2] = min(sp[0], r.start_timestamp), max(sp[1], r.end_timestamp), sp[2] + r.duration
    ST["flushed"] = OPID()
    ST["chunk"] = ST.get("chunk", 0) + 1


def tensors(x):
    if isinstance(x, ttnn.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from tensors(y)


def signature(name, a, k, out):
    ins = [t for v in list(a) + list(k.values()) for t in tensors(v)]
    outs = list(tensors(out))
    desc, nbytes = [], 0.0
    for t in ins + outs:
        try:
            shp, dt = tuple(t.padded_shape), str(t.dtype).split(".")[-1]
        except Exception:
            continue
        n = 1
        for d in shp:
            n *= d
        nbytes += n * ESZ.get(dt, 2)
        desc.append("x".join(map(str, shp)) + ":" + dt.lower().replace("bfloat", "bf"))
    flops = 0.0
    short = name.rsplit(".", 1)[-1]
    if short in ("matmul", "linear") and len(ins) >= 2:
        sa, sb = tuple(ins[0].padded_shape), tuple(ins[1].padded_shape)
        m = 1
        for d in sa[:-1]:
            m *= d
        flops = 2.0 * m * sa[-1] * sb[-1]
    sig = "%s | %s -> %s" % (short, ",".join(desc[:len(ins)]), ",".join(desc[len(ins):]))
    SIGS[sig] = (nbytes, flops)
    return sig


def hook(orig):
    def call(self, *a, **k):
        if not ST["on"] or ST["depth"]:
            return orig(self, *a, **k)
        ST["depth"] += 1
        try:
            id0 = OPID()
            out = orig(self, *a, **k)
            id1 = OPID()
            if id1 > id0:
                name = self.python_fully_qualified_name
                OPS.append((id0, id1, name, REG[-1] if REG else "glue", ST["step"], signature(name, a, k, out),
                            ST.get("chunk", 0)))
            if id1 - ST["flushed"] >= 1000:
                drain()
            return out
        finally:
            ST["depth"] -= 1
    return call


D.FastOperation.__call__ = hook(D.FastOperation.__call__)
D.Operation.__call__ = hook(D.Operation.__call__)

_step = M.RFD3DiffusionModule.__call__


def step(self, *a, **k):
    if ST["arm"]:
        # the design's one-time initializer has run; profile from the first step on
        drain()
        KERN.clear()
        ST.update(arm=False, on=True, step=0)
    if ST["on"] and ST["step"] >= PSTEPS:
        drain()
        ST["on"] = False
    t0 = time.perf_counter()
    out = _step(self, *a, **k)
    if ST["on"]:
        STEP_WALL.append(time.perf_counter() - t0)
        ST["step"] += 1
    return out


M.RFD3DiffusionModule.__call__ = step

WALL = []
_sample = RFD3Sampler.sample


def _timed(self, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, *a, **k)
    WALL.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(tag):
    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_tracy_%s" % tag
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, fast=FAST, verbose=False)
    return WALL[0]


warm_s = fold("warm")
plain_s = fold("plain")
print("[tracy] warm %.2f s, plain %.2f s (%.1f ms/step)" % (warm_s, plain_s, 1e3 * plain_s / STEPS), flush=True)
ST["arm"] = True
prof_s = fold("profiled")
if ST["on"]:
    drain()
    ST["on"] = False

n = max(ST["step"], 1)
by_op = collections.defaultdict(lambda: [0.0, 0])
by_sig = collections.defaultdict(lambda: [0.0, 0])
by_reg = collections.defaultdict(float)
missing = 0
for id0, id1, name, reg, _s, sig, _c in OPS:
    ns = sum(KERN.get(i, 0) for i in range(id0, id1))
    missing += all(i not in KERN for i in range(id0, id1))
    by_op[(reg, name)][0] += ns / 1e6 / n
    by_op[(reg, name)][1] += 1
    by_sig[(reg, sig)][0] += ns / 1e6 / n
    by_sig[(reg, sig)][1] += 1
    by_reg[reg] += ns / 1e6 / n
dev_ms = sum(by_reg.values())
wall_ms = 1e3 * plain_s / STEPS
print("[tracy] %d profiled steps, %d ops (%d with no program read back); device kernel %.1f ms/step against "
      "the plain wall %.1f ms/step: device busy %.0f %%" % (n, len(OPS), missing, dev_ms, wall_ms, 100 * dev_ms / wall_ms))
for reg, ms in sorted(by_reg.items(), key=lambda kv: -kv[1]):
    print("  %-14s %7.2f ms/step" % (reg, ms))
top = sorted(by_op.items(), key=lambda kv: -kv[1][0])[:40]
for (reg, name), (ms, c) in top:
    print("  %-14s %-48s %7.2f ms/step  %6.1f calls/step" % (reg, name[-48:], ms, c / n))

# Device idle between consecutive kernels inside one profiler chunk (a drain ends a chunk, so its stall is not counted),
# charged to the op that waited to start: launch cost shows as many small gaps, a host stall or upload as a few large.
spans = []
for id0, id1, name, reg, _s, sig, c in OPS:
    sp = [SPAN[i] for i in range(id0, id1) if i in SPAN]
    if sp:
        spans.append((c, min(x[0] for x in sp), max(x[1] for x in sp), reg, name))
ratio = sorted(SPAN[i][2] / max(SPAN[i][1] - SPAN[i][0], 1) for i in SPAN)
ns_per_unit = ratio[len(ratio) // 2] if ratio else 1.0
spans.sort(key=lambda x: (x[0], x[1]))
gap_by = collections.defaultdict(lambda: [0.0, 0])
gap_hist = collections.defaultdict(lambda: [0.0, 0])
for prev, cur in zip(spans, spans[1:]):
    if prev[0] != cur[0]:
        continue
    g = max(cur[1] - prev[2], 0) * ns_per_unit / 1e3
    b = next(lim for lim in (2, 10, 50, 200, 1000, 10000, float("inf")) if g < lim)
    gap_hist[b][0] += g / 1e3 / n
    gap_hist[b][1] += 1
    gap_by[(cur[3], cur[4].rsplit(".", 1)[-1], prev[4].rsplit(".", 1)[-1])][0] += g / 1e3 / n
    gap_by[(cur[3], cur[4].rsplit(".", 1)[-1], prev[4].rsplit(".", 1)[-1])][1] += 1
gap_ms = sum(v[0] for v in gap_hist.values())
print("[tracy] device idle between kernels %.1f ms/step (in-chunk); by gap size (us):" % gap_ms)
for b, (ms, c) in sorted(gap_hist.items()):
    print("  < %-6s %7.2f ms/step %8.1f gaps/step" % (b, ms, c / n))
gap_top = sorted(gap_by.items(), key=lambda kv: -kv[1][0])[:40]
for (reg, op, after), (ms, c) in gap_top[:20]:
    print("  idle before %-14s %-28s after %-28s %6.2f ms/step %6.1f/step" % (reg, op[-28:], after[-28:], ms, c / n))
sigs = [{"region": r, "sig": g, "ms_per_step": ms, "calls_per_step": c / n, "bytes_per_call": SIGS[g][0],
         "flops_per_call": SIGS[g][1]} for (r, g), (ms, c) in sorted(by_sig.items(), key=lambda kv: -kv[1][0])]
rec = {"fixture": FIXTURE, "fast": FAST, "steps": STEPS, "profiled_steps": n, "warm_s": warm_s, "plain_s": plain_s,
       "profiled_fold_s": prof_s, "step_wall_ms_profiled": [1e3 * x for x in STEP_WALL],
       "device_ms_per_step": dev_ms, "plain_wall_ms_per_step": wall_ms, "ops": len(OPS), "ops_missing": missing,
       "regions_ms_per_step": dict(by_reg),
       "top_ops": [{"region": r, "op": o, "ms_per_step": ms, "calls_per_step": c / n} for (r, o), (ms, c) in top],
       "sigs": sigs, "idle_ms_per_step": gap_ms, "ns_per_timestamp": ns_per_unit,
       "idle_hist_us": {str(b): {"ms_per_step": ms, "gaps_per_step": c / n} for b, (ms, c) in sorted(gap_hist.items())},
       "idle_top": [{"region": r, "op": o, "after": a, "ms_per_step": ms, "gaps_per_step": c / n}
                    for (r, o, a), (ms, c) in gap_top],
       "arch": T.arch_name(),
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
