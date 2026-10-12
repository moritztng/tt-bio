#!/usr/bin/env python3
"""RFD3 per-region host enqueue time against device drain time: is a region dispatch-bound?

    gap_enqueue.py ARM OUT.json [STEPS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json]

ARM is gap_screen.py's '+'-joined knob set (absolute: every knob not named is off). One warm design, one plain timed
design, then a measured design. In the measured design every top-level census region (token encoder, encoder,
downcast_q, DiT, decoder, seq head) starts on a drained card: ENQ is the host time from entry to return with no sync,
DRAIN the time the card then needs to finish what was enqueued. DRAIN near 0 means the card kept up with the host,
so that region is bound by host dispatch and a trace can buy up to ENQ minus its device time; a large DRAIN means
the host ran ahead and a trace buys ~0 there (the precondition the trace plan asks for before building P1).
The syncs remove overlap, so the measured fold is slower than plain; the per-region split is the reading.
"""
import collections
import json
import os
import pathlib
import shutil
import sys
import time

ENV_KNOBS = {"gsm": "RFD3_GATHERED_SOFTMAX", "bs": "RFD3_BLOCK_SPARSE",
             "pzc": "RFD3_PROCESS_Z_COLLAPSE", "fgrid": "RFD3_FAST_GRID",
             "asdpa": "RFD3_ATOM_SDPA", "tb8": "RFD3_TRANSITION_B8",
             "dsdpa": "RFD3_DIT_SDPA", "hifi2": "RFD3_HIFI2", "sf": "RFD3_SIGMOID_FOLD",
             "rs": "RFD3_RESIDENT_STEP", "pws": "RFD3_PAIR_WS"}
_arm = set(sys.argv[1].split("+"))
assert _arm <= set(ENV_KNOBS), _arm - set(ENV_KNOBS)
for _k, _env in ENV_KNOBS.items():
    os.environ[_env] = "1" if _k in _arm else "0"

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
import ttnn.decorators as D                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 200
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")

REGIONS = [
    ("encoder", M.LocalAtomTransformer, "__call__"),
    ("token_encoder", M.DiffusionTokenEncoder, "run_device"),
    ("dit", M.LocalTokenTransformer, "__call__"),
    ("decoder", M.CompactStreamingDecoder, "run_full_device"),
    ("seq_head", M.LinearSequenceHead, "__call__"),
    ("downcast_q", M.RFD3DiffusionModule, "_downcast_q"),
]
ST = {"on": False, "depth": 0, "op_depth": 0, "region": None}
ACC = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])   # region -> [enq s, drain s, calls, ops]
STEP_WALL = []


def sync():
    ttnn.synchronize_device(T._device)


def region(label, owner, name):
    fn = getattr(owner, name)

    def w(*a, **k):
        if not ST["on"] or ST["depth"]:
            return fn(*a, **k)
        sync()
        ST["depth"] += 1
        ST["region"] = label
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            t1 = time.perf_counter()
            sync()
            acc = ACC[label]
            acc[0] += t1 - t0
            acc[1] += time.perf_counter() - t1
            acc[2] += 1
            ST["depth"] -= 1
            ST["region"] = None
    setattr(owner, name, w)


for r in REGIONS:
    region(*r)


def hook(orig):
    def call(self, *a, **k):
        if ST["on"] and not ST["op_depth"]:
            ACC[ST["region"] or "glue"][3] += 1
        ST["op_depth"] += 1
        try:
            return orig(self, *a, **k)
        finally:
            ST["op_depth"] -= 1
    return call


D.FastOperation.__call__ = hook(D.FastOperation.__call__)
D.Operation.__call__ = hook(D.Operation.__call__)

_step = M.RFD3DiffusionModule.__call__


def step(self, *a, **k):
    t0 = time.perf_counter()
    out = _step(self, *a, **k)
    if ST["on"]:
        STEP_WALL.append(time.perf_counter() - t0)
    return out


M.RFD3DiffusionModule.__call__ = step

WALL = []
_sample = RFD3Sampler.sample


def _timed(self, *a, **k):
    ST["on"] = ST.get("measure", False)
    t0 = time.perf_counter()
    try:
        out = _sample(self, *a, **k)
    finally:
        ST["on"] = False
    WALL.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(tag):
    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_enqueue_%s_%d" % (tag, os.getpid())
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, verbose=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    return WALL[0]


warm_s = fold("warm")
plain_s = fold("plain")
ST["measure"] = True
meas_s = fold("measured")
n = max(len(STEP_WALL), 1)
plain_ms = 1e3 * plain_s / STEPS
print("[enqueue] %s: warm %.2f s, plain %.2f s (%.1f ms/step), measured %.2f s (%.1f ms/step over %d steps)"
      % (ARM, warm_s, plain_s, plain_ms, meas_s, 1e3 * sum(STEP_WALL) / n, n), flush=True)
rows = {}
for reg, (enq, drain, calls, ops) in sorted(ACC.items(), key=lambda kv: -kv[1][0]):
    rows[reg] = {"enq_ms_per_step": 1e3 * enq / n, "drain_ms_per_step": 1e3 * drain / n,
                 "calls_per_step": calls / n, "ops_per_step": ops / n,
                 "enq_us_per_op": 1e6 * enq / ops if ops and reg != "glue" else None}
    print("  %-14s enq %7.2f  drain %7.2f ms/step  %6.1f ops/step" % (
        reg, rows[reg]["enq_ms_per_step"], rows[reg]["drain_ms_per_step"], rows[reg]["ops_per_step"]))
glue_ms = 1e3 * sum(STEP_WALL) / n - sum(r["enq_ms_per_step"] + r["drain_ms_per_step"] for r in rows.values())
print("  host outside the regions (index builds, uploads, reads, sampler) %.2f ms/step" % glue_ms)
rec = {"arm": ARM, "fixture": FIXTURE, "steps": STEPS, "warm_s": warm_s, "plain_s": plain_s,
       "plain_ms_per_step": plain_ms, "measured_s": meas_s, "measured_step_ms": 1e3 * sum(STEP_WALL) / n,
       "outside_regions_ms_per_step": glue_ms,
       "regions": rows, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
