#!/usr/bin/env python3
"""Host cost of every sdpa_generic.sdpa call in an RFD3 design, split by what the call did.

    gap_sgprof.py ARM OUT.json [STEPS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json]

Tracy shows the card idle ~0.7 ms before each atom-SDPA generic_op on Blackhole (cenbh3). This prices the host side
of those calls: per call site (shape key), how often the descriptor was rebuilt, reused from the bound cache, or
already bound, and the host ms each path took, split into the Python before ttnn.generic_op and generic_op itself.
One warm design, then one measured design. No syncs: the times are what the host spends, card overlap untouched.
"""
import collections
import json
import os
import pathlib
import shutil
import sys
import time

from gap_knobs import ENV_KNOBS, apply                                   # noqa: E402
_arm = apply(sys.argv[1])
assert _arm <= set(ENV_KNOBS), _arm - set(ENV_KNOBS)

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
from tt_bio import mm_generic as MG                                       # noqa: E402
from tt_bio import sdpa_generic as SG                                     # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 200
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")

ON = [False]
CUR = {}
# site -> path -> [calls, total s, generic_op s, cores with runtime args]
ACC = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, 0.0, 0]))

_gop = ttnn.generic_op
_bd = MG.bound_descriptor
_sdpa = SG.sdpa


def gop(operands, pd):
    t0 = time.perf_counter()
    try:
        return _gop(operands, pd)
    finally:
        CUR["gop"] = CUR.get("gop", 0.0) + time.perf_counter() - t0
        try:
            CUR["cores"] = sum(len(k.runtime_args) for k in pd.kernels)
        except (AttributeError, TypeError):
            CUR["cores"] = -1


def bd(entry, addrs):
    pd = _bd(entry, addrs)
    CUR["path"] = "reuse" if pd is not None else "rebuild"
    return pd


def sdpa(device, q, k, v, mask, out, *a, **kw):
    if not ON[0]:
        return _sdpa(device, q, k, v, mask, out, *a, **kw)
    CUR.clear()
    CUR["path"] = "bound"
    t0 = time.perf_counter()
    r = _sdpa(device, q, k, v, mask, out, *a, **kw)
    dt = time.perf_counter() - t0
    site = "%s k%s css=%s" % (tuple(q.padded_shape), tuple(k.padded_shape), kw.get("css") is not None)
    acc = ACC[site][CUR["path"]]
    acc[0] += 1
    acc[1] += dt
    acc[2] += CUR.get("gop", 0.0)
    acc[3] = CUR.get("cores", 0)
    return r


ttnn.generic_op = gop
MG.bound_descriptor = bd
SG.bound_descriptor = bd
SG.sdpa = sdpa

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
    out_dir = "/tmp/gap_sgprof_%s_%d" % (tag, os.getpid())
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, verbose=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    return WALL[0]


warm_s = fold("warm")
ON[0] = True
meas_s = fold("measured")
ON[0] = False
rows = {}
tot = 0.0
for site, paths in ACC.items():
    rows[site] = {}
    for path, (n, s, g, cores) in paths.items():
        tot += s
        rows[site][path] = {"calls_per_step": n / STEPS, "us_per_call": 1e6 * s / n,
                            "generic_op_us_per_call": 1e6 * g / n, "ms_per_step": 1e3 * s / STEPS,
                            "kernel_rt_cores": cores}
        print("  %-60s %-8s %6.2f calls/step %8.1f us/call (generic_op %7.1f) %6.2f ms/step rt cores %d"
              % (site, path, n / STEPS, 1e6 * s / n, 1e6 * g / n, 1e3 * s / STEPS, cores), flush=True)
print("[sgprof] %s: warm %.2f s, measured %.2f s, sdpa_generic host %.2f ms/step" % (ARM, warm_s, meas_s,
                                                                                    1e3 * tot / STEPS))
rec = {"arm": ARM, "fixture": FIXTURE, "steps": STEPS, "warm_s": warm_s, "measured_s": meas_s,
       "host_ms_per_step": 1e3 * tot / STEPS, "sites": rows, "bound_stats": dict(MG.BOUND_STATS),
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
