"""How much of the Wormhole/Blackhole gap is the host rather than the chip?

RFD3's Blackhole step is 79.9 % exposed device and 21.2 % host torch plus host-side dispatch
(§1.2). A different chip does not make host work slower, so a Wormhole/Blackhole ratio cannot be
read as a device ratio until the host term is priced on both boxes. Both design models landed at
~1.9x, which is the signature of a machine-level term rather than a model-level one, and that is
what this separates.

No device is opened. These are the shapes and the op mix RFD3's host half actually runs between two
device barriers -- pair-sized gathers and index writes, small dense matmuls, and the float32
elementwise the noise schedule does -- sized from the R2 rung (3844 atoms). The absolute numbers are
not the point; the Galaxy-over-qb2 ratio is.

Threading is reported, not fixed: the Galaxy is a shared production box and its available cores are
part of the answer, so pinning threads would measure a machine we do not run on.

    PYTHONPATH=$PWD python3 perf/whdesign/wh_host_factor.py
"""
import json
import os
import pathlib
import platform
import statistics
import time

import torch

OUT = pathlib.Path(os.environ.get("HOSTF_OUT", "perf/whdesign/results/wh_host_factor.json"))
HOST = os.environ.get("HOSTF_HOST", platform.node())
REPS = int(os.environ.get("HOSTF_REPS", "9"))
N = 3844                       # R2 atoms
P = 384                        # pair channel width the trunk carries


def bench(fn, reps=REPS, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(statistics.median(ts), 4),
            "min_ms": round(min(ts), 4), "max_ms": round(max(ts), 4),
            "spread_pct": round((max(ts) - min(ts)) / statistics.median(ts) * 100, 2)}


torch.manual_seed(0)
x = torch.randn(N, P)
w = torch.randn(P, P)
idx = torch.randperm(N)
pair = torch.randn(1, 512, 512, 64)
coords = torch.randn(1, N, 3)

CASES = {
    # dense: the small per-block projections the host still does in fp32
    "matmul_3844x384x384": lambda: x @ w,
    # gather/scatter: index work is what --freeze-indices deleted 74.0 ms/step of
    "index_select_rows": lambda: x.index_select(0, idx),
    "advanced_index_write": lambda: pair.__setitem__((0, slice(0, 256), slice(0, 256)),
                                                     pair[0, 256:, 256:]),
    # elementwise fp32: the noise schedule and the coordinate updates
    "coords_axpy": lambda: coords.mul(0.997).add_(coords, alpha=0.003),
    "pair_softmax_512": lambda: torch.softmax(pair.view(1, 512, 512 * 64), dim=-1),
    # allocation + copy, which is most of what a host half actually costs
    "clone_pair_512": lambda: pair.clone(),
}

res = {"host": HOST, "torch": torch.__version__,
       "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
       "cpu_count": os.cpu_count(), "loadavg": open("/proc/loadavg").read().split()[:3],
       "reps": REPS, "cases": {k: bench(v) for k, v in CASES.items()}}
res["total_median_ms"] = round(sum(c["median_ms"] for c in res["cases"].values()), 4)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1) + "\n")
print("[hostf] " + json.dumps(res), flush=True)
