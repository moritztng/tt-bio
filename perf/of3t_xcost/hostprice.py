"""What the exact host float64 softmax costs as a function of its WORKING SET, no card.

`of3t-bwattrib`'s per-shape-class table prices one tensor class -- the triangle attention
score block `[384,4,384,384]` FLOAT32 -- at 437.99 s of a 705.78 s backward:

    softmax_in_place  108 calls  186.918 s self   the HOST float64 arithmetic
    to_torch          216 calls  153.854 s        1.91 GB/s
    from_torch        216 calls   97.214 s        3.02 GB/s

`perf/of3t_tapedfwd/exactprice.py` put the CPU arithmetic floor at 1.9813 ns/element, which
over 226,492,416 elements is 0.4488 s. Production reads 1.7307 s per call. The gap is 3.86x and
it is not the arithmetic: exactprice CHUNKED the tensor on its leading axis (its own docstring
says why -- "pc has 30 GB and one such tensor is 1.81 GB in float64") and
`autograd._exact_softmax_raw` does not. It hands the whole [384,4,384,384] to `.double()` and
then to `torch.softmax`, which allocates two 1.81 GB float64 temporaries on a box with 6 GB
free and a full 976 MB swap.

So this sweeps the chunk size over the SAME total work and reads the per-element rate off it.
One knob, one tensor, one process. Chunking a softmax on its leading axis is arithmetically a
no-op -- the reduction is over the last axis and every row is independent -- so the arms are
checked BIT-IDENTICAL against the monolithic one rather than assumed to be.

Board-insensitive by construction: no device is opened, `import ttnn` never happens. The
numbers are a property of this host's memory system (AMD Ryzen 5 8600G, 6 cores / 12 threads,
16 MiB L3), which is the host every figure above was measured on.

    hostprice.py --out perf/of3t_xcost/out/HOSTPRICE.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time

import torch


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def loadavg():
    return [round(x, 2) for x in os.getloadavg()]


def fwd_mono(x, out):
    """`_exact_softmax_raw`'s arithmetic verbatim: whole-tensor cast, softmax, round back."""
    out.copy_(torch.softmax(x.double(), dim=-1).float())


def fwd_chunk(x, out, rows):
    for i in range(0, x.shape[0], rows):
        out[i:i + rows] = torch.softmax(x[i:i + rows].double(), dim=-1).float()


def bw_mono(y64, g, out):
    """`host_f64_softmax`'s bw closure: dx = y * (g - sum(g*y))."""
    g64 = g.double()
    inner = (g64 * y64).sum(dim=-1, keepdim=True)
    out.copy_((y64 * (g64 - inner)).float())


def bw_chunk(y64, g, out, rows):
    for i in range(0, g.shape[0], rows):
        g64 = g[i:i + rows].double()
        y = y64[i:i + rows]
        inner = (g64 * y).sum(dim=-1, keepdim=True)
        out[i:i + rows] = (y * (g64 - inner)).float()


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunks", default="8,16,32,48,96,192,384")
    ap.add_argument("--headroom", type=float, default=0.70,
                    help="fraction of MemAvailable an arm's predicted peak may use")
    ap.add_argument("--out", default="perf/of3t_xcost/out/HOSTPRICE.json")
    a = ap.parse_args()

    n, h = a.tokens, a.heads
    shape = (n, h, n, n)
    el = n * h * n * n
    rep = {
        "doc": __doc__.strip(),
        "argv": vars(a),
        "env": {"host": platform.node(), "torch": torch.__version__,
                "threads": torch.get_num_threads(), "cpus": os.cpu_count(),
                "mem_available_gb_start": round(mem_available_gb(), 2),
                "loadavg_start": loadavg(),
                "board_insensitive": "yes -- CPU arithmetic only, no device is opened",
                "axis": "self wall-clock seconds of the host float64 arithmetic ONLY; no "
                        "to_torch/from_torch is in any number here"},
        "shape": list(shape), "elements": el,
        "banked_for_comparison": {
            "production_softmax_in_place_self_s_per_call": 1.7307192,
            "production_source": "perf/of3t_bwattrib/out/hist_384_base.json by_shape_class, "
                                 "108 calls / 186.9177 s, arm base, pc card 0, AICLK 1350 "
                                 "median sampled DURING (env.aiclk_during)",
            "exactprice_floor_ns_per_element": 1.9813,
            "exactprice_floor_s_at_this_shape": el * 1.9813 / 1e9,
        },
        "arms": {},
    }

    x = torch.randn(*shape, dtype=torch.float32)
    out = torch.empty(*shape, dtype=torch.float32)
    base_gb = (x.numel() * 4 * 2) / 1e9

    chunks = [int(c) for c in a.chunks.split(",")]
    ref = None
    for c in sorted(chunks):
        peak_gb = c * h * n * n * 8 * 2 / 1e9
        avail = mem_available_gb()
        arm = {"rows": c, "predicted_peak_temporaries_gb": round(peak_gb, 3),
               "mem_available_gb": round(avail, 2)}
        if peak_gb > a.headroom * avail:
            arm["skipped"] = (f"predicted peak {peak_gb:.2f} GB exceeds {a.headroom:.0%} of "
                              f"{avail:.2f} GB available; not run rather than swapping this "
                              f"host, which other rows share")
            rep["arms"][f"fwd_rows{c}"] = arm
            continue
        fn = (lambda: fwd_mono(x, out)) if c >= n else (lambda c=c: fwd_chunk(x, out, c))
        ts = timed(fn, a.reps)
        if ref is None:
            ref = out.clone()
            arm["bit_identical_to"] = "reference (smallest chunk run)"
        else:
            arm["bit_identical"] = bool(torch.equal(out, ref))
            arm["max_abs_diff"] = float((out - ref).abs().max())
        arm["s"] = ts
        arm["median_s"] = statistics.median(ts)
        arm["ns_per_element"] = arm["median_s"] / el * 1e9
        rep["arms"][f"fwd_rows{c}"] = arm
        print(f"fwd rows={c:>4}  {arm['median_s']:7.3f} s  "
              f"{arm['ns_per_element']:6.3f} ns/el  peak+{peak_gb:.2f} GB", flush=True)

    del out, ref
    gc.collect()

    # the backward closure, same sweep. y64 is RETAINED by the tape (1.81 GB of float64 per
    # live softmax node) whatever the chunk size, so only the g64 and the two intermediates
    # move with it.
    y64 = torch.softmax(x[:32].double(), dim=-1).repeat(n // 32, 1, 1, 1)
    del x
    gc.collect()
    g = torch.randn(*shape, dtype=torch.float32)
    dout = torch.empty(*shape, dtype=torch.float32)
    bref = None
    for c in sorted(chunks):
        peak_gb = c * h * n * n * 8 * 3 / 1e9
        avail = mem_available_gb()
        arm = {"rows": c, "predicted_peak_temporaries_gb": round(peak_gb, 3),
               "mem_available_gb": round(avail, 2),
               "retained_y64_gb": round(y64.numel() * 8 / 1e9, 3)}
        if peak_gb > a.headroom * avail:
            arm["skipped"] = (f"predicted peak {peak_gb:.2f} GB exceeds {a.headroom:.0%} of "
                              f"{avail:.2f} GB available")
            rep["arms"][f"bw_rows{c}"] = arm
            continue
        fn = (lambda: bw_mono(y64, g, dout)) if c >= n else (lambda c=c: bw_chunk(y64, g, dout, c))
        ts = timed(fn, a.reps)
        if bref is None:
            bref = dout.clone()
            arm["bit_identical_to"] = "reference (smallest chunk run)"
        else:
            arm["bit_identical"] = bool(torch.equal(dout, bref))
            arm["max_abs_diff"] = float((dout - bref).abs().max())
        arm["s"] = ts
        arm["median_s"] = statistics.median(ts)
        arm["ns_per_element"] = arm["median_s"] / el * 1e9
        rep["arms"][f"bw_rows{c}"] = arm
        print(f"bw  rows={c:>4}  {arm['median_s']:7.3f} s  "
              f"{arm['ns_per_element']:6.3f} ns/el  peak+{peak_gb:.2f} GB", flush=True)

    rep["env"]["loadavg_end"] = loadavg()
    rep["env"]["mem_available_gb_end"] = round(mem_available_gb(), 2)
    rep["env"]["resident_base_gb"] = round(base_gb, 2)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({"env": rep["env"]}, indent=1))


if __name__ == "__main__":
    main()
