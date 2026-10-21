"""ANOMALIES A1: is `tri_mult_out` at n=256 really 8 s, or was the chip contended?

`scripts/rfd4_ceiling_bench.py` measured 8.136 s for one `tri_mult_out` call at n=256 against 4.0 ms
for `tri_mult_in` at the same size and 15.3 ms for outgoing at n=512. Its `timed()` already threw a
warmup away and averaged five calls, so a first-call compile does not explain it; what is left is
contention on the card or a genuinely different path at that one shape.

This reports every repeat separately instead of a mean, so one slow call cannot hide inside four fast
ones and four slow ones cannot be read as one. Both directions run at every size, each on its own
freshly built module, with the device synchronised around each call.

    TT_VISIBLE_DEVICES=<idle card> TT_BIO_LEASE_CARDS=<card> python perf/rfd4/e2e/a1_trimul.py \
        --out state/rfd4/census/a1-trimul.json

The chip must be idle: contention is the hypothesis under test, so a shared card cannot refute it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()  # p300 is a CUSTOM cluster; ttnn refuses to open without the descriptor

import ttnn  # noqa: E402

from tt_bio import tenstorrent as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    sys.path.insert(0, "tests")
    import rfd4_reference as REF

    from tt_bio.rfd4 import trunk as TR

    sys.path.insert(0, "perf")
    import clocksample

    # `get_device` is the only correct opener on a p300: it takes the host-local card lease and
    # knows the custom mesh descriptor a bare `open_device` refuses without.
    dev = T.get_device()
    weights = REF.pair_update_weights(1024, 384, 256, seed=0, prefix="p")
    scope = T.WeightScope({k: v.float() for k, v in weights.items()}).child("p")
    rows = []
    with clocksample.during(period=0.5) as clk:
        for n in a.sizes:
            for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
                mod = T.TriangleMultiplication(ending, TR._trimul_weights(scope.child(name)),
                                               TR.compute_kernel_config())
                z = ttnn.from_torch(torch.randn(1, n, n, 384), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev)
                pm = ttnn.from_torch(torch.ones(1, n, n), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=dev)
                ttnn.deallocate(mod(z, pm))          # warmup, discarded
                ttnn.synchronize_device(dev)
                per_call = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    out = mod(z, pm)
                    ttnn.synchronize_device(dev)
                    per_call.append(time.perf_counter() - t0)
                    ttnn.deallocate(out)
                fl = 2 * n * n * (4 * 384 * 256 + 256 * 384 + 384 * 384) + 2 * 256 * n ** 3
                row = {"op": name, "n": n, "c_z": 384, "c_hidden": 256,
                       "s_per_call": [round(s, 6) for s in per_call],
                       "median_s": round(sorted(per_call)[len(per_call) // 2], 6),
                       "max_s": round(max(per_call), 6),
                       "tflops_median": round(fl / sorted(per_call)[len(per_call) // 2] / 1e12, 2)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                ttnn.deallocate(z)
                ttnn.deallocate(pm)
    aiclk = clk.summary()
    by = {(r["op"], r["n"]): r for r in rows}
    # Two different claims live in these numbers and must not be collapsed. The rate claim is the
    # median against the incoming control at the same size; the tail claim is the worst call.
    rate_slow = [r for r in rows if r["op"] == "tri_mult_out"
                 and r["median_s"] > 3 * by[("tri_mult_in", r["n"])]["median_s"]]
    stalls = [r for r in rows if r["max_s"] > 5 * r["median_s"]]
    if rate_slow:
        verdict, reading = "RATE", (
            "outgoing really is slower per call than incoming at the same shape: a path difference "
            "in a module Boltz-2 and OpenFold3 also run, and a throughput bug in all three")
    elif stalls:
        verdict, reading = "TAIL", (
            "the steady-state rate is normal and matches the incoming control; what is real is a "
            "sporadic stall of hundreds of milliseconds on a call that is not the first, so the "
            "original 8.136 s was one stall averaged over five calls, not a 2000x path. It is a "
            "tail-latency defect, not a throughput one -- but TriangleMultiplication is tt-bio's "
            "shared module, so a one-second stall lands in Boltz-2 and OpenFold3 latency too")
    else:
        verdict, reading = "NOT REPRODUCED", (
            "every repeat is fast and no call stalls: the 8.136 s was contention on a shared chip, "
            "the ceiling's triangle number stands and nothing is owed")
    out = {"anomaly": "A1", "claim": "tri_mult_out n=256 = 8.136 s vs 4.0 ms for tri_mult_in",
           "method": f"{a.repeats} repeats after a discarded warmup, synchronised per call, idle chip",
           "aiclk": aiclk, "rows": rows,
           "stalled": [{"op": r["op"], "n": r["n"], "max_s": r["max_s"], "median_s": r["median_s"]}
                       for r in stalls],
           "verdict": verdict, "reading": reading}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "aiclk", "reading")}, indent=2))



if __name__ == "__main__":
    main()
