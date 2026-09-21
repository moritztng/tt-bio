"""What the softmax precision lever costs, per op, at the shapes OF3 actually runs.

Three arms on the same input, interleaved A/B/C/A/B/C... because an ordered sweep charges
the first arm for compile and hands the last arm a warm cache:

    none      ttnn.softmax(x, dim=-1)                         <- what the five unconfigured
                                                                 call sites ship today
    precise   ttnn.softmax(x, dim=-1, compute_kernel_config=precise_config())
    accurate  _accurate_softmax(x, precise_config())          <- the 5-op chain

Accuracy is scored in the same harness against a float64 host softmax on the identical
values, so cost and accuracy come from one run and cannot drift apart.

The AICLK is sampled DURING the timed work (perf/clocksample.py). A timing without one is
not a measurement on this fleet.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import ttnn

from perf import clocksample
from tt_bio.autograd import host_f64_softmax_values, precise_config
from tt_bio.tenstorrent import _accurate_softmax


def f64_softmax(x: torch.Tensor) -> torch.Tensor:
    xd = x.to(torch.float64)
    m = xd.amax(dim=-1, keepdim=True)
    e = (xd - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


ARMS = {
    "none": lambda x, cfg: ttnn.softmax(x, dim=-1),
    "precise": lambda x, cfg: ttnn.softmax(x, dim=-1, compute_kernel_config=cfg),
    "accurate": lambda x, cfg: _accurate_softmax(x, cfg),
    # of3t-f64softmax's path, on the same shapes and in the same harness, so its cost and its
    # accuracy come from one run like the other three. This is the FORWARD round trip: under a
    # tape the backward pays a second one, and the both-ways figure is the scope cost in
    # perf/of3t_f64softmax/COST_ON_THE_REAL_ARM.json rather than anything here.
    "host_f64": lambda x, cfg: host_f64_softmax_values(x, -1)[1],
}


def rel_rms(got: np.ndarray, ref: np.ndarray) -> float:
    d = got.astype(np.float64) - ref
    return float(np.sqrt((d * d).mean()) / np.sqrt((ref * ref).mean()))


def run_shape(device, shape, dtype, iters, rounds, spread, seed=0):
    torch.manual_seed(seed)
    host = (torch.randn(*shape, dtype=torch.float32) * spread)
    ref = f64_softmax(host).numpy()
    tt_dtype = ttnn.float32 if dtype == "fp32" else ttnn.bfloat16
    cfg = precise_config()

    x = ttnn.from_torch(host, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    acc, per_call = {}, {a: [] for a in ARMS}

    # warm every arm once before any timing, so no arm pays for its own compile
    for name, fn in ARMS.items():
        y = fn(x, cfg)
        ttnn.synchronize_device(device)
        acc[name] = rel_rms(ttnn.to_torch(y).float().numpy(), ref)
        ttnn.deallocate(y)

    order = list(ARMS)
    for r in range(rounds):
        seq = order if r % 2 == 0 else order[::-1]   # A/B/C then C/B/A
        for name in seq:
            fn = ARMS[name]
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                y = fn(x, cfg)
                ttnn.deallocate(y)
            ttnn.synchronize_device(device)
            per_call[name].append((time.perf_counter() - t0) / iters)

    ttnn.deallocate(x)
    return {
        "shape": list(shape), "dtype": dtype, "spread": spread,
        "iters": iters, "rounds": rounds,
        "rel_rms_vs_float64": acc,
        "ms_per_call_median": {k: 1e3 * statistics.median(v) for k, v in per_call.items()},
        "ms_per_call_min": {k: 1e3 * min(v) for k, v in per_call.items()},
        "ms_per_call_rounds": {k: [1e3 * t for t in v] for k, v in per_call.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_softmax/softmax_cost_qb2c0.json")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--card", type=int, default=0, help="the PHYSICAL card, for the record")
    a = ap.parse_args()

    # OF3's own softmax shapes. The diffusion-transformer site at 384 tokens is the one
    # of3t-adaln scored; the others are the same site at the sizes users get, plus the
    # docstring's [1,16,1024,1024] so this row's numbers can be read against the 4.22x
    # that docstring quotes.
    shapes = [
        ((1, 16, 384, 384), "fp32", 8.0),
        ((1, 16, 384, 384), "bf16", 8.0),
        ((1, 16, 512, 512), "fp32", 8.0),
        ((1, 16, 768, 768), "fp32", 8.0),
        ((1, 16, 1024, 1024), "fp32", 8.0),
    ]
    if a.shapes:
        shapes = [(tuple(int(v) for v in s.split("x")), "fp32", 8.0)
                  for s in a.shapes.split(",")]

    # A lone p300 chip is a CUSTOM cluster and ttnn.open_device aborts without a mesh
    # graph descriptor. Use the engine's own resolver so this harness opens the card the
    # way a production fold does.
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    device = get_device()
    results = []
    try:
        with clocksample.during(period=2.0) as clk:
            for shape, dtype, spread in shapes:
                r = run_shape(device, shape, dtype, a.iters, a.rounds, spread)
                r["clock"] = clk.summary()
                results.append(r)
                print(json.dumps(r["ms_per_call_median"]), shape, dtype,
                      json.dumps(r["rel_rms_vs_float64"]), flush=True)
        clock = clk.summary()
    finally:
        pass

    out = {"host": "qb2", "card": a.card, "board": "p300c",
           "clock_aiclk_during": clock, "clock_line": clk.line(0),
           "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(clk.line(0))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
