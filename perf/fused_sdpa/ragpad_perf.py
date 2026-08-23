#!/usr/bin/env python3
"""Device-only price of the ragged-tail fix, per `_tri_att_sdpa` call.

The fix pads and then slices. `ttnn.pad` aliases here so it costs no DRAM, but `ttnn.slice` is
never a view, so the output copy is real and has to be priced. Arms interleave round-robin and the
OFF arm runs twice so its own A/A spread is on the page: an arm inside that spread is not a result.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, sys, time

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

assert pathlib.Path(T.__file__).is_relative_to(REPO), T.__file__


def operands(dev, S, heads, dim, seed):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: ttnn.from_torch(  # noqa: E731
        torch.randn(S, heads, S, dim, generator=g, dtype=torch.float32) * 0.5,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    q, k, v = mk(), mk(), mk()
    b = ttnn.from_torch(
        torch.randn(1, heads, S, S, generator=g, dtype=torch.float32) - 4.0,
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return q, k, v, b


def time_arm(dev, S, heads, dim, seed, ragpad, iters, warmup):
    """ms per call. Fresh operands per arm: the pad writes into the caller's physical tile tail."""
    q, k, v, b = operands(dev, S, heads, dim, seed)
    T._SDPA_RAGGED_PAD = ragpad
    scale = float(dim) ** 0.5
    for _ in range(warmup):
        ttnn.deallocate(T._tri_att_sdpa(q, k, v, b, scale))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.deallocate(T._tri_att_sdpa(q, k, v, b, scale))
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / iters
    for t in (q, k, v, b):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="298,580")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = T.get_device()
    rows = []
    for S in [int(x) for x in a.lens.split(",") if x]:
        # OFF twice (its own A/A), ON once, interleaved so drift hits every arm equally.
        acc = {"off_a": [], "off_b": [], "on": []}
        for r in range(a.reps):
            acc["off_a"].append(time_arm(dev, S, a.heads, a.dim, 7, False, a.iters, a.warmup))
            acc["on"].append(time_arm(dev, S, a.heads, a.dim, 7, True, a.iters, a.warmup))
            acc["off_b"].append(time_arm(dev, S, a.heads, a.dim, 7, False, a.iters, a.warmup))
        med = {k: statistics.median(v) for k, v in acc.items()}
        aa = abs(med["off_b"] - med["off_a"]) / med["off_a"]
        off = statistics.median(med["off_a"] for _ in (0,)) if False else (med["off_a"] + med["off_b"]) / 2
        cost = med["on"] / off
        rows.append({"S": S, "aligned": S % 32 == 0, "off_a_ms": med["off_a"],
                     "off_b_ms": med["off_b"], "on_ms": med["on"], "aa_spread": aa,
                     "on_over_off": cost, "raw": acc})
        print(f"S={S:5d}  A/A spread {aa*100:5.2f}%   off {off:8.3f} ms   on {med['on']:8.3f} ms   "
              f"{cost:6.4f}x  ({'inside A/A' if abs(cost - 1) <= aa else 'real cost'})")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"heads": a.heads, "dim": a.dim, "iters": a.iters, "reps": a.reps, "rows": rows},
            indent=2))
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
