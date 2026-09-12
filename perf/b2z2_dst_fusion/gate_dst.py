#!/usr/bin/env python3
"""Is a deleted L1 tile pass worth anything in an op whose reader is already the slow end?

`reblock_permute_gated` (E6) is the trimul's fused channel move. Per output tile its compute kernel
runs three separate DST acquires, each packing to an L1 CB and unpacking again:

    g   -> DST -> sigmoid -> pack sig_cb
    p, sig -> DST -> mul  -> pack mul_cb
    mul -> DST -> transpose_wh -> pack out_cb          3 packs, 4 unpacks, 3 acquires

transpose_wh is a pure index permutation, so it commutes elementwise with both the sigmoid and the
multiply. Move it to the two unpacks at the front and the middle CB disappears:

    g   -> DST(transposed) -> sigmoid -> pack sig_cb
    p, sig -> DST(p transposed) -> mul -> pack out_cb   2 packs, 3 unpacks, 2 acquires

Bit-exact: the same values in the same order with the same two roundings (the sigmoid still lands in
bf16 in sig_cb before the multiply reads it, which is the rounding ttnn has and the whole reason
this kernel exists).

This measures both halves: `torch.equal` between the two arms, and an interleaved paired A/B with
its own A/A floor. The op reads 4 Z and writes 2 Z of DRAM per trimul, so the honest prediction is
that the deleted passes hide behind the reader and the ratio is at or near 1.000x.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device


def run_once(xw, C, mc):
    out = RB.reblock_permute_gated(xw, 2 * C, 0, C, mc)
    return out


def timed(dev, xw, C, mc, reps):
    """Wall of `reps` back-to-back calls, device-synced at both ends."""
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = run_once(xw, C, mc)
        ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) * 1e3 / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--reps", type=int, default=4, help="calls per timed arm")
    ap.add_argument("--pairs", type=int, default=7, help="interleaved A/B pairs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    N, C = a.n, a.c
    mc = ttnn.DRAM_MEMORY_CONFIG
    res = {"N": N, "C": C, "reps": a.reps, "pairs": a.pairs,
           "grid": str(dev.compute_with_storage_grid_size()),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn")}

    torch.manual_seed(0)
    h = torch.randn(1, N, N, 4 * C, dtype=torch.bfloat16)
    xw = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=mc)

    # --- parity: both arms in one process, torch.equal ---
    prev = RB.set_gate_dst_resident(False)
    base_t = run_once(xw, C, mc); base = ttnn.to_torch(base_t); ttnn.deallocate(base_t)
    RB.set_gate_dst_resident(True)
    dst_t = run_once(xw, C, mc); dst = ttnn.to_torch(dst_t); ttnn.deallocate(dst_t)
    res["shape_ok"] = list(base.shape) == list(dst.shape)
    res["equal"] = bool(res["shape_ok"] and torch.equal(base, dst))
    if not res["equal"] and res["shape_ok"]:
        d = (dst.float() - base.float()).abs()
        res["max_abs"] = float(d.max())
        res["frac_diff"] = round(float((d > 0).float().mean()), 6)
    # A live check that the variant actually reached the kernel: the two programs must differ.
    res["programs"] = len(RB._CACHE_GATED)
    print(f"parity: equal={res['equal']} programs_cached={res['programs']} "
          f"(must be 2, or the A/B is reading one compiled program twice)", flush=True)

    # --- interleaved paired A/B, plus an A/A floor from the same interleave ---
    base_ms, dst_ms, aa_ms = [], [], []
    for i in range(a.pairs + 1):
        RB.set_gate_dst_resident(False)
        b1 = timed(dev, xw, C, mc, a.reps)
        RB.set_gate_dst_resident(True)
        d1 = timed(dev, xw, C, mc, a.reps)
        RB.set_gate_dst_resident(False)
        b2 = timed(dev, xw, C, mc, a.reps)
        if i == 0:
            continue                      # first pair is the warm-up, discarded
        base_ms += [b1, b2]
        dst_ms.append(d1)
        aa_ms.append(b2 / b1)
        print(f"  pair {i}: base {b1:.4f} / {b2:.4f} ms   dst {d1:.4f} ms   "
              f"ratio {(b1 + b2) / 2 / d1:.4f}x   A/A {b2 / b1:.4f}", flush=True)
    RB.set_gate_dst_resident(prev)

    res["base_ms_median"] = statistics.median(base_ms)
    res["dst_ms_median"] = statistics.median(dst_ms)
    res["ratio"] = res["base_ms_median"] / res["dst_ms_median"]
    res["aa_floor"] = statistics.median(aa_ms)
    res["aa_spread"] = max(aa_ms) - min(aa_ms)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nbase {res['base_ms_median']:.4f} ms  dst {res['dst_ms_median']:.4f} ms  "
          f"RATIO {res['ratio']:.4f}x  A/A floor {res['aa_floor']:.4f} "
          f"(spread {res['aa_spread']:.4f})  bit-exact {res['equal']}")


if __name__ == "__main__":
    main()
