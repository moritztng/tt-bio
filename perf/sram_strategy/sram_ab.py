#!/usr/bin/env python3
"""W7 placement A/B on one Protenix-v2 Pairformer block at the 298-aa shape.

Each variant changes only WHERE tensors live, never the arithmetic or its order, so every
one must come out bit-exact against the default. That is checked here with torch.equal on
both outputs, not assumed.

Variants:
  base       production placement on this host
  chunk64    trimul hidden-channel chunk 64 (4 chunks instead of 8 at N=320)
  chunk128   trimul hidden-channel chunk 128 (2 chunks)
  l1norm     the trimul's normalised pair tensor stays in L1 across the channel loop
             (the loop re-reads it once per chunk: 8 x 52.4 MB per trimul from DRAM today)
  oneconcat  one concat of all channel chunks instead of the running O(n^2) one
  best       every winning gate together

Legs are interleaved round-robin (PLAYBOOKS measurement rule 3), each timed region has a
synchronize_device on both sides, and the first --warm calls per leg are discarded.

    TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=... \\
      python3 perf/sram_strategy/sram_ab.py --n 320 --out perf/sram_strategy/ab_n320.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

VARIANTS = {
    "base":      {},
    "chunk64":   {"_TRIMUL_CHUNK_OVERRIDE": 64},
    "chunk128":  {"_TRIMUL_CHUNK_OVERRIDE": 128},
    "l1norm":    {"_TRIMUL_L1_NORM": True},
    "oneconcat": {"_TRIMUL_ONE_CONCAT": True},
}
DEFAULTS = {"_TRIMUL_CHUNK_OVERRIDE": 0, "_TRIMUL_L1_NORM": False, "_TRIMUL_ONE_CONCAT": False}


def set_gates(gates):
    for k, v in DEFAULTS.items():
        setattr(T, k, gates.get(k, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--best", default=None,
                    help="comma-separated gate names to combine into one extra leg, e.g. l1norm,oneconcat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.variant or list(VARIANTS)
    legs = {n: dict(VARIANTS[n]) for n in names}
    if args.best:
        combo = {}
        for part in args.best.split(","):
            combo.update(VARIANTS[part.strip()])
        legs["best"] = combo

    dev = get_device()
    print(f"grid={T.COMPUTE_GRID_MAIN} cores={T.COMPUTE_GRID_MAIN[0]*T.COMPUTE_GRID_MAIN[1]}", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = args.n

    def one_block(gates):
        """One block on a fresh copy of the inputs; returns (s, z) as torch."""
        set_gates(gates)
        torch.manual_seed(0)
        s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        out = (ttnn.to_torch(s), ttnn.to_torch(z))
        ttnn.deallocate(s)
        ttnn.deallocate(z)
        return out

    # --- bit-exactness, one block from identical inputs, before any timing ---
    exact = {}
    ref = None
    for name, gates in legs.items():
        try:
            got = one_block(gates)
        except Exception as e:  # a variant that will not fit says so instead of dying
            exact[name] = f"FAIL {type(e).__name__}: {str(e)[:120]}"
            print(f"{name}: {exact[name]}", flush=True)
            continue
        if ref is None and name == "base":
            ref = got
            exact[name] = "ref"
        elif ref is not None:
            exact[name] = bool(torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]))
        print(f"{name}: bit-exact vs base = {exact[name]}", flush=True)

    # --- timing, interleaved ---
    state, times = {}, {n: [] for n in legs}
    ok = [n for n in legs if not isinstance(exact.get(n), str) or exact[n] == "ref"]
    for name in ok:
        set_gates(legs[name])
        torch.manual_seed(0)
        s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for _ in range(args.warm):
            s, z = layer(s, z)
        state[name] = (s, z)
    ttnn.synchronize_device(dev)

    for _ in range(args.iters):
        for name in ok:
            set_gates(legs[name])
            s, z = state[name]
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            s, z = layer(s, z)
            ttnn.synchronize_device(dev)
            times[name].append((time.perf_counter() - t0) * 1e3)
            state[name] = (s, z)
    set_gates({})

    base_med = None
    res = {}
    print(f"\n{'variant':<12} {'median_ms':>10} {'min_ms':>9} {'vs base':>9}  bit-exact")
    for name in ok:
        med = sorted(times[name])[len(times[name]) // 2]
        if name == "base":
            base_med = med
        res[name] = {"median_ms": med, "min_ms": min(times[name]), "ms": times[name],
                     "bit_exact": exact.get(name)}
    for name in ok:
        med = res[name]["median_ms"]
        res[name]["speedup_vs_base"] = base_med / med if base_med else None
        print(f"{name:<12} {med:>10.2f} {res[name]['min_ms']:>9.2f} "
              f"{(base_med / med if base_med else 0):>9.4f}  {exact.get(name)}")
    for name, v in exact.items():
        if isinstance(v, str) and v != "ref":
            print(f"{name:<12} {v}")
            res[name] = {"error": v}
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": N, "c_z": c_z, "grid": list(T.COMPUTE_GRID_MAIN), "variants": res}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
