#!/usr/bin/env python3
"""How much aggregate L1 a Pairformer block actually leaves free at the 298-aa shape.

Every W7 placement variant that added ~52 MB of L1 residency (trimul chunk 64/128, an
L1-resident normalised pair tensor, a single concat of all channel chunks) died the same
way:

    Statically allocated circular buffers in program N clash with L1 buffers on core range
    [(0,0)-(10,9)]. L1 buffer allocated at 1148928 and static circular buffer region ends
    at 1159680

so the binding constraint is not the 1.53 MB/core of unreserved L1, it is what the block's
own kernels leave under their circular buffers. This measures that directly: pin a dummy
L1-interleaved tensor of X MB, run the workload, and bisect X for the largest one that
still runs. The answer is the L1 budget any residency strategy at this shape has to fit in.

Run it per phase as well as for the whole block, because the tightest kernel sets the
budget for everything downstream of it.

    TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=... \\
      python3 perf/sram_strategy/l1_headroom.py --n 320 --out perf/sram_strategy/headroom_n320.json
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


def pad_tensor(dev, mb):
    """An L1-interleaved bf16 tensor of about `mb` MB, tile-aligned."""
    tiles = max(1, int(mb * 1e6 / 2048))          # 32x32 bf16 tile = 2048 B
    rows = 32
    cols = tiles * 32
    return ttnn.zeros((1, 1, rows, cols), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                      device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=140.0)
    ap.add_argument("--tol", type=float, default=2.0, help="MB")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    gx, gy = T.COMPUTE_GRID_MAIN
    cores = gx * gy
    per_core = ttnn.get_max_worker_l1_unreserved_size()
    print(f"grid={gx}x{gy} cores={cores} l1_per_core={per_core} "
          f"aggregate_L1={cores*per_core/1e6:.1f} MB", flush=True)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = args.n
    torch.manual_seed(0)

    def fresh():
        s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        return s, z

    workloads = {
        "block": lambda s, z: layer(s, z),
        "trimul_start": lambda s, z: (s, layer.triangle_multiplication_start(z)),
        "tri_att_start": lambda s, z: (s, layer.triangle_attention_start(z)),
        "transition_z": lambda s, z: (s, layer.transition_z(z)),
        "attention_pair_bias": lambda s, z: (s, layer.attention_pair_bias(s, z)),
    }

    # warm every kernel once so the bisect never pays a compile
    s, z = fresh()
    for _ in range(args.warm):
        for fn in workloads.values():
            fn(s, z)
    ttnn.synchronize_device(dev)
    ttnn.deallocate(s)
    ttnn.deallocate(z)

    def fits(fn, mb):
        s, z = fresh()
        pad = None
        try:
            if mb > 0:
                pad = pad_tensor(dev, mb)
            fn(s, z)
            ttnn.synchronize_device(dev)
            ok, why = True, ""
        except Exception as e:  # noqa: BLE001
            ok, why = False, str(e).splitlines()[0][:100]
        finally:
            for t in (pad, s, z):
                try:
                    if t is not None:
                        ttnn.deallocate(t)
                except Exception:
                    pass
        return ok, why

    res = {}
    for name, fn in workloads.items():
        ok0, why0 = fits(fn, 0.0)
        if not ok0:
            print(f"{name:<20} does not run even with no pad: {why0}", flush=True)
            res[name] = {"headroom_MB": None, "note": why0}
            continue
        lo, hi = args.lo, args.hi
        ok_hi, _ = fits(fn, hi)
        if ok_hi:
            print(f"{name:<20} headroom >= {hi:.0f} MB (probe ceiling)", flush=True)
            res[name] = {"headroom_MB": hi, "note": "at probe ceiling"}
            continue
        while hi - lo > args.tol:
            mid = (lo + hi) / 2
            ok, why = fits(fn, mid)
            if ok:
                lo = mid
            else:
                hi = mid
        res[name] = {"headroom_MB": lo, "per_core_KB": lo * 1e6 / cores / 1e3}
        print(f"{name:<20} headroom = {lo:6.1f} MB aggregate "
              f"({lo*1e6/cores/1e3:6.1f} KB/core of {per_core/1e3:.0f})", flush=True)

    pair_MB = N * N * c_z * 2 / 1e6
    print(f"\npair tensor at N={N}, c_z={c_z} = {pair_MB:.1f} MB "
          f"({pair_MB*1e6/cores/1e3:.1f} KB/core)")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": N, "c_z": c_z, "grid": [gx, gy], "cores": cores,
             "l1_per_core_B": per_core, "aggregate_L1_MB": cores * per_core / 1e6,
             "pair_tensor_MB": pair_MB, "headroom": res}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
