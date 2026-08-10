#!/usr/bin/env python3
"""P4 / p2-layout-kernel -- can a tt-metal kernel move tile faces at compute-engine rates?

Prototype + probes for the two protenix-v2 trimul channel moves at 298 aa (N=320, C=32):

    in-move   [1,320,320,32] -> [1,32,320,320]   permute(0,3,1,2)   3200 tiles
    out-move  [1,32,320,320] -> [1,320,320,32]   permute(0,2,3,1)   3200 tiles

Every roof and every baseline here is measured in this same process, on this card, against the
same ttnn build as the prototype. Nothing is inherited.

    TT_VISIBLE_DEVICES=2 \
    TT_MESH_GRAPH_DESC_PATH=<fused>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_METAL_HOME=<fused> PYTHONPATH=<fused>/ttnn \
    python3 perf/p2_layout_kernel/tileface.py --n 320
"""
import argparse, json, time
from pathlib import Path

import torch
import ttnn

BF16 = 2  # bytes


def timeit(device, fn, reps=12, warmup=3):
    """Sync on BOTH sides of every timed region (charter §4.4). Report the mean of the fastest half."""
    for _ in range(warmup):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    ts.sort()
    return sum(ts[: max(1, len(ts) // 2)]) / max(1, len(ts) // 2)


def mem(kind):
    return ttnn.L1_MEMORY_CONFIG if kind == "l1" else ttnn.DRAM_MEMORY_CONFIG


def to_dev(device, t, kind):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16,
                           memory_config=mem(kind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="perf/p2_layout_kernel/tileface_n320.json")
    a = ap.parse_args()
    N, C = a.n, a.c
    device = ttnn.open_device(device_id=0)
    grid = device.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    ntiles = (N // 32) * (N // 32) * (C // 32) * 32  # = N*N*C/1024
    assert ntiles == N * N * C // 1024
    bytes_one_way = N * N * C * BF16
    R = {"n": N, "c": C, "grid": [grid.x, grid.y], "cores": ncores, "tiles": ntiles,
         "bytes_one_way": bytes_one_way, "has_reblock": hasattr(ttnn.experimental, "reblock_permute"),
         "arms": {}}

    def rec(name, us, cores, note="", tiles=None):
        t = tiles or ntiles
        R["arms"][name] = {
            "us": round(us, 2),
            "us_per_tile_per_core": round(us * cores / t, 4),
            "gbs_two_way": round(2 * bytes_one_way / us / 1e3, 1),
            "cores": cores, "note": note,
        }
        print(f"{name:38s} {us:9.2f} us   {us*cores/t:7.4f} us/tile/core   "
              f"{2*bytes_one_way/us/1e3:7.1f} GB/s(2-way)  cores={cores}  {note}")

    torch.manual_seed(0)
    x_in = torch.randn(1, N, N, C, dtype=torch.float32)   # in-move source
    x_out = torch.randn(1, C, N, N, dtype=torch.float32)  # out-move source

    print(f"# card grid {grid.x}x{grid.y} = {ncores} cores; {ntiles} tiles; "
          f"{bytes_one_way/1e6:.3f} MB one way\n")

    # ---------------------------------------------------------------- roofs, this card, this pass
    print("== roofs measured on this card, this pass (not inherited) ==")
    a_l1 = to_dev(device, x_in, "l1")
    rec("roof: clone L1->L1 (same 3200 tiles)",
        timeit(device, lambda: ttnn.clone(a_l1, memory_config=mem("l1")), a.reps), ncores,
        "the floor: same bytes, no reorder")
    rec("roof: transpose(-2,-1) L1->L1 (in-tile)",
        timeit(device, lambda: ttnn.transpose(a_l1, -2, -1, memory_config=mem("l1")), a.reps), ncores,
        "compute-engine in-tile move")
    rec("roof: clone L1->DRAM",
        timeit(device, lambda: ttnn.clone(a_l1, memory_config=mem("dram")), a.reps), ncores)
    a_dr = to_dev(device, x_in, "dram")
    rec("roof: clone DRAM->DRAM",
        timeit(device, lambda: ttnn.clone(a_dr, memory_config=mem("dram")), a.reps), ncores)
    rec("roof: clone DRAM->L1",
        timeit(device, lambda: ttnn.clone(a_dr, memory_config=mem("l1")), a.reps), ncores)

    # ------------------------------------------------------- E0: the cheap test, buffer-type ladder
    print("\n== E0 cheap test: does staging through L1 / bank locality matter? (no kernel) ==")
    for src in ("l1", "dram"):
        s = a_l1 if src == "l1" else a_dr
        for dst in ("l1", "dram"):
            rec(f"E0 permute(0,3,1,2) {src}->{dst}",
                timeit(device, lambda s=s, dst=dst: ttnn.permute(s, (0, 3, 1, 2),
                                                                 memory_config=mem(dst)), a.reps),
                ncores)
    b_l1 = to_dev(device, x_out, "l1")
    b_dr = to_dev(device, x_out, "dram")
    for src, s in (("l1", b_l1), ("dram", b_dr)):
        for dst in ("l1", "dram"):
            rec(f"E0 permute(0,2,3,1) {src}->{dst}",
                timeit(device, lambda s=s, dst=dst: ttnn.permute(s, (0, 2, 3, 1),
                                                                 memory_config=mem(dst)), a.reps),
                ncores)

    # -------------------------------------------------------------- E1/E2: baseline vs the prototype
    print("\n== E1 production baseline vs E2 prototype kernel ==")
    ref_in = ttnn.to_torch(ttnn.permute(a_l1, (0, 3, 1, 2), memory_config=mem("l1")))
    ref_out = ttnn.to_torch(ttnn.permute(b_l1, (0, 2, 3, 1), memory_config=mem("l1")))
    R["parity"] = {}
    if R["has_reblock"]:
        for src, s, name, op, ref in (
                ("l1", a_l1, "in-move", ttnn.experimental.reblock_permute, ref_in),
                ("dram", a_dr, "in-move", ttnn.experimental.reblock_permute, ref_in),
                ("l1", b_l1, "out-move", ttnn.experimental.reblock_permute_back, ref_out),
                ("dram", b_dr, "out-move", ttnn.experimental.reblock_permute_back, ref_out)):
            for dst in ("l1", "dram"):
                try:
                    got = ttnn.to_torch(op(s, memory_config=mem(dst)))
                    eq = bool(torch.equal(got, ref))
                    R["parity"][f"reblock {name} {src}->{dst}"] = eq
                    rec(f"E2 reblock {name} {src}->{dst}",
                        timeit(device, lambda s=s, op=op, dst=dst: op(s, memory_config=mem(dst)),
                               a.reps), ncores, f"torch.equal={eq}")
                except Exception as e:  # noqa: BLE001
                    print(f"E2 reblock {name} {src}->{dst}: FAILED {type(e).__name__}: {e}")
                    R["arms"][f"E2 reblock {name} {src}->{dst}"] = {"error": str(e)[:200]}
    else:
        print("!! reblock ops absent from this ttnn build -- wrong PYTHONPATH?")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print("\nwrote", a.out)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
