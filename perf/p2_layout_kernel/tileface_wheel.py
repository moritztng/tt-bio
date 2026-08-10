#!/usr/bin/env python3
"""P4 / p2-layout-kernel -- the SAME arms on the production ttnn 0.68.0 wheel, same card.

`tileface.py` runs the prototype, which only exists in the `tt-metal-fused` build (a June
tt-metal). A ratio taken there is only transferable to production if the production wheel's
baseline is known on the SAME card, so this reruns the baseline and floor arms on the wheel.

It also runs the cheap no-kernel lever the brief asks for first: for a DRAM-resident source, is
`clone(DRAM->L1)` + `permute(L1->L1)` + `clone(L1->DRAM)` faster than the direct DRAM->DRAM
permute, and is it bit-exact?

    TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=<wheel>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    python3 perf/p2_layout_kernel/tileface_wheel.py --n 320
"""
import argparse, json, time
from pathlib import Path

import torch
import ttnn

BF16 = 2


def timeit(device, fn, reps=12, warmup=3):
    for _ in range(warmup):
        ttnn.deallocate(fn())
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


def mem(k):
    return ttnn.L1_MEMORY_CONFIG if k == "l1" else ttnn.DRAM_MEMORY_CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="perf/p2_layout_kernel/tileface_wheel_n320.json")
    a = ap.parse_args()
    N, C = a.n, a.c
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    ncores = g.x * g.y
    ntiles = N * N * C // 1024
    one_way = N * N * C * BF16
    R = {"n": N, "c": C, "ttnn": getattr(ttnn, "__version__", "wheel"), "cores": ncores,
         "tiles": ntiles, "arms": {}, "parity": {}}

    def rec(name, us, note=""):
        R["arms"][name] = {"us": round(us, 2),
                           "us_per_tile_per_core": round(us * ncores / ntiles, 4),
                           "gbs_two_way": round(2 * one_way / us / 1e3, 1), "cores": ncores,
                           "note": note}
        print(f"{name:44s} {us:9.2f} us   {us*ncores/ntiles:7.4f} us/tile/core   "
              f"{2*one_way/us/1e3:7.1f} GB/s  {note}")

    torch.manual_seed(0)
    xi = torch.randn(1, N, N, C)
    xo = torch.randn(1, C, N, N)
    dev = lambda t, k: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device,
                                       dtype=ttnn.bfloat16, memory_config=mem(k))
    a_l1, a_dr = dev(xi, "l1"), dev(xi, "dram")
    b_l1, b_dr = dev(xo, "l1"), dev(xo, "dram")
    print(f"# wheel ttnn {R['ttnn']}, grid {g.x}x{g.y}={ncores}, {ntiles} tiles, "
          f"{one_way/1e6:.3f} MB one way\n")

    rec("floor: clone L1->L1", timeit(device, lambda: ttnn.clone(a_l1, memory_config=mem("l1")), a.reps))
    rec("floor: transpose(-2,-1) L1->L1",
        timeit(device, lambda: ttnn.transpose(a_l1, -2, -1, memory_config=mem("l1")), a.reps))
    rec("floor: clone DRAM->L1", timeit(device, lambda: ttnn.clone(a_dr, memory_config=mem("l1")), a.reps))
    rec("floor: clone L1->DRAM", timeit(device, lambda: ttnn.clone(a_l1, memory_config=mem("dram")), a.reps))

    for tag, perm, s_l1, s_dr in (("in-move (0,3,1,2)", (0, 3, 1, 2), a_l1, a_dr),
                                  ("out-move (0,2,3,1)", (0, 2, 3, 1), b_l1, b_dr)):
        for src, s in (("l1", s_l1), ("dram", s_dr)):
            for dst in ("l1", "dram"):
                rec(f"permute {tag} {src}->{dst}",
                    timeit(device, lambda s=s, p=perm, d=dst: ttnn.permute(s, p, memory_config=mem(d)),
                           a.reps))

        # the cheap no-kernel lever: stage a DRAM source through L1 in tiles
        def staged(s=s_dr, p=perm):
            t = ttnn.clone(s, memory_config=mem("l1"))
            u = ttnn.permute(t, p, memory_config=mem("l1"))
            ttnn.deallocate(t)
            v = ttnn.clone(u, memory_config=mem("dram"))
            ttnn.deallocate(u)
            return v
        rec(f"STAGED {tag} dram->L1->L1->dram", timeit(device, staged, a.reps),
            "clone in, permute in L1, clone out")
        ref = ttnn.to_torch(ttnn.permute(s_dr, perm, memory_config=mem("dram")))
        got = ttnn.to_torch(staged())
        R["parity"][f"staged {tag}"] = bool(torch.equal(got, ref))
        print(f"    torch.equal(staged, direct) = {R['parity'][f'staged {tag}']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=2))
    print("\nwrote", a.out)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
