#!/usr/bin/env python3
"""Time tt_bio.tenstorrent.batched_matmul against plain ttnn.matmul on census shapes.

Cases come from a JSON list of {name, a, b, note} written from the live census, so this measures
the shape the model actually issues rather than one a probe reconstructed. A program config that
does not fit L1 falls back silently, so the shipped path is confirmed by timing, never by the diff.

  TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT python3 perf/shared_mm/mm_shipped.py --cases c.json --out o.json
"""
import argparse, json, time

import torch
import ttnn

import tt_bio.tenstorrent as T


def mk(dev, shape, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(shape, generator=g, dtype=torch.float32) - 0.5) * 0.2
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def time_arm(dev, fn, iters):
    o = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / iters
    for o in outs[1:]:
        ttnn.deallocate(o)
    return dt * 1e3, outs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    dev = T.get_device()
    ttnn.SetDefaultDevice(dev)
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = T.COMPUTE_GRID_MAIN
    print("COMPUTE_GRID_MAIN", T.COMPUTE_GRID_MAIN, flush=True)

    rows = []
    for case in json.load(open(a.cases)):
        ash, bsh = case["a"], case["b"]
        ta, tb = mk(dev, ash, 1), mk(dev, bsh, 2)
        B = 1
        for d in ash[:-2]:
            B *= d
        Mt, Kt, Nt = ash[-2] // 32, ash[-1] // 32, bsh[-1] // 32
        pc = T._batched_matmul_program_config(B, Mt, Nt, T.COMPUTE_GRID_MAIN)
        blocks = B * (Mt // pc.per_core_M) if pc is not None else None
        row = dict(case=case["name"], note=case.get("note", ""), a=ash, b=bsh,
                   B=B, Mt=Mt, Kt=Kt, Nt=Nt, calls_per_fold=case.get("calls"),
                   per_core_M=(pc.per_core_M if pc else None),
                   per_core_N=(pc.per_core_N if pc else None),
                   in0_block_w=(pc.in0_block_w if pc else None),
                   out_subblock=[pc.out_subblock_h, pc.out_subblock_w] if pc else None,
                   grid=[gx, gy], blocks=blocks,
                   cores_engaged=(min(blocks, gx * gy) if blocks else None))

        auto, new = [], []
        for r in range(a.repeats):
            ms, o_auto = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg), a.iters)
            auto.append(ms)
            if r == 0:
                t_auto = ttnn.to_torch(o_auto)
            ttnn.deallocate(o_auto)
            ms, o_new = time_arm(dev, lambda: T.batched_matmul(ta, tb, compute_kernel_config=cfg), a.iters)
            new.append(ms)
            if r == 0:
                t_new = ttnn.to_torch(o_new)
            ttnn.deallocate(o_new)

        ms_auto, ms_new = min(auto), min(new)
        d = (t_new.double() - t_auto.double()).abs()
        row.update(ms_auto=ms_auto, ms_shipped=ms_new, speedup=ms_auto / ms_new,
                   ms_auto_all=auto, ms_shipped_all=new,
                   bit_exact=bool(torch.equal(t_new, t_auto)),
                   maxabs_vs_auto=float(d.max()),
                   rel_l2_vs_auto=float(d.norm() / t_auto.double().norm()))
        del t_auto, t_new
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
