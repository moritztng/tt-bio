"""Time the SHIPPED tt_bio.tenstorrent.batched_matmul against plain ttnn.matmul.

The shapes are the padded per-fold shapes the census recorded for a 298 aa OpenFold3 fold, so
this measures the config the model will actually issue, not a config a probe rebuilt. A
core_grid/program_config that does not fit L1 falls back to the slow path with no error, which is
why the shipped path has to be confirmed by timing.

  TT_VISIBLE_DEVICES=1 PYTHONPATH=$WT python3 perf/of3_mm/mm_shipped.py --out ...json
"""
import argparse, json, time

import torch
import ttnn

import tt_bio.tenstorrent as T

CASES = [
    ("triatt_qk", [298, 4, 320, 32], [298, 4, 32, 320], "trunk tri-att q@kT (tenstorrent.py:259)"),
    ("triatt_av", [298, 4, 320, 320], [298, 4, 320, 32], "trunk tri-att attn@v (tenstorrent.py:272)"),
    ("atom_qk", [1, 75, 4, 32, 32], [1, 75, 4, 32, 128], "atom-tf q@kT rank-5 (:153)"),
    ("atom_av", [1, 75, 4, 32, 128], [1, 75, 4, 128, 32], "atom-tf attn@v rank-5 (:159)"),
    ("dit_qk", [1, 16, 320, 64], [1, 16, 64, 320], "DiT q@kT (:184)"),
    ("dit_av", [1, 16, 320, 320], [1, 16, 320, 64], "DiT attn@v (:193)"),
]


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    T._configure_active_compute_grid(dev)   # same grid the fold runs on
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print("COMPUTE_GRID_MAIN", T.COMPUTE_GRID_MAIN, flush=True)

    rows = []
    for name, ash, bsh, note in CASES:
        ta, tb = mk(dev, ash, 1), mk(dev, bsh, 2)
        B = 1
        for d in ash[:-2]:
            B *= d
        Mt, Kt, Nt = ash[-2] // 32, ash[-1] // 32, bsh[-1] // 32
        pc = T._batched_matmul_program_config(B, Mt, Kt, Nt, T.COMPUTE_GRID_MAIN)
        gx, gy = T.COMPUTE_GRID_MAIN
        blocks = B * (Mt // pc.per_core_M) if pc is not None else None
        row = dict(case=name, note=note, a=ash, b=bsh, B=B, Mt=Mt, Kt=Kt, Nt=Nt,
                   per_core_M=(pc.per_core_M if pc else None),
                   per_core_N=(pc.per_core_N if pc else None),
                   in0_block_w=(pc.in0_block_w if pc else None),
                   out_subblock=[pc.out_subblock_h, pc.out_subblock_w] if pc else None,
                   grid=[gx, gy], blocks=blocks,
                   cores_engaged=(min(blocks, gx * gy) if blocks else None))

        ms_auto, out_auto = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg),
                                     a.iters)
        t_auto = ttnn.to_torch(out_auto)
        ttnn.deallocate(out_auto)
        ms_new, out_new = time_arm(dev, lambda: T.batched_matmul(ta, tb, compute_kernel_config=cfg),
                                   a.iters)
        t_new = ttnn.to_torch(out_new)
        ttnn.deallocate(out_new)

        d = (t_new.double() - t_auto.double()).abs()
        row.update(ms_auto=ms_auto, ms_shipped=ms_new, speedup=ms_auto / ms_new,
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
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
