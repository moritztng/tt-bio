"""Is there a config arm that is BOTH fast and bit-exact against ttnn's own choice?

E1 ships in0_block_w=1, which is bit-exact only where ttnn's own choice also takes one K block.
G1 reports its batched-matmul fix is bit-exact by CIF sha on five OF3 targets, so a bit-exact arm
exists; this finds which one, per shape, at an iteration count that is actually a measurement.

Arms: auto (baseline), block_k1 (what E1 ships), block_kall (in0_block_w=Kt), core_grid (let ttnn
choose everything but the grid).
"""
import argparse, json, time

import torch
import ttnn

import tt_bio.tenstorrent as T

CASES = [
    ("triatt_qk", [298, 4, 320, 32], [298, 4, 32, 320]),
    ("triatt_av", [298, 4, 320, 320], [298, 4, 320, 32]),
    ("atom_qk", [1, 75, 4, 32, 32], [1, 75, 4, 32, 128]),
    ("atom_av", [1, 75, 4, 32, 128], [1, 75, 4, 128, 32]),
    ("dit_qk", [1, 16, 320, 64], [1, 16, 64, 320]),
    ("dit_av", [1, 16, 320, 320], [1, 16, 320, 64]),
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
    keep = outs[0]
    for o in outs[1:]:
        ttnn.deallocate(o)
    return dt * 1e3, keep


def cfg_with(pc_src, in0_block_w):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=pc_src.compute_with_storage_grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=pc_src.out_subblock_h,
        out_subblock_w=pc_src.out_subblock_w,
        per_core_M=pc_src.per_core_M,
        per_core_N=pc_src.per_core_N,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    T._configure_active_compute_grid(dev)
    ck = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = T.COMPUTE_GRID_MAIN

    rows = []
    for name, ash, bsh in CASES:
        ta, tb = mk(dev, ash, 1), mk(dev, bsh, 2)
        B = 1
        for d in ash[:-2]:
            B *= d
        Mt, Kt, Nt = ash[-2] // 32, ash[-1] // 32, bsh[-1] // 32
        pc1 = T._batched_matmul_program_config(B, Mt, Kt, Nt, T.COMPUTE_GRID_MAIN)
        row = dict(case=name, B=B, Mt=Mt, Kt=Kt, Nt=Nt, arms={})

        ms, out = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=ck), a.iters)
        ref = ttnn.to_torch(out)
        ttnn.deallocate(out)
        row["arms"]["auto"] = dict(ms=ms, bit_exact=True)

        arms = [("block_k1", dict(program_config=pc1)),
                ("core_grid", dict(core_grid=ttnn.CoreGrid(y=gy, x=gx)))]
        if Kt > 1:
            arms.insert(1, ("block_kall", dict(program_config=cfg_with(pc1, Kt))))
            for d in (2, 5):
                if Kt % d == 0 and d != Kt:
                    arms.insert(2, ("block_k%d" % d, dict(program_config=cfg_with(pc1, d))))
        for tag, kw in arms:
            try:
                ms, out = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=ck, **kw),
                                   a.iters)
                t = ttnn.to_torch(out)
                ttnn.deallocate(out)
                row["arms"][tag] = dict(ms=ms, bit_exact=bool(torch.equal(t, ref)),
                                        speedup=row["arms"]["auto"]["ms"] / ms)
                del t
            except Exception as e:
                row["arms"][tag] = dict(error=repr(e)[:200])
        del ref
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
