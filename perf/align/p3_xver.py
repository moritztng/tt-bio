#!/usr/bin/env python3
"""Cross-version arm: the SAME contraction A/B and the SAME output tensor, run under the 0.68.0
production wheel and under the patched source build, so the alignment penalty and the arithmetic
can be compared version to version on one chip.

No tt_bio import: the program config is spelled out so the script runs under either ttnn.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

GRID = (11, 10)


def ckc():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)


def trimul_pc(kt=10):
    gx, gy = GRID
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=max(d for d in range(min(10, kt), 0, -1) if kt % d == 0),
        out_subblock_h=1, out_subblock_w=1,
        out_block_h=-(-kt // gy), out_block_w=-(-kt // gx),
        per_core_M=-(-kt // gy), per_core_N=-(-kt // gx),
        transpose_mcast=False, fused_activation=None, fuse_batch=False)


def timeit(dev, fn, reps=20, warm=5):
    for _ in range(warm):
        o = fn()
        del o
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(reps)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / reps
    del outs
    return dt * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    L1 = ttnn.L1_MEMORY_CONFIG
    dev = ttnn.open_device(device_id=0)
    out = {"tag": a.tag}
    try:
        g = torch.Generator().manual_seed(1234)
        at = torch.randn(1, 32, 320, 320, generator=g, dtype=torch.float32)
        bt = torch.randn(1, 32, 320, 320, generator=g, dtype=torch.float32)
        at[:, :, 298:, :] = 0.0
        at[:, :, :, 298:] = 0.0
        bt[:, :, 298:, :] = 0.0
        bt[:, :, :, 298:] = 0.0
        pc = trimul_pc()
        arms = {}
        # logical 320 (aligned) and logical 298 (production), identical bytes both arms:
        # the 298 arm is built by slicing the SAME zero-tailed host tensor, so the padded
        # tile contents are identical and only the logical metadata differs.
        for name, sl in (("logical_320", slice(None)), ("logical_298", slice(0, 298))):
            aa = ttnn.from_torch(at[:, :, sl, sl], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=L1)
            bb = ttnn.from_torch(bt[:, :, sl, sl], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=L1)
            us = timeit(dev, lambda aa=aa, bb=bb: ttnn.matmul(
                aa, bb, compute_kernel_config=ckc(), memory_config=L1, program_config=pc,
                dtype=ttnn.bfloat16), reps=20, warm=5)
            r = ttnn.to_torch(ttnn.matmul(aa, bb, compute_kernel_config=ckc(), memory_config=L1,
                                          program_config=pc, dtype=ttnn.bfloat16))
            arms[name] = {"us": us, "logical": list(aa.shape), "padded": list(aa.padded_shape)}
            if a.dump:
                torch.save(r[:, :, :298, :298].contiguous(), f"{a.dump}.{name}.pt")
            ttnn.deallocate(aa)
            ttnn.deallocate(bb)
            print(f"  {name:12s} {us:8.2f} us  logical {list(aa.shape)}")
        out["arms"] = arms
        out["ratio_298_over_320"] = arms["logical_298"]["us"] / arms["logical_320"]["us"]
        print(f"  ratio 298/320 = {out['ratio_298_over_320']:.4f}")
        # the fill, and a whole-tensor clone for scale
        x = ttnn.from_torch(at[:, :, :298, :298], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=L1)
        try:
            out["fill_implicit_tile_padding_us"] = timeit(
                dev, lambda: ttnn.fill_implicit_tile_padding(x, 0.0), reps=50, warm=10)
        except Exception as e:                                   # noqa: BLE001
            out["fill_error"] = str(e)[:300]
        out["clone_us"] = timeit(dev, lambda: ttnn.clone(x, memory_config=L1), reps=20, warm=5)
        print(f"  fill {out.get('fill_implicit_tile_padding_us')} clone {out['clone_us']}")
    finally:
        Path(a.out).write_text(json.dumps(out, indent=1))
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
