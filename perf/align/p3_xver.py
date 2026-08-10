#!/usr/bin/env python3
"""Cross-version arm: the SAME contraction A/B and the SAME output tensor, run under the 0.68.0
production wheel and under the patched source build, so the alignment penalty and the arithmetic
can be compared version to version on one chip.

No tt_bio import: the program config is spelled out so the script runs under either ttnn.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Optional tree override: the built tt-metal trees on this host are reached through an editable
# install that registers a meta-path finder, so PYTHONPATH cannot redirect `ttnn` on its own.
# Strip the finder, then put the requested tree's package directory first.
_TREE = os.environ.get("P3_TREE")
if _TREE:
    sys.meta_path = [f for f in sys.meta_path
                     if "ttnn" not in getattr(f, "__name__", type(f).__name__).lower()]
    sys.path.insert(0, os.path.join(_TREE, "ttnn"))

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



def sdpa_pc(chunk=64):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=GRID, exp_approx_mode=False,
                                  q_chunk_size=chunk, k_chunk_size=chunk)


def mk(dev, shape, mc, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ttnn.from_torch(torch.randn(*shape, generator=g, dtype=torch.float32),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=mc)


def other_sites(dev, out):
    """The three sites P2 measured beside the contraction: SDPA @1629, attn@v @378 and softmax.

    Same construction as P2: the PADDED shape is held fixed across each pair and only the logical
    length of the reduced/contracted axis moves.
    """
    L1 = ttnn.L1_MEMORY_CONFIG
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    sites = {}
    for tag, S in (("sdpa_298", 298), ("sdpa_320", 320)):
        try:
            q = mk(dev, (298, 8, S, 32), DRAM)
            k = mk(dev, (298, 8, S, 32), DRAM, seed=1)
            v = mk(dev, (298, 8, S, 32), DRAM, seed=2)
            sites[tag] = timeit(dev, lambda q=q, k=k, v=v: ttnn.transformer.
                                scaled_dot_product_attention(q, k, v, is_causal=False, scale=0.176,
                                                             program_config=sdpa_pc()),
                                reps=6, warm=2)
            for t in (q, k, v):
                ttnn.deallocate(t)
        except Exception as e:                                   # noqa: BLE001
            sites[tag] = {"error": str(e)[:200]}
    for tag, S in (("attnv_298", 298), ("attnv_320", 320)):
        try:
            a_ = mk(dev, (1, 16, S, S), DRAM)
            b_ = mk(dev, (1, 16, S, 32), DRAM, seed=1)
            sites[tag] = timeit(dev, lambda a_=a_, b_=b_: ttnn.matmul(
                a_, b_, compute_kernel_config=ckc(), memory_config=DRAM,
                dtype=ttnn.bfloat16), reps=20, warm=5)
            ttnn.deallocate(a_)
            ttnn.deallocate(b_)
        except Exception as e:                                   # noqa: BLE001
            sites[tag] = {"error": str(e)[:200]}
    for tag, S in (("softmax_298", 298), ("softmax_320", 320)):
        try:
            x = mk(dev, (1, 32, 320, S), L1)
            sites[tag] = timeit(dev, lambda x=x: ttnn.softmax(x, dim=-1, memory_config=L1),
                                reps=10, warm=3)
            ttnn.deallocate(x)
        except Exception as e:                                   # noqa: BLE001
            sites[tag] = {"error": str(e)[:200]}
    # SDPA is the biggest row in the trunk and it did not get the matmul fix, so sweep its own
    # program config before calling a version difference a regression: a version may simply want a
    # different chunk size.
    sweep = {}
    for chunk in (32, 64, 128, 256):
        for tag, S in ((f"sdpa_c{chunk}_298", 298), (f"sdpa_c{chunk}_320", 320)):
            try:
                q = mk(dev, (298, 8, S, 32), DRAM)
                k = mk(dev, (298, 8, S, 32), DRAM, seed=1)
                v = mk(dev, (298, 8, S, 32), DRAM, seed=2)
                sweep[tag] = timeit(dev, lambda q=q, k=k, v=v, c=chunk: ttnn.transformer.
                                    scaled_dot_product_attention(q, k, v, is_causal=False,
                                                                 scale=0.176,
                                                                 program_config=sdpa_pc(c)),
                                    reps=4, warm=2)
                for t in (q, k, v):
                    ttnn.deallocate(t)
            except Exception as e:                               # noqa: BLE001
                sweep[tag] = {"error": str(e)[:160]}
        a_, b_ = sweep.get(f"sdpa_c{chunk}_298"), sweep.get(f"sdpa_c{chunk}_320")
        if isinstance(a_, float) and isinstance(b_, float):
            print(f"  sdpa chunk {chunk:3d}: 298 {a_:9.2f} us   320 {b_:9.2f} us   "
                  f"ratio {a_ / b_:.4f}")
        else:
            print(f"  sdpa chunk {chunk:3d}: refused")
    out["sdpa_chunk_sweep"] = sweep
    out["other_sites"] = sites
    for a_, b_ in (("sdpa_298", "sdpa_320"), ("attnv_298", "attnv_320"),
                   ("softmax_298", "softmax_320")):
        x, y = sites.get(a_), sites.get(b_)
        if isinstance(x, float) and isinstance(y, float):
            print(f"  {a_[:-4]:9s} 298 {x:9.2f} us   320 {y:9.2f} us   ratio {x / y:.4f}   "
                  f"delta {x - y:+8.2f} us")
            out.setdefault("other_ratios", {})[a_[:-4]] = {
                "us_298": x, "us_320": y, "ratio": x / y, "delta_us": x - y}


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
        ttnn.deallocate(x)
        out["ttnn_file"] = ttnn.__file__
        other_sites(dev, out)
    finally:
        Path(a.out).write_text(json.dumps(out, indent=1))
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
