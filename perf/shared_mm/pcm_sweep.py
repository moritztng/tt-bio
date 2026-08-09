#!/usr/bin/env python3
"""Does the helper's M-split rule leave cores idle, and is a wrapping split actually wrong?

E1's helper only splits M when the resulting block count fits the grid, on G1's predicate that a
core taking a second block makes the reuse factory advance the batch stride wrongly. That rule
costs real occupancy: opendde's DiT attn@v has Mt = 19, prime, so the only legal per_core_M is 19
and the op runs on 16 of 110 cores. This sweeps every divisor of Mt, records cores engaged and
time, and checks each arm against ttnn's own choice -- so 'wrapping is wrong' becomes a measured
fact rather than an inherited one.
"""
import argparse, json, time

import torch
import ttnn

import tt_bio.tenstorrent as T


def mk(dev, shape, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(shape, generator=g, dtype=torch.float32) - 0.5) * 0.2
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def timed(dev, fn, iters):
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
    a = ap.parse_args()

    dev = T.get_device()
    ttnn.SetDefaultDevice(dev)
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = T.COMPUTE_GRID_MAIN
    cores = gx * gy

    rows = []
    for case in json.load(open(a.cases)):
        if case.get("arm") == "core_grid":
            continue
        ash, bsh = case["a"], case["b"]
        dt = ttnn.float32 if case.get("dtype") == "fp32" else ttnn.bfloat16
        ta, tb = mk(dev, ash, 1, dt), mk(dev, bsh, 2, dt)
        B = 1
        for d in ash[:-2]:
            B *= d
        Mt, Nt = ash[-2] // 32, bsh[-1] // 32
        ms_auto, o = timed(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg), a.iters)
        ref = ttnn.to_torch(o)
        ttnn.deallocate(o)
        arms = []
        for pcm in [d for d in range(1, Mt + 1) if Mt % d == 0]:
            h, w = T._out_subblock(pcm, Nt)
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=1,
                out_subblock_h=h, out_subblock_w=w, per_core_M=pcm, per_core_N=Nt)
            blocks = B * (Mt // pcm)
            try:
                ms, o = timed(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg,
                                                       program_config=pc), a.iters)
                got = ttnn.to_torch(o)
                ttnn.deallocate(o)
                d_ = (got.double() - ref.double()).abs()
                arms.append(dict(per_core_M=pcm, blocks=blocks, cores=min(blocks, cores),
                                 wraps=blocks > cores, ms=ms, speedup=ms_auto / ms,
                                 bit_exact=bool(torch.equal(got, ref)),
                                 maxabs=float(d_.max()),
                                 rel_l2=float(d_.norm() / ref.double().norm())))
                del got
            except Exception as e:
                arms.append(dict(per_core_M=pcm, blocks=blocks, err=str(e)[:120]))
            print(json.dumps(arms[-1]), flush=True)
        rows.append(dict(case=case["name"], B=B, Mt=Mt, Nt=Nt, grid=[gx, gy],
                         ms_auto=ms_auto, arms=arms))
        del ref
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        print("== " + case["name"] + " done", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
