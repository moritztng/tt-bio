"""AF2's pair-transition linears at the AF2-IG token count, against their own roofline.

The WH op census at 195 tokens puts the two transition linears at 0.26 + 0.19 s per fold, about
4x off both the compute and the DRAM roof for their shape. The suspect is the 4D form: the pair
tensor is [1, N, N, 128], so the matmul runs as N batches of [N,128]x[128,512], and a batch that
small does not fill the grid. A 2D [N*N, 128] matmul is the same arithmetic on the whole grid,
but reaching it needs the row axis tile-aligned (N=224 bucket), which is a model change.

So measure, before changing anything: the shipped 4D form, the 2D form at the same element count
(the upper bound a bucketed reshape could buy), and the 4D form at the 224 bucket.

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N python perf/spd/af2_transition_mm.py [N...]
"""
import json
import os
import sys
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

NS = [int(x) for x in sys.argv[1:]] or [195, 224]
C, F = 128, 512
REPS = 20
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                       fp32_dest_acc_en=False, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def timed(fn):
    """Warm twice, then the median of REPS synchronised calls, in seconds."""
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        t0 = time.monotonic()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.monotonic() - t0)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


torch.manual_seed(0)
w1, b1 = tt(torch.randn(C, F) * 0.05), tt(torch.randn(1, F) * 0.05)
w2, b2 = tt(torch.randn(F, C) * 0.05), tt(torch.randn(1, C) * 0.05)
rows = []
for n in NS:
    z4 = tt(torch.randn(1, n, n, C) * 0.5)
    z2 = tt(torch.randn(1, 1, n * n, C) * 0.5)
    arms = {
        "fc1_4d": lambda z=z4: ttnn.linear(z, w1, bias=b1, activation="relu",
                                           compute_kernel_config=ckc,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "fc1_2d": lambda z=z2: ttnn.linear(z, w1, bias=b1, activation="relu",
                                           compute_kernel_config=ckc,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }
    h4 = ttnn.linear(z4, w1, bias=b1, activation="relu", compute_kernel_config=ckc,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h2 = ttnn.linear(z2, w1, bias=b1, activation="relu", compute_kernel_config=ckc,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    arms["fc2_4d"] = lambda h=h4: ttnn.linear(h, w2, bias=b2, compute_kernel_config=ckc,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    arms["fc2_2d"] = lambda h=h2: ttnn.linear(h, w2, bias=b2, compute_kernel_config=ckc,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # The batch folded into M without a data move: a 1D program config with fuse_batch runs the
    # padded B*Mt row tiles as one matmul. Each config either runs or is refused by the op.
    mt = n * ((n + 31) // 32)
    for cfg_name, grid, per_m, sub_w in (("fold64", (8, 8), -(-mt // 64), 8), ("fold64w4", (8, 8), -(-mt // 64), 4)):
        for leg, (src, w, b, k, nt, act) in (("fc1", (z4, w1, b1, C // 32, F // 32, "relu")),
                                             ("fc2", (h4, w2, b2, F // 32, C // 32, None))):
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid), in0_block_w=min(k, 4),
                out_subblock_h=1, out_subblock_w=min(sub_w, nt), per_core_M=per_m, per_core_N=nt,
                fuse_batch=True, fused_activation=None, mcast_in0=False)
            arms[f"{leg}_{cfg_name}"] = (lambda src=src, w=w, b=b, cfg=cfg, act=act:
                                         ttnn.linear(src, w, bias=b, activation=act, program_config=cfg,
                                                     compute_kernel_config=ckc,
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG))
    # What the 2D form would cost if it had to move data: a reshape of the pair tensor there and back.
    # Its own source and a clone, so deallocating the result can never free a tensor other arms read.
    zr = tt(torch.randn(1, n, n, C) * 0.5)
    arms["reshape_to_2d"] = lambda z=zr: ttnn.clone(ttnn.reshape(z, (1, 1, n * n, C)))
    arms["ln_4d"] = lambda z=z4: ttnn.layer_norm(z, epsilon=1e-5, compute_kernel_config=ckc,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ref1 = ttnn.to_torch(h4).float()
    for name, fn in arms.items():
        try:
            s = timed(fn)
        except RuntimeError as exc:
            print(f"N={n:4d} {name:14} refused: {str(exc).splitlines()[0][:120]}", flush=True)
            rows.append(dict(n=n, arm=name, refused=str(exc)[:300]))
            continue
        if name.startswith("fc1_fold"):
            o = fn()
            same = torch.equal(ttnn.to_torch(o).float(), ref1)
            ttnn.deallocate(o)
            print(f"N={n:4d} {name:14} equal to fc1_4d: {same}", flush=True)
        k, m = (C, F) if name.startswith("fc1") else ((F, C) if name.startswith("fc2") else (C, 0))
        if name.startswith("reshape"):
            m = 0
        flops = 2.0 * n * n * k * m
        rows.append(dict(n=n, arm=name, s=s, gflops=flops / s / 1e9 if m else None))
        print(f"N={n:4d} {name:14} {s * 1e3:8.3f} ms"
              + (f"  {flops / s / 1e9:8.1f} GFLOP/s" if m else ""), flush=True)
    for t in (z4, z2, h4, h2, zr):
        ttnn.deallocate(t)

out = os.environ.get("OUT")
if out:
    with open(out, "w") as f:
        json.dump(dict(host=os.uname().nodename, chip=os.environ.get("TT_VISIBLE_DEVICES"),
                       aiclk=T.aiclk() if hasattr(T, "aiclk") else None, rows=rows), f, indent=1)
    print(f"wrote {out}", flush=True)
