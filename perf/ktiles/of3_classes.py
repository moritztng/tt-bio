#!/usr/bin/env python3
"""E1 + E4 acceptance: do the OpenFold3 classes take the batched program config, and is it exact?

Shapes are the ones perf/ktiles/of3_shapes.py read off a real 298 aa OF3 fold, not derived by
hand. Each rank-4 class is checked for the chooser's decision, the block-stride rule, and
torch.equal against the plain ttnn.matmul it would replace.

The rank-5 class runs LAST and its result is flushed before it runs: a program config ttnn
rejects raises TT_FATAL, which aborts the process rather than raising in Python, so it cannot be
wrapped in try/except and anything after it would never print.
"""
import json, sys
import torch
import ttnn

from tt_bio import tenstorrent as T

F32, BF16 = ttnn.float32, ttnn.bfloat16

# (label, a shape, b shape, dtype, out dtype) -- every batched matmul class a 298 aa OF3 fold issues.
CASES = [
    ("of3 DiT AV            (diffusion_transformer.py:193)", (1, 16, 320, 320), (1, 16, 320, 64), F32, None),
    ("of3 DiT QK^T          (declines, Nt=10)", (1, 16, 320, 64), (1, 16, 64, 320), F32, None),
    ("of3 triatt AV         (tenstorrent.py:269)", (298, 4, 298, 298), (298, 4, 298, 32), BF16, BF16),
    ("of3 triatt QK^T       (declines, Nt=10)", (298, 4, 298, 32), (298, 4, 32, 298), BF16, None),
    ("of3 APB AV            (tenstorrent.py:269)", (1, 16, 298, 298), (1, 16, 298, 32), BF16, BF16),
    ("of3 APB QK^T          (declines, Nt=10)", (1, 16, 298, 32), (1, 16, 32, 298), BF16, None),
    ("of3 trimul-class      (declines, Nt=10)", (1, 64, 298, 298), (1, 64, 298, 298), BF16, BF16),
]
RANK5 = ("of3 atom AV  (atom_transformer.py)", (1, 75, 4, 32, 32), (1, 75, 4, 32, 128), F32, None)


def build(dev, shape, dt):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cfg_for(sa, sb, dt):
    batch = 1
    for d in sa[:-2]:
        batch *= d
    return batch, T._batched_matmul_config(
        batch, -(-sa[-2] // 32), -(-sa[-1] // 32), -(-sb[-1] // 32), 4 if dt == F32 else 2)


def main():
    dev = ttnn.open_device(device_id=0)
    out = []
    try:
        T._configure_active_compute_grid(dev)
        gx, gy = T.COMPUTE_GRID_MAIN
        ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        for label, sa, sb, dt, odt in CASES:
            torch.manual_seed(0)
            a, b = build(dev, sa, dt), build(dev, sb, dt)
            batch, cfg = cfg_for(sa, sb, dt)
            row = {"label": label, "a": list(sa), "b": list(sb), "dtype": str(dt),
                   "applied": cfg is not None}
            if cfg is None:
                print(f"  declined  {label}", flush=True)
            else:
                mt = -(-sa[-2] // 32)
                blocks = batch * mt // cfg.per_core_M
                assert cfg.per_core_M == mt or blocks <= gx * gy, (
                    f"{sa}: per_core_M={cfg.per_core_M} splits Mt={mt} over {blocks} blocks "
                    f"on {gx*gy} cores -- the kernels stride a whole batch per block")
                kw = {"compute_kernel_config": ckc}
                if odt is not None:
                    kw["dtype"] = odt
                ref = ttnn.to_torch(ttnn.matmul(a, b, **kw))
                got = ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg, **kw))
                exact = torch.equal(ref, got)
                row.update(per_core_M=cfg.per_core_M, per_core_N=cfg.per_core_N,
                           in0_block_w=cfg.in0_block_w, blocks=blocks, bit_exact=exact)
                print(f"  applied   {label}  per_core_M={cfg.per_core_M} "
                      f"per_core_N={cfg.per_core_N} in0_block_w={cfg.in0_block_w} "
                      f"blocks={blocks}  {'bit-exact' if exact else 'NOT EXACT'}", flush=True)
                assert exact, f"{sa}x{sb}: program config is not bit-exact"
            out.append(row)
            ttnn.deallocate(a); ttnn.deallocate(b)

        # ---- rank 5, last: a rejected program config is TT_FATAL, which aborts the process ----
        label, sa, sb, dt, odt = RANK5
        batch, cfg = cfg_for(sa, sb, dt)
        print(f"\nrank-5 probe: {label} batch={batch} cfg={'built' if cfg else 'declined'}",
              flush=True)
        json.dump({"rank4": out, "rank5_cfg_built": cfg is not None},
                  open(sys.argv[1], "w"), indent=1)
        if cfg is not None:
            a, b = build(dev, sa, dt), build(dev, sb, dt)
            ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
            print("  rank-5 plain matmul ok, now the program config (may TT_FATAL)...", flush=True)
            got = ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg, compute_kernel_config=ckc))
            exact = torch.equal(ref, got)
            print(f"  rank-5 ACCEPTED  per_core_M={cfg.per_core_M} "
                  f"per_core_N={cfg.per_core_N}  {'bit-exact' if exact else 'NOT EXACT'}",
                  flush=True)
            json.dump({"rank4": out, "rank5_cfg_built": True, "rank5_accepted": True,
                       "rank5_bit_exact": exact}, open(sys.argv[1], "w"), indent=1)
        print("\nPASS", flush=True)
    finally:
        ttnn.close_device(dev)


main()
