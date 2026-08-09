#!/usr/bin/env python3
"""Synthetic reproduction of the padded-operand corruption, so the guard is a rule and not a
correlation over five fold measurements.

In the fold, `MatmulMultiCoreReuseProgramConfig` returns wrong results for the DiT attention at
per_core_M=1 and right ones at per_core_M>=2, but only when the operands are tile-padded (logical
580 in a 608 padding, logical 298 in 320). A tensor built by from_torch has zero padding and cannot
reproduce it, so this builds a full-size tensor and shrinks its logical shape, which leaves the old
values sitting in the pad.
"""
import json, sys
import torch, ttnn
from tt_bio import tenstorrent as T


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def main(out_path):
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    gx, gy = T.COMPUTE_GRID_MAIN
    cores = gx * gy
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    # B, padded M/K, logical M/K, N, label
    CASES = [(16, 608, 580, 64, "opendde DiT AV"), (16, 320, 298, 32, "the 298-token DiT AV"),
             (8, 608, 580, 64, "opendde DiT AV tail chunk")]
    for B, P, L, N, label in CASES:
        torch.manual_seed(0)
        a = ttnn.from_torch(torch.randn(1, B, P, P), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn(1, B, P, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        try:
            a = ttnn.reshape(a, (1, B, L, L), (1, B, P, P))
            b = ttnn.reshape(b, (1, B, L, N), (1, B, P, N))
        except Exception as e:
            print(f"{label}: cannot build a non-zero pad ({type(e).__name__}: {str(e)[:80]})")
            continue
        Mt, Nt = -(-L // 32), -(-N // 32)
        ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
        print(f"\n{label}: B={B} logical {L}x{L} @ {L}x{N} in a {P} padding, Mt={Mt} Nt={Nt}",
              flush=True)
        for pcm in divisors(Mt):
            blocks = B * Mt // pcm
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=1,
                out_subblock_h=1, out_subblock_w=Nt, per_core_M=pcm, per_core_N=Nt)
            try:
                got = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc))
            except Exception as e:
                print(f"   pcm={pcm:3d} REJECT {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            d = (ref.float() - got.float())
            nz = int((d != 0).sum())
            print(f"   pcm={pcm:3d} blocks={blocks:4d} ({'>' if blocks > cores else '<='} "
                  f"{cores} cores)  equal={nz == 0}  differing={nz}/{d.numel()}  "
                  f"max|d|={d.abs().max().item():.3e}", flush=True)
            rows.append(dict(label=label, B=B, padded=P, logical=L, N=N, per_core_M=pcm,
                             blocks=blocks, cores=cores, torch_equal=nz == 0, n_differing=nz,
                             n_elems=int(d.numel()), max_abs_diff=d.abs().max().item()))
        ttnn.deallocate(a); ttnn.deallocate(b)
    ttnn.close_device(dev)
    json.dump({"rows": rows}, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


main(sys.argv[1])
