#!/usr/bin/env python3
"""Reproduce the opendde fold-parity break at the op level.

The fold bisect (state doc 7.2.1) put the whole break on one class: B=8, Mt=19, Kt=19, Nt=2, bf16.
Its B=16 twin is torch.equal against the default. The only thing that differs between them is the
per_core_M the chooser lands on (1 at B=8, 19 at B=16), and per_core_M does not reorder the K
accumulation -- so either that reasoning is wrong or the break is somewhere else. This sweeps both
per_core_M values at both batch sizes, at the production compute-kernel config, against the default.
"""
import json, statistics, sys, time
import torch, ttnn
from tt_bio import tenstorrent as T

REPS, ITERS = 20, 5


def timeit(fn, dev):
    fn(); ttnn.synchronize_device(dev)
    out = []
    for _ in range(ITERS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / REPS * 1e3)
    return statistics.median(out)


def main(out_path):
    dev = ttnn.open_device(device_id=0)
    T._configure_active_compute_grid(dev)
    gx, gy = T.COMPUTE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    for B in (16, 8, 4, 2):
        sa, sb = (1, B, 608, 608), (1, B, 608, 64)
        torch.manual_seed(0)
        ta, tb = torch.randn(sa), torch.randn(sb)
        a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
        base = timeit(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc)), dev)
        chosen = T._batched_matmul_config(B, 19, 19, 2, 2)
        print(f"\nB={B}  default {base:.4f} ms   chooser picks "
              f"{'None' if chosen is None else f'per_core_M={chosen.per_core_M} bw={chosen.in0_block_w}'}",
              flush=True)
        for pcm in (1, 19):
            for bw in (1, 19):
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                    out_subblock_h=1, out_subblock_w=2, per_core_M=pcm, per_core_N=2)
                try:
                    got = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc,
                                                    program_config=pc))
                    t = timeit(lambda: ttnn.deallocate(ttnn.matmul(
                        a, b, compute_kernel_config=ckc, program_config=pc)), dev)
                except Exception as e:
                    print(f"   pcm={pcm:3d} bw={bw:3d} REJECT {type(e).__name__}: {str(e)[:80]}",
                          flush=True)
                    continue
                d = (ref.float() - got.float())
                exact = torch.equal(ref, got)
                nbad = int((d != 0).sum())
                print(f"   pcm={pcm:3d} bw={bw:3d} blocks={B * 19 // pcm:4d}  {t:8.4f} ms  "
                      f"{base / t:5.2f}x  equal={exact}  differing elems={nbad}/{d.numel()}  "
                      f"max|d|={d.abs().max().item():.3e}", flush=True)
                rows.append(dict(B=B, per_core_M=pcm, in0_block_w=bw, base_ms=base, cfg_ms=t,
                                 ratio=base / t, torch_equal=exact, n_differing=nbad,
                                 n_elems=int(d.numel()), max_abs_diff=d.abs().max().item(),
                                 chooser_picks=None if chosen is None else chosen.per_core_M))
        ttnn.deallocate(a); ttnn.deallocate(b)
    ttnn.close_device(dev)
    json.dump({"rows": rows}, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


main(sys.argv[1])
