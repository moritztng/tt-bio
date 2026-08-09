#!/usr/bin/env python3
"""Sweep MatmulMultiCoreReuseProgramConfig over (per_core_M, in0_block_w) for the six batched
diffusion matmul classes, on the real 298 aa shapes. ttnn's automatic path never selects this
factory for DRAM-interleaved batched operands, so B stays a sequential loop inside one core;
this factory splits B*Mt/per_core_M output blocks over the grid.

Constraints read out of v0.67.4 matmul_device_operation.cpp:850-1020:
  N == per_core_N (no N split), M % per_core_M == 0, Kt % in0_block_w == 0,
  per_core_M % out_subblock_h == 0, per_core_N % out_subblock_w == 0,
  out_subblock_h * out_subblock_w <= dest_reg_count.

Host-timed, synchronize_device on both sides, warm, median of ITERS.
"""
import json, statistics, sys, time
import torch, ttnn

F32 = ttnn.float32
BF16 = ttnn.bfloat16

# label, a_shape, b_shape, dtype, model, site, qb2_ms_per_fold
CASES = [
    ("pv2 :417 atomQK", (75, 4, 32, 128), (75, 4, 128, 32), F32, "protenix-v2", "protenix.py:417", 633.0),
    ("pv2 :414 atomAV", (75, 4, 32, 32), (75, 4, 32, 128), F32, "protenix-v2", "protenix.py:414", 387.8),
    ("pv2 :1656 DiT AV", (1, 16, 320, 320), (1, 16, 320, 64), F32, "protenix-v2", "tenstorrent.py:1656", 627.4),
    ("pv2 :1650 DiT QK", (1, 16, 320, 64), (1, 16, 64, 320), F32, "protenix-v2", "tenstorrent.py:1650", 223.9),
    ("odd :417 atomQK", (75, 4, 32, 128), (75, 4, 128, 32), BF16, "opendde", "protenix.py:417", 592.6),
    ("odd :414 atomAV", (75, 4, 32, 32), (75, 4, 32, 128), BF16, "opendde", "protenix.py:414", 357.4),
    ("odd :1678 DiT AV", (1, 16, 608, 608), (1, 16, 608, 64), BF16, "opendde", "tenstorrent.py:1678", 1325.7),
    ("odd :1670 DiT QK", (1, 16, 608, 64), (1, 16, 64, 608), BF16, "opendde", "tenstorrent.py:1670", 369.6),
]
REPS = 20
ITERS = 5
# tt-metal SUBBLOCK_HW_CHOICES order: largest area first.
SUBBLOCK_HW = [(4, 2), (2, 4), (8, 1), (1, 8), (4, 1), (1, 4), (2, 2), (2, 1), (1, 2), (1, 1)]


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def subblock(pcm, pcn, regs):
    for h, w in SUBBLOCK_HW:
        if h * w <= regs and pcm % h == 0 and pcn % w == 0:
            return h, w
    return 1, 1


def l1_bytes(pcm, pcn, bw, in_tile, out_tile, interm_tile):
    """CB footprint per core, matmul_multicore_reuse_optimized_program_factory.cpp:286-306."""
    return (2 * pcm * bw * in_tile + 2 * pcn * bw * in_tile
            + pcm * pcn * out_tile + pcm * pcn * interm_tile)


def timeit(fn, dev):
    fn()
    ttnn.synchronize_device(dev)
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
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    print(f"grid {g.x}x{g.y} = {cores} cores, worker L1 unreserved {l1 / 1024:.0f} KB", flush=True)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    regs = 4  # fp32_dest_acc_en halves the dest register file
    results = []
    for label, sa, sb, dt, model, site, qb2_ms in CASES:
        torch.manual_seed(0)
        ta, tb = torch.randn(sa), torch.randn(sb)
        a = ttnn.from_torch(ta, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(tb, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B = 1
        for d in sa[:-2]:
            B *= d
        mt, kt, nt = sa[-2] // 32, sa[-1] // 32, sb[-1] // 32
        base = timeit(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc)), dev)
        ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
        gold = (ta.float() @ tb.float())
        base_rms = (gold - ref.float()).pow(2).mean().sqrt().item()
        print(f"\n{label}  B={B} Mt={mt} Kt={kt} Nt={nt}  default {base:.4f} ms  "
              f"rmsd-vs-torch {base_rms:.3e}", flush=True)
        in_tile = 4096 if dt == F32 else 2048
        best = None
        for pcm in divisors(mt):
            for bw in divisors(kt):
                blocks = B * mt // pcm
                need = l1_bytes(pcm, nt, bw, in_tile, in_tile, 4096)
                if need > l1:
                    continue
                h, w = subblock(pcm, nt, regs)
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(g.x, g.y), in0_block_w=bw,
                    out_subblock_h=h, out_subblock_w=w, per_core_M=pcm, per_core_N=nt)
                try:
                    got = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc))
                    t = timeit(lambda: ttnn.deallocate(
                        ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc)), dev)
                except Exception as e:
                    print(f"   pcm={pcm:3d} bw={bw:3d} REJECT {type(e).__name__}: {str(e)[:90]}", flush=True)
                    continue
                exact = torch.equal(ref, got)
                rms = (gold - got.float()).pow(2).mean().sqrt().item()
                print(f"   pcm={pcm:3d} bw={bw:3d} blocks={blocks:4d}/{cores} l1={need // 1024:4d}KB "
                      f"sub={h}x{w}  {t:8.4f} ms  {base / t:6.2f}x  equal={exact} "
                      f"rmsd-vs-torch={rms:.3e}", flush=True)
                row = dict(label=label, model=model, site=site, B=B, Mt=mt, Kt=kt, Nt=nt,
                           dtype=str(dt), per_core_M=pcm, in0_block_w=bw, blocks=blocks,
                           cores=cores, l1_KB=need // 1024, sub_h=h, sub_w=w,
                           default_ms=base, cfg_ms=t, ratio=base / t, torch_equal=exact,
                           rmsd_vs_torch_fp32=rms, base_rmsd_vs_torch_fp32=base_rms,
                           qb2_ms_per_fold=qb2_ms)
                results.append(row)
                if best is None or t < best["cfg_ms"]:
                    best = row
        if best:
            print(f"   BEST pcm={best['per_core_M']} bw={best['in0_block_w']} "
                  f"{best['ratio']:.2f}x equal={best['torch_equal']}", flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    ttnn.close_device(dev)
    json.dump({"grid_cores": cores, "l1_unreserved": l1, "rows": results},
              open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


main(sys.argv[1])
