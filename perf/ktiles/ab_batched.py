"""A/B the two heaviest diffusion attention matmuls: ttnn default vs an explicit
MatmulMultiCoreReuseProgramConfig, which is the only factory that spreads the BATCH
dimension over cores (num_output_blocks_total = B*Mt/per_core_M * Nt/per_core_N).
Host-timed, synchronize_device on both sides, warm, median of N.
"""
import statistics, time, torch, ttnn

CASES = [
    # label, a, b, dtype, in0_block_w, per_core_M, per_core_N, sub_h, sub_w
    ("atom-attn AV  (Kt=1,B=300)", (75, 4, 32, 32), (75, 4, 32, 128), ttnn.float32, 1, 1, 4, 1, 4),
    ("atom-attn QK^T(Kt=4,B=300)", (75, 4, 32, 128), (75, 4, 128, 32), ttnn.float32, 4, 1, 1, 1, 1),
    ("DiT AV pv2    (Kt=10,B=16)", (1, 16, 320, 320), (1, 16, 320, 64), ttnn.float32, 10, 10, 2, 2, 2),
    ("DiT QK^T pv2  (Kt=2,B=16)", (1, 16, 320, 64), (1, 16, 64, 320), ttnn.float32, 2, 10, 10, 1, 4),
    ("DiT AV odde   (Kt=19,B=16)", (1, 16, 608, 608), (1, 16, 608, 64), ttnn.bfloat16, 19, 19, 2, 1, 2),
]
REPS = 20
ITERS = 5


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


def main():
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid {g.x}x{g.y} = {g.x * g.y} cores")
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    for label, sa, sb, dt, bw, pcm, pcn, sh, sw in CASES:
        ta, tb = torch.randn(sa), torch.randn(sb)
        a = ttnn.from_torch(ta, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(tb, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        base = timeit(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=ckc)), dev)
        ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc))
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y), in0_block_w=bw,
            out_subblock_h=sh, out_subblock_w=sw, per_core_M=pcm, per_core_N=pcn)
        try:
            got = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc))
            exact = torch.equal(ref, got)
            rms = (ref.float() - got.float()).pow(2).mean().sqrt().item()
            new = timeit(lambda: ttnn.deallocate(
                ttnn.matmul(a, b, compute_kernel_config=ckc, program_config=pc)), dev)
            print(f"{label}: default {base:9.4f} ms | reuse {new:9.4f} ms | {base / new:6.2f}x | "
                  f"torch.equal={exact} rmsd={rms:.3e}")
        except Exception as e:
            print(f"{label}: default {base:9.4f} ms | reuse FAILED {type(e).__name__}: {str(e)[:150]}")
        ttnn.deallocate(a); ttnn.deallocate(b)
    ttnn.close_device(dev)


main()
