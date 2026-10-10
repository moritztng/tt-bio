"""`k1_linear`'s out_block_h on Wormhole: one output block per core (the shipped choice) vs several.

k1_probe found the k1 linear bound by neither math (HiFi2 = HiFi4) nor the per-K-tile sync (K block 2/4 only -16 %),
while an fp32 output costs +45 % over bf16. `_k1_program` takes the LARGEST out_block_h that fits, i.e. the whole
per-core M as one block, so the writer only starts once every K tile of every output tile is done: the write-back is
serial with the compute. Smaller blocks let block i's write overlap block i+1's compute. Every output tile still sums
its K tiles in the same order, so each variant must be bit-identical to the shipped one; this checks that.
Shapes: the six DiT k1 signatures of the c730 census (cen25), bf16 operands, fp32 accumulation, bf16 out.

usage: TT_VISIBLE_DEVICES=N timeout 900 python perf/spd/k1_oh_probe.py
"""
import glob
import statistics
import threading
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

dev = T.get_device()
CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
SHAPES = [(768, 3072), (1536, 768), (768, 1536), (768, 1024), (1024, 768)]  # (K, N) at [5, 768, K]
torch.manual_seed(0)


def timed(f, n=40):
    ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


CLK = []


def sample_clock():
    while True:
        v = [int(open(f).read().split()[0]) for f in glob.glob("/sys/class/tenstorrent/*/tt_aiclk")]
        CLK.extend(x for x in v if 0 < x < 5000)
        time.sleep(0.5)


threading.Thread(target=sample_clock, daemon=True).start()
g = dev.compute_with_storage_grid_size()
for K, N in SHAPES:
    a = ttnn.from_torch(torch.randn(5, 768, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch((torch.randn(K, N) / K ** 0.5).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev)
    pc = T._k1_program(5 * 768 // 32, K // 32, N // 32, g.x, g.y, None, (2048, 2048, 2048 + 4096, 0))
    base = None
    for oh in [d for d in range(pc.out_subblock_h, pc.per_core_M + 1, pc.out_subblock_h) if pc.per_core_M % d == 0]:
        v = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y), in0_block_w=1, out_subblock_h=pc.out_subblock_h,
            out_subblock_w=pc.out_subblock_w, out_block_h=oh, out_block_w=pc.out_block_w,
            per_core_M=pc.per_core_M, per_core_N=pc.per_core_N, transpose_mcast=False, fused_activation=None,
            fuse_batch=True)
        f = lambda: ttnn.linear(a, w, dtype=ttnn.bfloat16, compute_kernel_config=CKC, program_config=v)
        try:
            us = timed(f)
            o = f()
            r = ttnn.to_torch(o)
            ttnn.deallocate(o)
            if base is None:
                base = r
            tag = "shipped" if oh == pc.out_block_h else ""
            print(f"K={K:5d} N={N:5d} pm={pc.per_core_M} pn={pc.per_core_N} sub={pc.out_subblock_h}x{pc.out_subblock_w} "
                  f"oh={oh:3d} {us:8.1f} us  bitexact_vs_first {torch.equal(r, base)} {tag}", flush=True)
        except Exception as e:
            print(f"K={K:5d} N={N:5d} oh={oh:3d} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(w)
print(f"aiclk all nodes during the probe: median {statistics.median(CLK) if CLK else 'n/a'} min {min(CLK, default='n/a')} n {len(CLK)}",
      flush=True)
