"""What bounds `k1_linear` on Wormhole: the per-K-tile block sync, or the per-K-tile fp32 pack.

cen25 puts ~13 s of the WH c730 fold in the OpenFold3 DiT's k1 linears at 0.36-0.41 of the math roof, and HiFi2 runs
them no faster than HiFi4. `k1_linear` takes one K tile per block (in0_block_w 1, the WH fp32-dest erratum fix), so
every K tile costs one mcast round and one fp32 pack with L1 accumulation per output tile. Variants at the DiT shapes
(bf16 operands, fp32 accumulation, [5, 768, 768] x [768, N]):
  k1           the shipped program
  k1_f32out    same, fp32 output (pack bytes of the last pass double)
  k1_noacc     packer_l1_acc off (partials reloaded into dest per block)
  k1_in0l1     activation in L1 interleaved
  reuse        MatmulMultiCoreReuseProgramConfig, in0_block_w 1 (no mcast: each core reads its own operands)
  kb2, kb4     in0_block_w 2 / 4: fewer syncs AND fewer packs, but erratum-exposed (timing only; wrong pixels counted)
Each result is checked against a float64 reference: max |err| and the count of |err| > 0.25 (the erratum's +-2 / +-4).

usage: TT_VISIBLE_DEVICES=N timeout 900 python perf/spd/k1_probe.py
"""
import glob
import statistics
import threading
import time

import torch
import ttnn

import tt_bio.tenstorrent as T

dev = T.get_device()
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
S, M, K = 5, 768, 768
torch.manual_seed(0)


def ckc(fid=ttnn.MathFidelity.HiFi4, l1acc=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                            fp32_dest_acc_en=True, packer_l1_acc=l1acc)


def timed(f, n=40):
    ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(f())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


def k1_pc(x, w, dtype, kb=1):
    shp = tuple(x.padded_shape)
    g = dev.compute_with_storage_grid_size()
    tiles = (T._TILE_BYTES[x.dtype], T._TILE_BYTES[w.dtype],
             T._TILE_BYTES[dtype] + (0 if dtype == ttnn.float32 else 4096), 0)
    pc = T._k1_program(S * M // 32, K // 32, int(w.padded_shape[-1]) // 32, g.x, g.y, None, tiles)
    if kb == 1:
        return pc
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), in0_block_w=kb, out_subblock_h=pc.out_subblock_h,
        out_subblock_w=pc.out_subblock_w, out_block_h=pc.out_block_h, out_block_w=pc.out_block_w,
        per_core_M=pc.per_core_M, per_core_N=pc.per_core_N, transpose_mcast=False, fused_activation=None,
        fuse_batch=True)


def reuse_pc(x, w):
    g = dev.compute_with_storage_grid_size()
    mt, nt = S * M // 32, int(w.padded_shape[-1]) // 32
    ncores = g.x * g.y
    # per-core blocks of (pm x pn) output tiles covering mt x nt on <= ncores cores
    best = None
    for pm in (d for d in range(1, mt + 1) if mt % d == 0):
        for pn in (d for d in range(1, nt + 1) if nt % d == 0):
            if (mt // pm) * (nt // pn) <= ncores:
                c = (mt // pm) * (nt // pn)
                if best is None or c > best[0] or (c == best[0] and pm * pn < best[1] * best[2]):
                    best = (c, pm, pn)
    _, pm, pn = best
    h, w_ = max(((h, w_) for h in range(1, min(pm, 4) + 1) for w_ in range(1, min(pn, 4) + 1)
                 if pm % h == 0 and pn % w_ == 0 and h * w_ <= 4), key=lambda s: (s[0] * s[1], s[1]))
    return ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(g.x, g.y), in0_block_w=1,
                                                  out_subblock_h=h, out_subblock_w=w_, per_core_M=pm,
                                                  per_core_N=pn)


CLK = []


def sample_clock():
    while True:
        v = [int(open(f).read().split()[0]) for f in glob.glob("/sys/class/tenstorrent/*/tt_aiclk")]
        CLK.extend(x for x in v if 0 < x < 5000)
        time.sleep(0.5)


threading.Thread(target=sample_clock, daemon=True).start()
for N in (2304, 768):
    A = torch.randn(S, M, K)
    W = torch.randn(K, N) / K ** 0.5
    Ab, Wb = A.bfloat16(), W.bfloat16()
    ref = (Ab.double() @ Wb.double())
    a = ttnn.from_torch(Ab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    al1 = ttnn.from_torch(Ab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    w = ttnn.from_torch(Wb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    bf, f32 = ttnn.bfloat16, ttnn.float32
    arms = {
        "k1": lambda: ttnn.linear(a, w, dtype=bf, compute_kernel_config=ckc(), program_config=k1_pc(a, w, bf)),
        "k1_f32out": lambda: ttnn.linear(a, w, dtype=f32, compute_kernel_config=ckc(), program_config=k1_pc(a, w, f32)),
        "k1_hifi2": lambda: ttnn.linear(a, w, dtype=bf, compute_kernel_config=ckc(ttnn.MathFidelity.HiFi2),
                                        program_config=k1_pc(a, w, bf)),
        "k1_noacc": lambda: ttnn.linear(a, w, dtype=bf, compute_kernel_config=ckc(l1acc=False),
                                        program_config=k1_pc(a, w, bf)),
        "k1_in0l1": lambda: ttnn.linear(al1, w, dtype=bf, compute_kernel_config=ckc(), program_config=k1_pc(al1, w, bf)),
        "reuse": lambda: ttnn.linear(a, w, dtype=bf, compute_kernel_config=ckc(), program_config=reuse_pc(a, w)),
        "kb2": lambda: ttnn.linear(a, w, dtype=f32, compute_kernel_config=ckc(), program_config=k1_pc(a, w, f32, 2)),
        "kb4": lambda: ttnn.linear(a, w, dtype=f32, compute_kernel_config=ckc(), program_config=k1_pc(a, w, f32, 4)),
    }
    for name, f in arms.items():
        try:
            us = timed(f)
            o = f()
            err = (ttnn.to_torch(o).double().reshape(ref.shape) - ref).abs()
            ttnn.deallocate(o)
            print(f"N={N:5d} {name:10s} {us:8.1f} us  max|err| {err.max().item():.4f}  >0.25: {(err > 0.25).sum().item()}",
                  flush=True)
        except Exception as e:  # a config the factory rejects is a result too
            print(f"N={N:5d} {name:10s} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
    for t in (a, al1, w):
        ttnn.deallocate(t)
print(f"aiclk all nodes during the probe: median {statistics.median(CLK) if CLK else 'n/a'} min {min(CLK, default='n/a')} n {len(CLK)}",
      flush=True)
