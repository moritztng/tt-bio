"""E6: what the 2D mcast matmul factory actually allocates in L1, and what the gate must subtract.

Reads the v0.68.0 factory's CB sizing (matmul_multicore_reuse_mcast_2d_program_factory.cpp,
create_program_mcast_in0_in1) as a formula, then checks it against the addresses the throw reports.
Nothing here is inherited: the CB base and every total is read back off this card.
"""
import re
import sys

import torch
import ttnn

BF16 = 2
FP32 = 4

ADDR = re.compile(r"L1 buffer allocated at (\d+).*?circular buffer region ends at (\d+)", re.S)


def cb_need(obh, obw, bw, kt, elem=BF16, fp32_dest=True, packer_l1_acc=True, bias=False, B=1):
    """Per-core static CB bytes for create_program_mcast_in0_in1, DRAM-interleaved operands."""
    num_blocks = kt // bw
    packer_en = packer_l1_acc and ((bias and num_blocks > 1) or num_blocks > 2)
    interm_tile = 1024 * (FP32 if fp32_dest else (BF16 if packer_en else elem))
    dbl = 2 if B * num_blocks > 1 else 1
    in0 = dbl * obh * bw * 1024 * elem
    in1 = dbl * obw * bw * 1024 * elem
    out = obh * obw * 1024 * elem
    interm = obh * obw * interm_tile
    share = interm_tile == 1024 * elem  # same format -> one shared CB
    return in0 + in1 + out + (0 if share else interm)


def subblock(obh, obw, fp32_dest=True):
    cap = 4 if fp32_dest else 8
    for h in range(min(obh, cap), 0, -1):
        if obh % h:
            continue
        for w in range(min(obw, cap // h), 0, -1):
            if obw % w == 0:
                return h, w
    return 1, 1


def cfg2d(grid, mt, kt, nt, bw, obh=None, obw=None, fuse_batch=True):
    gx, gy = grid
    per_core_M = -(-mt // gy)
    per_core_N = -(-nt // gx)
    obh = obh or per_core_M
    obw = obw or per_core_N
    sh, sw = subblock(obh, obw)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=obh,
        out_block_w=obw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=fuse_batch,
    ), per_core_M, per_core_N, obh, obw


def run(dev, x, w, pc, ckc, out_l1):
    try:
        y = ttnn.linear(
            x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, program_config=pc,
            memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(dev)
        return None, y
    except Exception as e:
        m = ADDR.search(str(e))
        return (int(m.group(1)), int(m.group(2))) if m else (None, str(e).strip()[:200]), None


def free_per_bank(dev):
    return ttnn.get_memory_view(dev, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank


def main():
    dev = ttnn.open_device(device_id=0)
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    gx, gy = dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y
    banks = ttnn.get_memory_view(dev, ttnn.BufferType.L1).num_banks
    print(f"grid {gx}x{gy}  banks {banks}  free/bank idle {free_per_bank(dev)}")

    def mk(shape, l1=False):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), device=dev,
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               memory_config=ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG)

    # the real chunked-Transition shapes, read off the fold by W-inblockw's census:
    # protenix-v2 pair chunk [1, 30, 320, 256] -> mt 300; fc1/fc2 kt 8 nt 32, fc3 kt 32 nt 8
    x4 = mk((1, 30, 320, 256))
    x3 = mk((1, 9600, 256))
    w = mk((256, 1024))

    print("\n== A. rank-3 vs rank-4 at identical folded M, deliberately oversized so both throw ==")
    print("   grid 11x5 -> per_core_M 60; the throw reports the total the factory wanted")
    for fb in (True, False):
        for label, x, Bfold in (("rank-4 [1,30,320,256]", x4, 30), ("rank-3 [1,9600,256]", x3, 1)):
            pc, pm, pn, obh, obw = cfg2d((gx, 5), 300, 8, 32, 8, fuse_batch=fb)
            B = 1 if fb else Bfold
            pred = cb_need(obh, obw, 8, 8, B=B)
            got, y = run(dev, x, w, pc, ckc, out_l1=False)
            if y is not None:
                ttnn.deallocate(y)
            end = got[1] if got and isinstance(got[1], int) else None
            base = end - pred if end else None
            print(f"   fuse_batch={fb!s:<5} {label:<22} obh={obh} obw={obw} B={B:<2} "
                  f"need={pred:>8}  region_end={end}  implied_CB_base={base}  {'' if end else got}")

    print("\n== B. the model against the reported total, across configs (rank-4, fuse_batch=True) ==")
    for gyy, bw, obh in ((5, 8, 60), (5, 4, 60), (5, 2, 60), (5, 8, 30), (4, 8, 75)):
        pc, pm, pn, _, obw = cfg2d((gx, gyy), 300, 8, 32, bw, obh=obh)
        pred = cb_need(obh, obw, bw, 8)
        got, y = run(dev, x4, w, pc, ckc, out_l1=False)
        if y is not None:
            ttnn.deallocate(y)
        end = got[1] if got and isinstance(got[1], int) else None
        print(f"   grid {gx}x{gyy} bw={bw} pcM={pm} obh={obh} obw={obw} need={pred:>8}  "
              f"region_end={end}  implied_base={end - pred if end else None}  {'' if end else got}")

    print("\n== C. the gate's budget: does the op's own L1 output eat it? ==")
    # emulate the swiglu residency at the fc2 call: x_norm [1,30,320,256] + x_1 [1,30,320,1024] live
    res1 = mk((1, 30, 320, 256), l1=True)
    res2 = mk((1, 30, 320, 1024), l1=True)
    fb = free_per_bank(dev)
    out_bytes = -(-(300 * 32) // banks) * 1024 * BF16
    print(f"   free/bank measured before the call {fb}; the op's own L1 output is {out_bytes} B/bank")
    for obh in (30, 15, 10, 6, 5, 3, 2):
        pc, pm, pn, _, obw = cfg2d((gx, gy), 300, 8, 32, 8, obh=obh)
        need = cb_need(obh, obw, 8, 8)
        got, y = run(dev, x4, w, pc, ckc, out_l1=True)
        ok = "OK" if y is not None else "CLASH"
        if y is not None:
            ttnn.deallocate(y)
        print(f"   obh={obh:>2} need={need:>7}  free={fb}  free-out={fb - out_bytes:>7}  "
              f"naive_gate={'fit' if need <= fb else 'no '}  "
              f"out_aware_gate={'fit' if need <= fb - out_bytes else 'no '}  -> {ok}")
    ttnn.deallocate(res1)
    ttnn.deallocate(res2)
    ttnn.close_device(dev)


if __name__ == "__main__":
    sys.exit(main())
