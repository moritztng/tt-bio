#!/usr/bin/env python3
"""Hand-built dot products that test the dest-add hypothesis for the Wormhole -2^k misses.

Hypothesis: an fp32 add in dest of two opposite-signed values whose upper halves (sign, exponent,
top 7 significand bits) are equal and which differ only in the low 16 bits comes back off by -2^k
when the addend is the larger one. Each case places a few nonzero terms in one row of in0 (a) and
in1 (b) = 1.0, every other term zero, on one 32 x K tile, 1x1 grid. The dot is summed by MVMULs of
16 inner terms each, so terms 0-15, 16-31, 32-47 and 48-63 reach dest as separate adds.

  cross_*   the cancelling group sits in the next K tile (k 32): dest carries the partial across
            tiles at in0_block_w 2, the packer adds the tiles in L1 at in0_block_w 1
  within_*  the cancelling group is the same tile's second MVMUL (k 16)
  *_small   the addend's magnitude is below dest's (hypothesis: correct)
  *_big     the addend's magnitude is above dest's (hypothesis: -2^k)
Terms inside one 16-term group are summed by the multiplier tree, which floors small terms against
the largest, so the deep bits reach dest through a group of their own.

Result (whglx, 2026-09-24): no case misses at any fidelity or K block, so a tie in the top bits is not
sufficient. At in0_block_w 1 the packer's L1 sum dropped the 2^-9.4 term: the tile partials it adds
carry about 10 significand bits.
"""
import json

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
K = 64
BIG = 7.5625                     # 111.1001b, 7 significant bits
D = 1.5 * 2.0 ** -10             # alone in its own MVMUL group, so dest holds BIG + D exactly
P7, P8 = 2.0 ** -7, 2.0 ** -8
cases = {
    # dest = BIG + D after tile 0; tile 1 adds -BIG (smaller) or -(BIG + 2^-8) (larger, same top bits)
    "cross_small": {0: BIG, 16: D, 32: -BIG},
    "cross_big": {0: BIG, 16: D, 32: -BIG, 33: -P8},
    # one tile: group 0 = BIG + 2^-7 or + 2^-8, group 1 = its near-negative; the tie is in the top bits
    "within_small": {0: BIG, 1: P7, 16: -BIG, 17: -P8},
    "within_big": {0: BIG, 1: P8, 16: -BIG, 17: -P7},
}
rows = []
for name, terms in cases.items():
    a = torch.zeros(1, 1, 32, K, dtype=torch.bfloat16)
    b = torch.zeros(1, 1, 32, K, dtype=torch.bfloat16)
    for k, v in terms.items():
        a[0, 0, 0, k] = v
        b[0, 0, 0, k] = 1.0
    ref = float(a[0, 0, 0].double() @ b[0, 0, 0].double())
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
    r = {"case": name, "terms": {k: float(torch.tensor(v).bfloat16()) for k, v in terms.items()}, "ref": ref}
    for fid in ("LoFi", "HiFi2", "HiFi4"):
        for w in (1, 2):
            ckc = kcls(math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=False,
                       fp32_dest_acc_en=True, packer_l1_acc=True)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(1, 1), in0_block_w=w, out_subblock_h=1, out_subblock_w=1,
                out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False,
                fused_activation=None, fuse_batch=False)
            o = ttnn.matmul(ta, tb, transpose_b=True, compute_kernel_config=ckc, program_config=pc,
                            dtype=ttnn.float32)
            r[f"{fid}_w{w}"] = round(float(ttnn.to_torch(o)[0, 0, 0, 0]) - ref, 6)
            ttnn.deallocate(o)
    rows.append(r)
    print(json.dumps(r), flush=True)
ttnn.close_device(dev)
