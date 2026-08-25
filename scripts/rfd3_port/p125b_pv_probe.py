#!/usr/bin/env python3
"""p125b -- when L5b is not bit-exact, say WHICH product the kernel computed.

`torch.equal` failing tells you nothing about why. This reconstructs candidate products on the
host from the same attention and value tensors the kernel saw, and reports which one the device
output correlates with. A contiguous key range winning means the accumulation dropped blocks; a
shifted value index winning means the in1 tile index is off; nothing winning means the MAC itself
is wrong rather than its addressing.

Runs at the T0 rung only (320 rows, 3520 key width) -- two seconds a pass, which is the point.
"""
import os
import sys

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic as SG                                 # noqa: E402
from tt_bio.tenstorrent import attn_value_matmul, get_device             # noqa: E402

TILE, HEADS, HEAD_DIM = 32, 4, 32
ROWS, KEY_W = 320, 3520


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    SG.set_pv_enabled(True)
    device = get_device()

    g = torch.Generator().manual_seed(7)
    x_t = torch.randn(1, HEADS, ROWS, KEY_W, generator=g, dtype=torch.float32) * 4.0
    v_t = torch.randn(1, HEADS, KEY_W, HEAD_DIM, generator=g, dtype=torch.bfloat16)
    scores = ttnn.from_torch(x_t, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    vv = ttnn.from_torch(v_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    v = SG.pv_classify(scores, vv, ttnn.bfloat16, ckc)
    print("classify:", v, flush=True)
    Wt, blk = v["Wt"], v["blk"]
    # `plan`'s cb_length is what splits a row into device passes, and a pass boundary is the most
    # likely place for a K index to drift.
    p = SG.plan(scores, scores, (13, 10), True, True)
    cb_len = p["cb_length"]
    print("Wt=%d blk=%d cb_length=%d passes=%d" % (Wt, blk, cb_len, -(-Wt // cb_len)), flush=True)

    attn = SG.softmax_bf16(scores, ttnn.bfloat16)
    ref = attn_value_matmul(attn, vv, ckc, ttnn.bfloat16)
    A = ttnn.to_torch(attn).float()
    R = ttnn.to_torch(ref).float()
    ttnn.deallocate(ref)
    ttnn.deallocate(attn)
    V = v_t.float()

    fused = SG.softmax_pv_fused(scores, vv, ttnn.bfloat16, ckc)
    assert fused is not None, "classify said yes but the lever declined"
    F = ttnn.to_torch(fused).float()
    ttnn.deallocate(fused)

    H = A @ V
    print("\nhost full product vs device shipped: pcc %.6f   (sanity: must be ~1)" % pcc(H, R))
    print("device fused vs device shipped:      pcc %.6f" % pcc(F, R))

    # Hypothesis 1: the accumulation covered only a contiguous run of key tiles.
    cands = []
    for lo in (0, cb_len, Wt - blk, Wt - 1, Wt - cb_len):
        for hi in (blk, cb_len, Wt, lo + blk, lo + cb_len):
            if 0 <= lo < hi <= Wt:
                cands.append((lo, hi))
    seen, rows = set(), []
    for lo, hi in cands:
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        part = A[..., lo * TILE:hi * TILE] @ V[:, :, lo * TILE:hi * TILE, :]
        rows.append((pcc(F, part), "keys[%d:%d] tiles" % (lo, hi)))

    # Hypothesis 2: every block was accumulated but against a shifted value tile.
    for shift in (-2, -1, 1, 2, blk, -blk, cb_len, -cb_len):
        Vs = torch.roll(V, shifts=shift * TILE, dims=2)
        rows.append((pcc(F, A @ Vs), "all keys, value tiles rolled by %+d" % shift))

    # Hypothesis 3: only the last block of each device pass landed.
    n_pass = -(-Wt // cb_len)
    acc = torch.zeros_like(H)
    for i in range(n_pass):
        lo = min(i * cb_len, Wt)
        hi = min(lo + cb_len, Wt)
        s = hi - blk
        acc = acc + A[..., s * TILE:hi * TILE] @ V[:, :, s * TILE:hi * TILE, :]
    rows.append((pcc(F, acc), "last block of each device pass only"))

    # Hypothesis 4: the value tiles were re-read per head from head 0 (a head-index bug).
    V0 = V[:, :1].expand_as(V).contiguous()
    rows.append((pcc(F, A @ V0), "every head used head 0's value tiles"))

    print("\nbest-matching products for the device fused output:")
    for c, name in sorted(rows, reverse=True)[:8]:
        print("  pcc %+.6f   %s" % (c, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
