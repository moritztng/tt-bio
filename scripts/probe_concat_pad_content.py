#!/usr/bin/env python3
"""Does assembling a pair tensor on the host instead of on device change its tile padding?

`ttnn.concat` of tile blocks carries whatever each block held in its padded columns; a host
`torch.cat` + `from_torch` re-tilizes and pads with zeros. Both agree on every logical element,
so `to_torch` cannot tell them apart -- but the trimul's triangle product contracts the token
axis, which is exactly the padded one, so a non-zero pad reaches the result.

This builds both assemblies from the same blocks and compares them twice: on the logical data,
and after a contraction over the padded axis. Needs one free card, no model and no fold.

    TT_VISIBLE_DEVICES=26 python3 scripts/probe_concat_pad_content.py [N] [C] [BLOCK]
"""
import sys

import torch
import ttnn

from tt_bio.tenstorrent import get_device


def main(N=200, C=64, block=32):
    dev = get_device()
    torch.manual_seed(0)
    # Blocks as the expander produces them: [rows, N, C] tile tensors whose dim -2 (the token
    # axis, N) is not a tile multiple, so each carries N_pad - N columns of padding.
    blocks = []
    for s in range(0, N, block):
        rows = min(block, N - s)
        t = torch.randn(rows, N, C, dtype=torch.float32).bfloat16()
        blocks.append(ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                      dtype=ttnn.bfloat16))
    host = [ttnn.to_torch(b) for b in blocks]

    dev_cat = ttnn.concat(blocks, dim=-3)
    host_cat = ttnn.from_torch(torch.cat(host, dim=-3), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)

    a, b = ttnn.to_torch(dev_cat), ttnn.to_torch(host_cat)
    print(f"N={N} C={C} block={block}  N_pad={(N + 31) // 32 * 32}  pad_cols={(N + 31) // 32 * 32 - N}")
    print(f"logical data equal: {torch.equal(a, b)}")

    # Contract the padded axis, the way the triangle product does: (N,N,C) x (N,N,C) over the
    # middle index. A zero pad and a dirty pad give different sums iff the pad is non-zero.
    def contract(z):
        x = ttnn.permute(z, (2, 0, 1))                       # (C, N, N)
        return ttnn.to_torch(ttnn.matmul(x, ttnn.permute(z, (2, 1, 0))))

    ca, cb = contract(dev_cat), contract(host_cat)
    same = torch.equal(ca, cb)
    print(f"after contracting the padded axis equal: {same}")
    if not same:
        d = (ca.float() - cb.float()).abs()
        print(f"  max |diff| {d.max().item():.6g}  mismatching elements {(d > 0).sum().item()} "
              f"of {d.numel()}")
        print("  => the device concat's padding is NOT zero, and it reaches the result.")
    else:
        print("  => padding content is not the difference; look elsewhere.")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
