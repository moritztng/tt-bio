#!/usr/bin/env python3
"""Does assembling a pair tensor on the host instead of on device change its tile padding?

`ttnn.concat` of tile blocks carries whatever each block holds in its padded columns; a host
`torch.cat` + `from_torch` re-tilizes and pads with zeros. Both agree on every logical element,
so `to_torch` cannot tell them apart -- but the trimul's triangle product contracts the token
axis, which is exactly the padded one, so a non-zero pad reaches the result.

The first version of this probe built its blocks with `from_torch`, which zero-pads, so both
assemblies started from clean padding and it could not have detected the thing it was testing.
It reported "not the difference" and that reading was worthless. Blocks are now built the way
`StructuralTokenExpander` builds them -- a ROW_MAJOR gather, reshaped to a row count that is not
a tile multiple, then tilized -- which is where dirty padding would come from if it does.

Needs one free card, no model and no fold.

    TT_VISIBLE_DEVICES=26 python3 scripts/probe_concat_pad_content.py [N] [C] [BLOCK]
"""
import sys

import torch
import ttnn

from tt_bio.tenstorrent import get_device


def _expander_style_block(dev, rows, N, C, table, seed):
    """A block built as the expander builds z_tile: gather in ROW_MAJOR, reshape to
    (rows, N, C) where N is not a tile multiple, then to_layout(TILE)."""
    g = torch.randint(0, table.shape[0], (1, rows * N), generator=seed, dtype=torch.int32)
    idx = ttnn.from_torch(g, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
    flat = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(ttnn.reshape(flat, (rows, N, C)), ttnn.TILE_LAYOUT)


def main(N=200, C=64, block=32):
    dev = get_device()
    seed = torch.Generator().manual_seed(0)
    npad = (N + 31) // 32 * 32
    tbl = ttnn.from_torch(torch.randn(4096, C, generator=seed, dtype=torch.float32).bfloat16(),
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    blocks = [_expander_style_block(dev, min(block, N - s), N, C, tbl, seed)
              for s in range(0, N, block)]
    print(f"N={N} C={C} block={block}  N_pad={npad}  pad_cols={npad - N}  "
          f"block dtype={blocks[0].dtype}")

    host = [ttnn.to_torch(b) for b in blocks]
    dev_cat = ttnn.concat(blocks, dim=-3)
    host_cat = ttnn.from_torch(torch.cat(host, dim=-3), layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)

    a, b = ttnn.to_torch(dev_cat), ttnn.to_torch(host_cat)
    print(f"logical data equal: {torch.equal(a, b)}")

    # Contract the padded axis, the way the triangle product does: sum over the middle index.
    # A zero pad and a dirty pad give different sums iff the pad is non-zero.
    def contract(z):
        return ttnn.to_torch(ttnn.matmul(ttnn.permute(z, (2, 0, 1)),
                                         ttnn.permute(z, (2, 1, 0))))

    ca, cb = contract(dev_cat), contract(host_cat)
    same = torch.equal(ca, cb)
    print(f"after contracting the padded axis equal: {same}")
    if not same:
        d = (ca.float() - cb.float()).abs()
        print(f"  max |diff| {d.max().item():.6g}  mismatching {(d > 0).sum().item()} of {d.numel()}")
        print("  => the device concat carries NON-ZERO padding and it reaches the result;")
        print("     the host-assembled tensor is the clean one.")
    else:
        print("  => padding content is not the difference; look elsewhere.")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
