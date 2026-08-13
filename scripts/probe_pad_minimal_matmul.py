#!/usr/bin/env python3
"""Does `minimal_matmul` read tile padding that `ttnn.matmul` masks?

Substituting a re-tilized copy of z_struct for the expander's own changes every opendde structure,
while a device-to-device `ttnn.clone` changes nothing:

    no touch                              e6227922 (9ncy)
    ttnn.clone, device -> device          e6227922
    to_layout(ROW_MAJOR) -> to_layout(TILE)  b8999b32
    to_torch -> from_torch                b8999b32

So it is the re-tilize, not the host. A re-tilize rewrites the tile padding; a clone copies it. That
makes dirty padding the mechanism after all -- and every padding probe so far compared the two
tensors with `ttnn.matmul`, which masks the padded region. The trimul's input projection does not
use `ttnn.matmul`; it uses `ttnn.experimental.minimal_matmul`.

This compares the two tensors both ways, on blocks built as the expander builds them.

    TT_VISIBLE_DEVICES=26 python3 scripts/probe_pad_minimal_matmul.py [N] [C] [BLOCK]
"""
import sys

import torch
import ttnn

from tt_bio.tenstorrent import get_device


def main(N=200, C=384, block=32):
    dev = get_device()
    seed = torch.Generator().manual_seed(0)
    tbl = ttnn.from_torch(torch.randn(4096, C, generator=seed, dtype=torch.float32).bfloat16(),
                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # Blocks the expander's way: ROW_MAJOR gather, reshape to a non-tile-multiple row count, tilize.
    blocks = []
    for s in range(0, N, block):
        rows = min(block, N - s)
        g = torch.randint(0, 4096, (1, rows * N), generator=seed, dtype=torch.int32)
        idx = ttnn.from_torch(g, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
        flat = ttnn.embedding(idx, tbl, layout=ttnn.ROW_MAJOR_LAYOUT,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        blocks.append(ttnn.to_layout(ttnn.reshape(flat, (rows, N, C)), ttnn.TILE_LAYOUT))

    A = ttnn.concat(blocks, dim=-3)                                   # as the expander leaves it
    B = ttnn.to_layout(ttnn.to_layout(A, ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT)  # re-tilized
    npad = (N + 31) // 32 * 32
    print(f"N={N} C={C} block={block}  N_pad={npad}  pad_cols={npad - N}")
    print(f"logical data equal: {torch.equal(ttnn.to_torch(A), ttnn.to_torch(B))}")

    # A weight of the shape the trimul in-projection uses: (c_z, 4*chunk).
    w = ttnn.from_torch(torch.randn(C, 128, generator=seed, dtype=torch.float32).bfloat16(),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    def _cmp(name, f):
        try:
            ra, rb = ttnn.to_torch(f(A)), ttnn.to_torch(f(B))
            same = torch.equal(ra, rb)
            msg = f"{name}: equal={same}"
            if not same:
                d = (ra.float() - rb.float()).abs()
                msg += (f" max|diff|={d.max().item():.6g} "
                        f"mismatch={(d > 0).sum().item()}/{d.numel()}")
            print(msg)
        except Exception as e:
            print(f"{name}: failed: {e}")

    _cmp("ttnn.matmul over the padded axis",
         lambda z: ttnn.matmul(ttnn.permute(z[:, :, :32], (2, 0, 1)),
                               ttnn.permute(z[:, :, :32], (2, 1, 0))))
    _cmp("ttnn.experimental.minimal_matmul (the trimul in-projection)",
         lambda z: ttnn.experimental.minimal_matmul(z, w))
    _cmp("ttnn.linear (the trimul output projection)", lambda z: ttnn.linear(z, w))
    _cmp("ttnn.layer_norm", lambda z: ttnn.layer_norm(z))


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
