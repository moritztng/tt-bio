#!/usr/bin/env python3
"""Is folding the residual's trailing cast into the add bit-exact?

`AF2PairBlock._residual` at `rne_residual = True` is four ttnn calls:

    wide  = typecast(x, f32); other = typecast(update, f32)
    wide  = add_(wide, other)
    out   = typecast(wide, bf16)          <- 6 of the path's 30 B/element

Under a tape the add is ALREADY out of place (`taped_ttnn.py:706` registers `add_` with
`out_of_place=ttnn.add`), so its output dtype is addressable and the trailing cast can ride on
it. That removes 1 of 4 calls and 20 % of the residual's traffic at unchanged fp32 compute
width -- IF `dtype=` rounds ties to even the way `ttnn.typecast` does. `dtype=` on a ttnn binary
op is an output cast and not a compute width, and nothing says which rounding it uses, so this
is a torch.equal gate before it is a lever.

Ties are the whole question, so they are constructed rather than hoped for: `mid` holds f32
values exactly halfway between two bf16 neighbours, where round-half-away and round-half-even
disagree on every odd-mantissa case.
"""
import os
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ttnn                                                        # noqa: E402


def bf16_ties(n, seed=0):
    """f32 values exactly midway between adjacent bf16 numbers, as an (x, u) bf16 pair sum.

    A bf16 is an f32 with the low 16 mantissa bits zero. Setting bit 15 puts the value exactly
    on the midpoint, which is the only input where the two rounding rules differ.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
    bits = base.view(torch.int32)
    bits = torch.where(base == 0, torch.full_like(bits, 0x3F800000), bits)
    return (bits | 0x00008000).view(torch.float32)


def run(dev, x_t, u_t):
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    u = ttnn.from_torch(u_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.typecast(x, ttnn.float32, memory_config=cfg)
    b = ttnn.typecast(u, ttnn.float32, memory_config=cfg)
    # A: the shipped four-call path
    wide = ttnn.add(a, b)
    out_a = ttnn.typecast(wide, ttnn.bfloat16, memory_config=cfg)
    # B: the candidate, trailing cast folded onto the add
    out_b = ttnn.add(a, b, dtype=ttnn.bfloat16)
    return ttnn.to_torch(out_a), ttnn.to_torch(out_b)


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        cases = {}
        g = torch.Generator().manual_seed(7)
        n = 288 * 288
        cases["random bf16 pairs"] = (
            torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32),
            torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32))
        # equal magnitudes: where af2.py:317 measured the shipped disagreement at 11.2 %
        v = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
        cases["equal magnitudes"] = (v, v.clone())
        # exact ties: x + u lands exactly between two bf16 values
        t = bf16_ties(n)
        cases["constructed bf16 ties"] = (t / 2, t / 2)
        ok = True
        for name, (x_t, u_t) in cases.items():
            a, b = run(dev, x_t.reshape(1, 1, 288, 288), u_t.reshape(1, 1, 288, 288))
            same = torch.equal(a, b)
            diff = (a.to(torch.float32) != b.to(torch.float32))
            ok &= same
            print("%-26s torch.equal=%-5s  differing elements %d / %d (%.3f %%)"
                  % (name, same, int(diff.sum()), diff.numel(),
                     100.0 * float(diff.sum()) / diff.numel()))
        print("\nVERDICT: the folded add is %s"
              % ("BIT-EXACT against the shipped path -- the lever is free"
                 if ok else "NOT bit-exact -- it is an accuracy lever, not a free one"))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
