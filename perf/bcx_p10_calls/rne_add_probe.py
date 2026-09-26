#!/usr/bin/env python3
"""What the residual's four calls can be replaced by, graded on REAL ties.

`AF2PairBlock._residual` at `rne_residual = True` is 30 B/element:

    typecast(x,f32) 6 | typecast(u,f32) 6 | add_ 12 | typecast(wide,bf16) 6

and the census sees only the 18 B of typecast. The path exists so the sum rounds half-to-even,
which `af2.py:317-322` measures as the whole of this trunk's error growth. So every candidate
is graded on whether it reproduces `round_rne_bf16(exact_sum)`, not on how close it is.

CANDIDATES (bytes per element of the whole residual)

    A     shipped                                        30   reference
    FOLD  typecast,typecast, add(f32,f32,dtype=bf16)      22   leg 1
    F     typecast(x,f32),  add(f32,bf16,dtype=bf16)      14   one up-cast survives
    G     typecast(u,f32),  add(bf16,f32,dtype=bf16)      14   the other one
    B     add(bf16,bf16)                                   6   leg 2's ideal
    BD    add(bf16,bf16,dtype=bf16)                        6

TIES ARE THE WHOLE QUESTION AND THEY HAVE TO BE REAL. A tie is a sum landing exactly on a bf16
midpoint. `x + x` is NOT one: doubling a bf16 value is exact in every format, so a probe built
that way rounds nothing and passes whatever it is shown. A real tie is `x + halfulp(x)`, where
`halfulp` is a power of two and therefore itself bf16-representable, so both operands reach the
card unrounded and the f32 sum sits exactly between two bf16 neighbours. Round-half-even keeps
the even mantissa; round-half-away always goes up; they disagree on every odd-mantissa element.
"""
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ttnn                                                        # noqa: E402

N = 288


def bf16(t):
    """f32 values that are exactly bf16-representable, so `from_torch` rounds nothing."""
    return t.to(torch.bfloat16).to(torch.float32)


def half_ulp(x):
    """2^(e-8) for each element of bf16-representable `x`: half the gap to its bf16 neighbour."""
    e = (x.view(torch.int32) >> 23) & 0xFF
    return ((e - 8).clamp(min=1) << 23).view(torch.float32)


def tie_pair(n, seed=0):
    """(x, h) with x bf16 and h = halfulp(x), so x + h is exactly a bf16 midpoint."""
    g = torch.Generator().manual_seed(seed)
    x = bf16(torch.randn(n, generator=g).abs() + 0.5)
    return x, half_ulp(x)


def odd_mantissa(x):
    return (((x.view(torch.int32) >> 16) & 1) == 1)


def to_dev(dev, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def variants(dev, x_t, u_t):
    cfg = ttnn.DRAM_MEMORY_CONFIG
    x = to_dev(dev, x_t, ttnn.bfloat16)
    u = to_dev(dev, u_t, ttnn.bfloat16)
    # from_torch must have rounded nothing, or the comparison grades the host
    assert torch.equal(ttnn.to_torch(x).to(torch.float32), x_t), "operand x not bf16-exact"
    assert torch.equal(ttnn.to_torch(u).to(torch.float32), u_t), "operand u not bf16-exact"
    a32 = ttnn.typecast(x, ttnn.float32, memory_config=cfg)
    b32 = ttnn.typecast(u, ttnn.float32, memory_config=cfg)
    out = {
        "A":    ttnn.typecast(ttnn.add(a32, b32), ttnn.bfloat16, memory_config=cfg),
        "FOLD": ttnn.add(a32, b32, dtype=ttnn.bfloat16),
        "F":    ttnn.add(a32, u, dtype=ttnn.bfloat16),
        "G":    ttnn.add(x, b32, dtype=ttnn.bfloat16),
        "B":    ttnn.add(x, u),
        "BD":   ttnn.add(x, u, dtype=ttnn.bfloat16),
    }
    return {k: ttnn.to_torch(v).to(torch.float32) for k, v in out.items()}


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        g = torch.Generator().manual_seed(7)
        n = N * N
        r1, r2 = bf16(torch.randn(n, generator=g)), bf16(torch.randn(n, generator=g))
        big = bf16(torch.randn(n, generator=g) * 64.0)
        small = bf16(torch.randn(n, generator=g) * 0.25)
        tx, th = tie_pair(n)
        cases = {
            "random bf16 pairs": (r1, r2),
            "ratio 256x": (big, small),
            "REAL ties (x + halfulp)": (tx, th),
            "REAL ties, negated": (-tx, -th),
        }
        keys = ("A", "FOLD", "F", "G", "B", "BD")
        totals = {k: 0 for k in keys}
        for name, (x_t, u_t) in cases.items():
            o = variants(dev, x_t.reshape(1, 1, N, N), u_t.reshape(1, 1, N, N))
            exact = (x_t.double() + u_t.double())
            ref = exact.to(torch.bfloat16).to(torch.float32).reshape(1, 1, N, N)
            print("\n== %s ==" % name)
            if name.startswith("REAL ties"):
                print("   %d / %d elements have an odd mantissa, where the two rounding rules "
                      "disagree" % (int(odd_mantissa(x_t).sum()), n))
            for k in keys:
                d = int((o[k] != ref).sum())
                totals[k] += d
                print("  %-5s vs round_rne_bf16(exact) : differ %6d / %d (%6.3f %%)"
                      % (k, d, ref.numel(), 100.0 * d / ref.numel()))
        print("\nTOTAL differing elements over %d cases x %d elements:" % (len(cases), n))
        for k in keys:
            print("  %-5s %7d   %s" % (k, totals[k],
                                       "EXACT" if totals[k] == 0 else "not exact"))
        print("\nVERDICT leg1 (FOLD): %s" % ("bit-exact on real ties" if totals["FOLD"] == 0
                                             else "NOT exact -- the fold changes rounding"))
        print("VERDICT leg2 (B/BD): %s" % ("a narrow add reproduces the wide path"
                                           if totals["BD"] == 0 else
                                           "a narrow add does NOT reproduce the wide path"))
        print("VERDICT mixed (F/G):  %s" % ("one up-cast is enough: 14 B/element, not 30"
                                            if totals["F"] == 0 or totals["G"] == 0 else
                                            "a mixed-dtype add is narrow too"))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
