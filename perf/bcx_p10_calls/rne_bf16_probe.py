#!/usr/bin/env python3
"""Leg 2 step 1: does a plain bf16 add round the way the fp32 residual path does?

`AF2PairBlock._residual` at `rne_residual = True` moves 18 B/element (2->4, 2->4, 4->2) so the
sum rounds half-to-even. A bf16-in/bf16-out add moves 6 B. The whole wide path is removable if
and only if the narrow add agrees with it, ties included.

Four candidates against the shipped path A, and against an exact float64 reference:

    A  typecast(x,f32); typecast(u,f32); add; typecast(bf16)      shipped, 18 B
    B  add(x_bf16, u_bf16)                                        6 B
    C  add(x_bf16, u_bf16, dtype=bf16)                            6 B
    D  add(x_bf16, u_bf16, dtype=f32)                             8 B, asks whether the
                                                                  DATAPATH is wide
    E  typecast(D, bf16)                                          14 B, the fallback if the
                                                                  datapath is wide and only the
                                                                  pack rounds wrong

D against the exact f64 sum separates the two ways a narrow add can be wrong: a narrow
datapath (the add itself loses bits) from a narrow pack (the add is exact, the store rounds
away from zero).
"""
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ttnn                                                        # noqa: E402

N = 288


def bf16_ties(n, seed=0):
    """f32 values exactly midway between adjacent bf16 numbers (bit 15 set)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
    bits = base.view(torch.int32)
    bits = torch.where(base == 0, torch.full_like(bits, 0x3F800000), bits)
    return (bits | 0x00008000).view(torch.float32)


def to_dev(dev, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def variants(dev, x_t, u_t):
    cfg = ttnn.DRAM_MEMORY_CONFIG
    x = to_dev(dev, x_t, ttnn.bfloat16)
    u = to_dev(dev, u_t, ttnn.bfloat16)
    a32 = ttnn.typecast(x, ttnn.float32, memory_config=cfg)
    b32 = ttnn.typecast(u, ttnn.float32, memory_config=cfg)
    A = ttnn.typecast(ttnn.add(a32, b32), ttnn.bfloat16, memory_config=cfg)
    B = ttnn.add(x, u)
    C = ttnn.add(x, u, dtype=ttnn.bfloat16)
    D = ttnn.add(x, u, dtype=ttnn.float32)
    E = ttnn.typecast(D, ttnn.bfloat16, memory_config=cfg)
    out = {k: ttnn.to_torch(v) for k, v in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E))}
    return out


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        g = torch.Generator().manual_seed(7)
        n = N * N
        r1 = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
        r2 = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
        v = torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
        t = bf16_ties(n)
        cases = {
            "random bf16 pairs": (r1, r2),
            "equal magnitudes": (v, v.clone()),
            "constructed bf16 ties": (t / 2, t / 2),
        }
        verdict = {}
        for name, (x_t, u_t) in cases.items():
            o = variants(dev, x_t.reshape(1, 1, N, N), u_t.reshape(1, 1, N, N))
            # exact reference: f64 sum of the two bf16-representable operands
            exact = (x_t.double() + u_t.double()).reshape(1, 1, N, N)
            ref_bf16 = exact.to(torch.bfloat16).to(torch.float64)
            print("\n== %s ==" % name)
            a = o["A"].to(torch.float64)
            print("  A (shipped 18 B) vs f64-rounded-RNE : equal=%s  differ %d / %d"
                  % (torch.equal(a, ref_bf16), int((a != ref_bf16).sum()), a.numel()))
            for k in ("B", "C", "E"):
                b = o[k].to(torch.float64)
                d = int((a != b).sum())
                verdict.setdefault(k, 0)
                verdict[k] += d
                print("  %s vs A                              : equal=%s  differ %d / %d (%.3f %%)"
                      % (k, torch.equal(o[k], o["A"]), d, b.numel(), 100.0 * d / b.numel()))
            d32 = o["D"].to(torch.float64)
            nd = int((d32 != exact).sum())
            verdict.setdefault("D_exact", 0)
            verdict["D_exact"] += nd
            mx = float((d32 - exact).abs().max())
            print("  D=add(bf16,bf16,dtype=f32) vs EXACT  : differ %d / %d (%.3f %%)  max|err| %.3e"
                  % (nd, d32.numel(), 100.0 * nd / d32.numel(), mx))
        print("\nSUMMARY across all three cases (differing elements):")
        for k in ("B", "C", "E", "D_exact"):
            print("  %-8s %d" % (k, verdict[k]))
        free = verdict["B"] == 0 or verdict["C"] == 0
        print("\nVERDICT leg2.1: a narrow add %s the wide path"
              % ("MATCHES" if free else "does NOT match"))
        print("VERDICT datapath: add(bf16,bf16)->f32 is %s vs the exact sum"
              % ("EXACT" if verdict["D_exact"] == 0 else "NOT exact"))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
