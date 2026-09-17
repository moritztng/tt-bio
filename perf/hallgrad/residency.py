#!/usr/bin/env python3
"""Where does one pairformer block's DRAM actually go?

Phase 3 measured a 512 aa ceiling at a byte-identical 34.0 GB with and without per-block
checkpointing, which says the block runs out inside its own forward. The leading hypothesis
was that these tape ops never deallocate an intermediate. This localises it: run the block
untaped, stage by stage, and print DRAM allocated after each stage plus the number of pair
tensors that accounts for.
"""
import argparse
import sys

import numpy as np
import torch

PAIR = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--taped", action="store_true",
                    help="build the tape, i.e. what the real gradient path holds")
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    n, c_z, heads, head_dim = args.n, args.c_z, args.heads, args.head_dim
    pair_mb = n * n * c_z * 2 / 2 ** 20
    print(f"# N={n} c_z={c_z} heads={heads}; one [N,N,c_z] bf16 pair tensor = {pair_mb:.1f} MB")

    def dram_mb():
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / 2 ** 20

    base = dram_mb()
    rng = np.random.default_rng(1)

    def W(i, o):
        return ag.Tensor(ttnn.from_torch(
            torch.from_numpy(rng.standard_normal((i, o)) / np.sqrt(i)).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))

    def G(c):
        return ag.Tensor(ttnn.from_torch(torch.ones(1, c, dtype=torch.bfloat16),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=device))
    w = {"a": W(c_z, c_z), "ag": W(c_z, c_z), "b": W(c_z, c_z), "bg": W(c_z, c_z),
         "g": W(c_z, c_z), "z": W(c_z, c_z), "ln": G(c_z), "ln2": G(c_z),
         "q": W(c_z, heads * head_dim), "k": W(c_z, heads * head_dim),
         "v": W(c_z, heads * head_dim), "gt": W(c_z, heads * head_dim),
         "bias": W(c_z, heads), "o": W(heads * head_dim, c_z),
         "ta": W(c_z, 4 * c_z), "tb": W(c_z, 4 * c_z), "to": W(4 * c_z, c_z)}
    after_w = dram_mb()
    print(f"{'stage':<28} {'DRAM_MB':>9} {'delta_MB':>9} {'pair tensors':>13}")
    print(f"{'weights':<28} {after_w - base:>9.1f} {after_w - base:>9.1f} "
          f"{(after_w - base) / pair_mb:>13.2f}")

    z = ag.Tensor(ttnn.from_torch(torch.randn(n, n, c_z).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
    prev = dram_mb()

    def mark(label):
        nonlocal prev
        now = dram_mb()
        print(f"{label:<28} {now - base:>9.1f} {now - prev:>9.1f} "
              f"{(now - prev) / pair_mb:>13.2f}")
        prev = now

    mark("z (block input)")
    if args.taped:
        z.requires_grad = True
    import contextlib
    with (contextlib.nullcontext() if args.taped else ag.no_grad()):
        for tag, incoming in (("out", False), ("in", True)):
            zn = ag.layer_norm(z, w["ln"])
            mark(f"trimul {tag}: layer_norm")
            a = ag.mul(ag.sigmoid(ag.linear(zn, w["ag"])), ag.linear(zn, w["a"]))
            b = ag.mul(ag.sigmoid(ag.linear(zn, w["bg"])), ag.linear(zn, w["b"]))
            mark(f"trimul {tag}: a and b gates")
            x = ag.layer_norm(ag.pair_contract(a, b, incoming=incoming), w["ln2"])
            mark(f"trimul {tag}: contract + ln")
            gate = ag.sigmoid(ag.linear(zn, w["g"]))
            z = ag.add(z, ag.mul(gate, ag.linear(x, w["z"])))
            mark(f"trimul {tag}: gate + residual")
            del zn, a, b, x, gate
            mark(f"trimul {tag}: after del locals")
        for tag in ("start", "end"):
            zin = ag.permute(z, (1, 0, 2)) if tag == "end" else z
            zn = ag.layer_norm(zin, w["ln"])
            mark(f"triatt {tag}: layer_norm")

            def heads_of(key):
                h = ag.linear(zn, w[key])
                return ag.permute(ag.reshape(h, [n, n, heads, head_dim]), (0, 2, 1, 3))
            q, k, v, g = (heads_of(x) for x in ("q", "k", "v", "gt"))
            mark(f"triatt {tag}: q k v gate")
            bias = ag.reshape(ag.permute(ag.linear(zn, w["bias"]), (2, 0, 1)),
                              [1, heads, n, n])
            o = ag.triangle_attention(q, k, v, bias, scale=head_dim ** -0.5,
                                      chunk=args.chunk, q_chunk=args.chunk)
            mark(f"triatt {tag}: attention")
            o = ag.mul(o, ag.sigmoid(g))
            o = ag.reshape(ag.permute(o, (0, 2, 1, 3)), [n, n, heads * head_dim])
            upd = ag.linear(o, w["o"])
            z = ag.add(z, ag.permute(upd, (1, 0, 2)) if tag == "end" else upd)
            mark(f"triatt {tag}: gate + residual")
            del zn, q, k, v, g, bias, o, upd, zin
            mark(f"triatt {tag}: after del locals")
        zn = ag.layer_norm(z, w["ln"])
        ta = ag.linear(zn, w["ta"])
        swi = ag.mul(ag.mul(ta, ag.sigmoid(ta)), ag.linear(zn, w["tb"]))
        mark("transition: swiglu 4x")
        z = ag.add(z, ag.linear(swi, w["to"]))
        del zn, ta, swi
        mark("transition: residual + del")
    live = dram_mb() - base
    print(f"\n# live at end of a {'TAPED' if args.taped else 'UNTAPED'} block forward: "
          f"{live:.1f} MB = {live / pair_mb:.2f} pair tensors")
    print(f"# one 48-block trunk at this retention would be {live * 48 / 1024:.1f} GB, "
          f"against 34.23 GB of card DRAM")
    for n2 in (512, 800):
        scaled = live * (n2 / n) ** 2
        print(f"# scaled to {n2} aa (N^2): {scaled / 1024:.2f} GB per block, "
              f"{scaled * 48 / 1024:.0f} GB for 48")
    return 0


if __name__ == "__main__":
    sys.exit(main())
