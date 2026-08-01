"""Pin down Transition's chunk-invariance failure: which widths break it, and which op inside it.

Established: `Transition` depth-chunked is bit-exact at 288 tokens and DIFFERS at 285 (maxabs 1.28e2),
so the trigger is the token (W) axis not being a tile multiple, not the depth chunking itself. Two
things are still unknown and both are pure measurement:

  1. the exact trigger condition -- every non-multiple-of-32, or only some residues?
  2. which op inside swiglu does it -- layer_norm, one of the three linears, the multiply, or the
     ttnn.chunk/concat around them.

Both are answered by sweeping W and by running the inner ops directly, chunked the same way
Transition chunks (over dim=1, its H axis). Seconds per case on one chip.

Run on the Galaxy with one chip free:
    TT_VISIBLE_DEVICES=<n> python3 -u scripts/abag_xm/probe_transition_width.py
"""
import os
import torch
import ttnn

from tt_bio.tenstorrent import (get_device, MSA_CHUNK_SIZE, Transition, _dtype,
                                CORE_GRID_MAIN)

D = int(os.environ.get("PROBE_DEPTH", 2048))     # smaller than the real 8722: still >1 chunk, faster
C_M = 128
CHUNK = int(os.environ.get("PROBE_CHUNK", MSA_CHUNK_SIZE))
WIDTHS = [int(x) for x in os.environ.get("PROBE_WIDTHS", "285,288").split(",")]
# Depths chosen for their REMAINDER mod the chunk width: D=2048 divides evenly, D=8722 leaves 18
# rows (less than one 32-row tile), D=8998 leaves 294 (more than a tile). The first failing case
# changed depth AND width together, so this separates them.
DEPTHS = [int(x) for x in os.environ.get("PROBE_DEPTHS", "2048,8722,8998,8736").split(",")]


def up(t, dev):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dtype())


def verdict(name, a, b):
    if torch.equal(a, b):
        print(f"  {name:44s} BIT-EXACT")
        return True
    d = (a.float() - b.float()).abs()
    print(f"  {name:44s} DIFFERS  maxabs {d.max():.3e}  frac {(d > 0).float().mean():.5f}")
    return False


def sweep_widths(dev):
    """Transition, whole vs depth-chunked, over a (depth, width) matrix."""
    print(f"\n=== Transition (depth x width) sweep, chunk={CHUNK} ===")
    H = 4 * C_M
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)
    for Dv in DEPTHS:
      for W in WIDTHS:
        torch.manual_seed(0)
        w = {"norm.weight": torch.randn(C_M), "norm.bias": torch.randn(C_M),
             "fc1.weight": torch.randn(H, C_M), "fc2.weight": torch.randn(H, C_M),
             "fc3.weight": torch.randn(C_M, H)}
        tr = Transition(w, kc)
        x = up(torch.randn(1, Dv, W, C_M), dev)
        whole = ttnn.to_torch(tr(x))
        parts = [tr(x[:, s:min(s + CHUNK, Dv), :, :]) for s in range(0, Dv, CHUNK)]
        chunked = ttnn.to_torch(ttnn.concat(parts, dim=1))
        rem = Dv % CHUNK
        verdict(f"D={Dv:5d} rem={rem:3d} (rem%32={rem % 32:2d})  W={W:4d} (W%32={W % 32:2d})",
                whole, chunked)
        ttnn.deallocate(x)


def inner_ops(dev, W):
    """The ops swiglu is built from, chunked over dim=1 exactly as Transition chunks its H axis.

    Names the guilty op without having to read tt-metal: whichever line flips from BIT-EXACT to
    DIFFERS at a non-tile-multiple W is the one that mishandles the pad columns.
    """
    print(f"\n=== inner ops at W={W} (W%32={W % 32}), chunked over dim=1 ===")
    torch.manual_seed(0)
    H = 4 * C_M
    x = up(torch.randn(1, D, W, C_M), dev)
    lnw, lnb = up(torch.randn(C_M), dev), up(torch.randn(C_M), dev)
    w1 = up(torch.randn(C_M, H), dev)
    w3 = up(torch.randn(H, C_M), dev)
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, packer_l1_acc=True)

    def chunks(t):
        return [t[:, s:min(s + CHUNK, D), :, :] for s in range(0, D, CHUNK)]

    ln = lambda t: ttnn.layer_norm(t, weight=lnw, bias=lnb, epsilon=1e-5,
                                  compute_kernel_config=kc, memory_config=ttnn.L1_MEMORY_CONFIG)
    verdict("layer_norm", ttnn.to_torch(ln(x)),
            ttnn.to_torch(ttnn.concat([ln(c) for c in chunks(x)], dim=1)))

    lin1 = lambda t: ttnn.linear(t, w1, activation="silu", compute_kernel_config=kc,
                                 dtype=_dtype(), core_grid=CORE_GRID_MAIN,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
    verdict("linear C->4C (silu, L1)", ttnn.to_torch(lin1(x)),
            ttnn.to_torch(ttnn.concat([lin1(c) for c in chunks(x)], dim=1)))

    xh = up(torch.randn(1, D, W, H), dev)
    lin3 = lambda t: ttnn.linear(t, w3, compute_kernel_config=kc, dtype=_dtype(),
                                 core_grid=CORE_GRID_MAIN,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
    verdict("linear 4C->C (DRAM out)", ttnn.to_torch(lin3(xh)),
            ttnn.to_torch(ttnn.concat([lin3(c) for c in chunks(xh)], dim=1)))

    verdict("ttnn.chunk+concat round-trip only", ttnn.to_torch(x),
            ttnn.to_torch(ttnn.concat(chunks(x), dim=1)))


def main():
    dev = get_device()
    sweep_widths(dev)
    print("done")


if __name__ == "__main__":
    main()
