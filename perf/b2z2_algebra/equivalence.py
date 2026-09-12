#!/usr/bin/env python3
"""The identity K2 rests on, checked at the production shape in fp64 and in bf16.

    [x @ Wq | x @ Wk | x @ Wv | x @ Wg]  ==  x @ [Wq | Wk | Wv | Wg]

Column block n of the right-hand side contracts x with column block n of the weight and with
nothing else, so it is the same sum of the same products in the same order. The identity is exact
in any arithmetic that computes each output element independently, which is what a matmul whose
N_block is one tile does -- so the device claim is bit-exactness, not a tolerance.

Two checks:
  fp64  -- max abs difference at [512, 512, 128] x [128, 512], the real Boltz-2 shape.
  bf16  -- the same in bf16 storage with an fp32 accumulator, the device's arithmetic, to show
           the reassociation the fusion does NOT do: none.

Host-only. The claim that matters is `torch.equal` on the real module, which perf/b2z2_algebra/
k2_ab.py measures on device; this file is the reason that result is not a coincidence.
"""
import torch

CZ, HEADS, HEAD_DIM, N = 128, 4, 32, 512
D = HEADS * HEAD_DIM


def check(dtype, acc=torch.float64):
    g = torch.Generator().manual_seed(11)
    x = (torch.randn(N * N, CZ, generator=g, dtype=torch.float32)).to(dtype)
    ws = [(torch.randn(CZ, D, generator=g, dtype=torch.float32) * 0.05).to(dtype)
          for _ in range(4)]
    sep = torch.cat([(x.to(acc) @ w.to(acc)) for w in ws], dim=-1)
    fused = x.to(acc) @ torch.cat(ws, dim=-1).to(acc)
    return bool(torch.equal(sep, fused)), float((sep - fused).abs().max())


def main():
    for name, dt in (("fp64", torch.float64), ("bf16 storage / fp32 acc", torch.bfloat16)):
        acc = torch.float64 if dt is torch.float64 else torch.float32
        eq, mx = check(dt, acc)
        print(f"{name:<26} x[{N*N},{CZ}] @ W[{CZ},{4*D}]   equal={eq}  max_abs={mx}")


if __name__ == "__main__":
    main()
