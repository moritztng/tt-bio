#!/usr/bin/env python3
"""of3t-fullstep64: distogram.linear.weight scored by hand, without the bijection or score.py.

One tensor, one reshape, one transpose. The device keeps W^T ([c_z, n_bins]) for its matmul and
dumps it flat; upstream holds W ([n_bins, c_z]). The wrong orientation is printed beside the right
one so a layout slip cannot pass as a small error.

    handcheck.py F64_GRADS NAME=DEVICE_GRADS ...
"""
import sys
import torch

KEY = "aux_heads.distogram.linear.weight"
DEV = "confidence_head._wd_cache.(\x27distogram.linear.weight\x27, True, DataType.BFLOAT16)"

ref = torch.load(sys.argv[1], weights_only=False)[KEY].to(torch.float64)
nb, cz = ref.shape
print(f"f64 {KEY} {tuple(ref.shape)} |g|^2 {float((ref * ref).sum()):.6f}")
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    g = torch.load(path, weights_only=False)[DEV].to(torch.float64)
    right = g.reshape(cz, nb).t()
    wrong = g.reshape(nb, cz)
    rel = lambda a: float((a - ref).norm() / ref.norm())
    print(f"{name}: |g|^2 {float((right * right).sum()):.6f}  rel {rel(right):.6f}  "
          f"(untransposed reading: rel {rel(wrong):.6f})")
