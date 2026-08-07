#!/usr/bin/env python3
"""Stats for two pairlayer_capacity legs: PCC, rel median, frac_diff, worst positions."""
import sys

import torch

a_s, a_z = torch.load(sys.argv[1], map_location="cpu")
b_s, b_z = torch.load(sys.argv[2], map_location="cpu")
for name, a, b in (("s", a_s, b_s), ("z", a_z, b_z)):
    af, bf = a.float(), b.float()
    same = torch.equal(a, b)
    d = (af - bf).abs()
    scale = af.abs().max().item()
    rel = (d / af.abs().clamp(min=1.0))
    pcc = torch.corrcoef(torch.stack([af.flatten(), bf.flatten()]))[0, 1].item()
    frac = (d > 0).float().mean().item()
    # where are the worst diffs? row/col position of the top-5
    flat = d.flatten()
    top = torch.topk(flat, min(5, flat.numel()))
    idx = top.indices.tolist()
    pos = [tuple(torch.unravel_index(torch.tensor(i), d.shape)) for i in idx]
    print(f"{name}: {'BIT-EXACT' if same else 'DIFFERS'} maxabs {d.max().item():.4e} "
          f"scale {scale:.3e} rel_med {rel.median().item():.3e} frac_diff {frac:.4f} PCC {pcc:.8f}")
    if not same:
        print(f"   worst at {pos}")
