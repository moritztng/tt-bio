"""Do the in-place and out-of-place forms of the same ttnn op give the same bits?

The untaped Evoformer block calls `softmax_in_place`, `add_` and `multiply_` where the taped one
calls `softmax`, `add` and `multiply`: a tape cannot differentiate through a buffer it overwrote.
`op_shadow_diff.py` compares each taped op against the SAME shipped verb, so it cannot see a
difference between two different verbs. This compares the pairs directly, on the block's own
shapes, on identical bf16 inputs, with each op's default compute kernel config.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/inplace_probe.py --card <card>
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)
    import torch
    import ttnn

    device = ttnn.open_device(device_id=0)
    rng = torch.Generator().manual_seed(args.seed)

    def upload(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def down(t):
        return ttnn.to_torch(t).float()

    def compare(name, a, b):
        diff = (a - b).abs()
        rel = float(diff.norm() / b.norm()) if float(b.norm()) else float("nan")
        print(f"{name:<34} bit-identical={bool(torch.equal(a, b))!s:<6} "
              f"max|d|={float(diff.max()):.3g} rel L2={rel:.3g}", flush=True)

    try:
        for shape in ((64, 4, 64, 64), (1, 8, 64, 64)):
            x = torch.randn(shape, generator=rng) * 4
            out_of_place = down(ttnn.softmax(upload(x), dim=-1))
            buffer = upload(x)
            ttnn.softmax_in_place(buffer)
            compare(f"softmax{shape}", down(buffer), out_of_place)

        for a_shape, b_shape in (((64, 4, 64, 64), (1, 4, 64, 64)), ((1, 8, 64, 64), (1, 8, 64, 64))):
            a, b = torch.randn(a_shape, generator=rng), torch.randn(b_shape, generator=rng)
            out_of_place = down(ttnn.add(upload(a), upload(b)))
            buffer = upload(a)
            ttnn.add_(buffer, upload(b))
            compare(f"add{a_shape}+{b_shape}", down(buffer), out_of_place)

        for a_shape, b_shape in (((1, 64, 256), (1, 64, 256)), ((1, 64, 32), (1, 64, 1))):
            a, b = torch.randn(a_shape, generator=rng), torch.randn(b_shape, generator=rng)
            out_of_place = down(ttnn.multiply(upload(a), upload(b)))
            buffer = upload(a)
            ttnn.multiply_(buffer, upload(b))
            compare(f"multiply{a_shape}*{b_shape}", down(buffer), out_of_place)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
