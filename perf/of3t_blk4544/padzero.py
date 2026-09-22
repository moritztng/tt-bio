#!/usr/bin/env python3
"""Zero the PAD cells of a captured boundary, and prove the reference does not notice.

`perf/of3t_widthattr/GROWTH.json` records that the two widths are the same object only on the 56
real tokens: `s_real_norm` and `z_real_norm` are bit-identical between padded 64 and padded 384,
while `z_norm` goes 1320432.531462332 to 16158081.287375676. At padded 384 the pair track's
boundary carries 251x more norm on its pad cells than on its real ones, and that pad total grows
12.2x with width while the real total does not move at all.

A correctly masked model is indifferent to what is in a pad cell. That is a claim, and this makes
it testable: zero every pad cell of the boundary and run the same arm. If the float64 reference's
real-token gradients are unchanged -- which is the control, not the result -- then the pads are
not load-bearing and any change OUR arm shows is pad contamination leaking through a reduction.

Writes the zeroed boundary beside the original and reports how much mass it removed.
"""
from __future__ import annotations

import argparse
import json

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    b = torch.load(a.boundary, map_location="cpu", weights_only=False)
    m1 = (b["single_mask"] > 0).to(torch.float64)                      # [1, N]
    N = int(m1.shape[-1])
    real = int(m1.sum())
    m2 = (m1.reshape(1, N, 1) * m1.reshape(1, 1, N))                   # [1, N, N]

    rep = {"boundary": a.boundary, "out": a.out,
           "tokens": {"real": real, "padded": N},
           "before": {}, "after": {}, "removed": {}}

    def norm(t):
        return float(torch.linalg.vector_norm(t.reshape(-1)))

    for key, msk in (("s_in", m1.reshape(1, N, 1)),
                     ("z_in", m2.reshape(1, N, N, 1)),
                     ("s_ref_050_captured", m1.reshape(1, N, 1)),
                     ("z_ref_050_captured", m2.reshape(1, N, N, 1))):
        if key not in b:
            continue
        t = b[key]
        rep["before"][key] = norm(t)
        real_only = t * msk
        rep["after"][key] = norm(real_only)
        rep["removed"][key] = norm(t - real_only)
        b[key] = real_only

    torch.save(b, a.out)
    json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
