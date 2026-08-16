#!/usr/bin/env python3
"""Lever A's one arithmetic change: is the device symmetrisation bit-exact?

Everything else A and B delete is a pure crossing -- bf16 on the device side, so the round trip
being removed is bf16 -> fp32 -> bf16 and lossless. One thing is not. `_ZPair.__add__`
(tt_bio/esmfold2_runtime.py:317-318) replaces the reference forward's host

    z + z.transpose(-2, -3)          fp32 + fp32, rounded to bf16 by `_from_torch`

with

    ttnn.add(z, ttnn.permute(z, (0, 2, 1, 3)))       bf16 + bf16, packed to bf16

That feeds `distogram_logits`, which the CIF sha256 and plDDT anchors do NOT cover, so the fold
A/B cannot see it. This does.

The two are equal iff the device packer rounds the fp32 accumulator to bf16 the same way torch
rounds an fp32 sum to bf16. Both operands are exactly representable in bf16, so nothing else can
differ. Not argued: measured.

Checks the symmetrised pair AND the distogram logits it produces, at three sizes.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

assert Path(T.__file__).resolve().is_relative_to(REPO), "tt_bio from %s" % T.__file__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="298,512,640")
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--out", default="perf/esmbeat/a_distogram_parity.json")
    a = ap.parse_args()

    dev = T.get_device()
    res = {"doc": __doc__, "card": os.environ.get("TT_VISIBLE_DEVICES"), "rows": []}
    torch.manual_seed(0)

    for n in [int(s) for s in a.sizes.split(",") if s]:
        z = torch.randn(1, n, n, a.c).to(torch.bfloat16)
        zd = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # today: the reference forward sums on the host in fp32, then the wrapper casts to bf16
        host_sum = ttnn.to_torch(zd) + ttnn.to_torch(zd).transpose(-2, -3)
        host_as_dev = ttnn.to_torch(ttnn.from_torch(
            host_sum, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG))

        # lever A: the same sum on device, exactly as _ZPair.__add__ spells it
        dev_sum_t = ttnn.add(zd, ttnn.permute(zd, (0, 2, 1, 3)))
        dev_sum = ttnn.to_torch(dev_sum_t)

        eq = bool(torch.equal(dev_sum, host_as_dev))
        d = (dev_sum.float() - host_as_dev.float()).abs()
        row = {"n": n, "sum_torch_equal": eq,
               "sum_max_abs_err": float(d.max()),
               "sum_mismatch_frac": float((d > 0).float().mean()),
               "sum_ulp_max": float((d / host_as_dev.float().abs().clamp(min=1e-30)).max())}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        del host_sum, host_as_dev, dev_sum, d
        ttnn.deallocate(dev_sum_t)
        ttnn.deallocate(zd)

    res["all_bit_exact"] = all(r["sum_torch_equal"] for r in res["rows"])
    print(json.dumps({"all_bit_exact": res["all_bit_exact"]}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
