#!/usr/bin/env python3
"""Confirm the mechanism behind ttnn.softmax's 2.4% mass deficit, and price the fix.

`probe_softmax_variants.py`: every ttnn.softmax variant lands on rel_rms 0.029192 against a
fp64 softmax, while max/exp/sum/divide out of individual ttnn ops lands on 0.000421 -- 69x.
The deficit is present at every logit range (rowsum_min 0.961 at within-row spread 1 as well
as 135), so it is the kernel, not RF3's data. The one difference between the two chains is
which SFPU exp they use: `ttnn.exp` defaults to the accurate one, the fused softmax kernel
does not. This pins that, then times both at the shapes RF3 runs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def rel_rms(a, b):
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().std())


def main() -> int:
    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    gen = torch.Generator().manual_seed(0)
    out = {"mechanism": [], "timing": []}

    # ---- mechanism: the same chain, accurate vs approximate exp ----
    x = torch.randn(1, 16, 512, 512, generator=gen) * 40.0
    ref = torch.softmax(x.double(), dim=-1).float()

    def tt(t):
        return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                              dtype=ttnn.float32)

    def chain(t_in, approx):
        m = ttnn.max(t_in, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(t_in, m), fast_and_approximate_mode=approx)
        return ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True,
                                       compute_kernel_config=cfg))

    for nm, got in (("fused ttnn.softmax", ttnn.softmax(tt(x), dim=-1)),
                    ("manual, accurate exp", chain(tt(x), False)),
                    ("manual, approximate exp", chain(tt(x), True))):
        p = torch.Tensor(ttnn.to_torch(got)).float()
        out["mechanism"].append({
            "variant": nm, "rel_rms_vs_fp64": round(rel_rms(p, ref), 6),
            "rowsum_mean": round(float(p.sum(-1).mean()), 6),
            "rowsum_min": round(float(p.sum(-1).min()), 6)})

    # ---- timing, warm, median of 7, at the rungs RF3 runs ----
    def timeit(fn, t_in, n=7):
        fn(t_in); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = fn(t_in)
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(r)
        ts.sort()
        return ts[len(ts) // 2] * 1e3

    for S in (512, 768, 1024):
        y = tt(torch.randn(1, 16, S, S, generator=gen) * 40.0)
        fused = timeit(lambda t: ttnn.softmax(t, dim=-1), y)
        man = timeit(lambda t: chain(t, False), y)
        out["timing"].append({"shape": [1, 16, S, S],
                              "fused_ms": round(fused, 3), "manual_ms": round(man, 3),
                              "manual_over_fused": round(man / fused, 2)})
        ttnn.deallocate(y)

    print(json.dumps(out, indent=2))
    Path("/tmp/sm_mech.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
