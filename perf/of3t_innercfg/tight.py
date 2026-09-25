#!/usr/bin/env python3
"""Is the zero in VJP.json a property of OF3's magnitudes, or of the op?

VJP.json measured `inner`'s config bit-identical on operands shaped like the trunk's, where
the cancellation is not tight: |g - inner| / |g| averages 1.08, so `dx = y*(g - inner)` loses
no significant bits. That leaves a live objection -- the config might matter in a regime the
trunk does not visit. So build that regime on purpose: hold y and drive g toward the constant
row that makes `g - inner` vanish, over six decades of tightness, and read the same two arms.

A reading that stays bit-identical as the cancellation goes to 1e-6 is a statement about
`ttnn.sum`, not about OpenFold3.
"""
from __future__ import annotations

import json, pathlib, platform, subprocess, sys, time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio  # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT)

import torch, ttnn  # noqa: E402
from tt_bio.autograd import precise_config  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

nrm = torch.linalg.vector_norm


def main():
    dev = get_device()
    cfg = precise_config()
    torch.manual_seed(17)
    B, H, n = 4, 4, 384
    s = torch.randn(B, H, n, n) * 3.0
    y = torch.softmax(s.to(torch.bfloat16).double(), dim=-1).to(torch.bfloat16)
    yt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    y64 = ttnn.to_torch(yt).double()
    base = torch.randn(B, H, n, n).double()

    rows = []
    for eps in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        # g = a constant per row plus eps*noise. As eps falls, inner -> the constant and
        # `g - inner` -> eps*(noise - its y-weighted mean): an arbitrarily tight cancellation.
        const = torch.randn(B, H, n, 1).double() * 4.0
        g64h = const + eps * base
        gt = ttnn.from_torch(g64h.to(torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev)
        g64 = ttnn.to_torch(gt).double()
        inner_ref = (g64 * y64).sum(-1, keepdim=True) / y64.sum(-1, keepdim=True)
        dx_ref = y64 * (g64 - inner_ref)
        den = ttnn.sum(yt, dim=-1, keepdim=True, compute_kernel_config=cfg)
        prod = ttnn.multiply(gt, yt)
        arms = {}
        first = None
        for tag, kw in (("ship_no_cfg", {}), ("fix_precise", {"compute_kernel_config": cfg})):
            i_tt = ttnn.divide(ttnn.sum(prod, dim=-1, keepdim=True, **kw), den)
            i = ttnn.to_torch(i_tt).double()
            dx = ttnn.to_torch(ttnn.multiply(yt, ttnn.subtract(gt, i_tt))).double()
            if first is None:
                first = (i, dx)
            arms[tag] = dict(
                inner_rel_l2=float(nrm(i - inner_ref) / nrm(inner_ref)),
                dx_rel_l2=float(nrm(dx - dx_ref) / nrm(dx_ref)),
                bitwise_equal_to_ship=bool(torch.equal(i, first[0]) and torch.equal(dx, first[1])))
            ttnn.deallocate(i_tt)
        tight = float((g64 - inner_ref).abs().mean() / g64.abs().mean())
        rows.append(dict(eps=eps, tightness_mean_abs=tight,
                         digits_cancelled=round(-torch.log10(torch.tensor(tight)).item(), 2),
                         arms=arms))
        print(json.dumps(rows[-1]))
        ttnn.deallocate(gt); ttnn.deallocate(prod)

    rep = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=platform.node(),
               commit=subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               shape=[B, H, n, n], rows=rows)
    p = pathlib.Path(_ROOT) / "perf/of3t_innercfg/TIGHT.json"
    p.write_text(json.dumps(rep, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
