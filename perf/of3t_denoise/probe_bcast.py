#!/usr/bin/env python3
"""of3t-denoise: ttnn.multiply(x, mask) with a last-dim-1 mask, by x dtype, mask dtype, rank and origin.

    probe_bcast.py --out F.json [--reps R]

probe_dcfwd.py found `_SwiGLUTransition`'s `multiply(out, mask_col)` returning garbage that varies
run to run, untaped too, when the mask is fp32. The rollout hands the same op a bf16 mask. The
mask is 0/1, so the exact product is known on host. Device memory is poisoned (large-valued
buffers allocated and freed) before every rep so a read of memory the op does not own shows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    reps = int(argv[argv.index("--reps") + 1]) if "--reps" in argv else 4
    import ttnn
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    g = torch.Generator().manual_seed(0)
    up = lambda x, d: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)  # noqa
    f32, b16 = ttnn.float32, ttnn.bfloat16

    def poison():
        junk = [up(torch.full((1, 64, 64, 128), 3.0e4), f32) for _ in range(8)]
        for j in junk:
            ttnn.deallocate(j)

    cases = {}
    for shape_x, shape_m in (((1, 64, 64, 128), (1, 64, 64, 1)), ((1, 64, 384), (1, 64, 1)),
                             ((1, 448, 128), (1, 448, 1))):
        x = torch.randn(*shape_x, generator=g)
        m = (torch.rand(*shape_m, generator=g) > 0.2).float()
        for xname, xd in (("xf32", f32), ("xbf16", b16)):
            xdev = up(x, xd)
            ref = ttnn.to_torch(xdev).double() * m.double()
            for mname, mk in (("f32_direct", lambda: up(m, f32)),
                              ("f32_typecast", lambda: ttnn.typecast(up(m, b16), f32)),
                              ("bf16", lambda: up(m, b16))):
                res = []
                for r in range(reps):
                    poison()
                    md = mk()
                    y = ttnn.multiply(xdev, md)
                    d = (ttnn.to_torch(y).double() - ref).abs()
                    res.append({"out_dtype": str(y.dtype), "max_abs_err": float(d.max()),
                                "n_wrong": int((d > 1e-6).sum())})
                    ttnn.deallocate(y)
                    ttnn.deallocate(md)
                key = f"{xname}{list(shape_x)}_m{list(shape_m)}_{mname}"
                cases[key] = res
                print(key, [(round(e["max_abs_err"], 3), e["n_wrong"]) for e in res], flush=True)
    out.write_text(json.dumps({"reps": reps, "cases": cases}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
