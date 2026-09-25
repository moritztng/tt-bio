#!/usr/bin/env python3
"""of3t-bcastaudit: which ttnn binary ops and broadcast axes share D259.

    probe_bcast.py --out F.json [--reps R]

Extends of3t-denoise's probe_bcast.py once: multiply, mul (alias), add and subtract, each with a
second operand broadcast on the last dim ([..., 1]) or on a non-last dim ([1, ..., C]), for every
(x dtype, b dtype) pair. b is 0/1 and the reference is float64 on the device's own inputs. An
element is wrong when it misses that reference by more than one ulp of the output dtype (x + 1
rounds in bf16, which is not the defect). Device memory is poisoned before every rep, and every
rep is also compared with rep 0: D259's signature is output that changes between identical calls.
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
    ops = {"multiply": (ttnn.multiply, torch.mul), "mul": (ttnn.mul, torch.mul),
           "add": (ttnn.add, torch.add), "subtract": (ttnn.subtract, torch.sub)}

    def poison():
        junk = [up(torch.full((1, 64, 64, 128), 3.0e4), f32) for _ in range(8)]
        for j in junk:
            ttnn.deallocate(j)

    shapes = {  # axis -> [(x shape, b shape)]
        "last": [((1, 64, 64, 128), (1, 64, 64, 1)), ((1, 64, 384), (1, 64, 1)),
                 ((1, 448, 128), (1, 448, 1))],
        "row": [((1, 64, 64, 128), (1, 1, 64, 128)), ((1, 448, 128), (1, 1, 128))],
        "outer": [((1, 64, 64, 128), (1, 64, 1, 128))],
    }
    cases, summary = {}, {}
    for axis, pairs in shapes.items():
        for shape_x, shape_b in pairs:
            x = torch.randn(*shape_x, generator=g)
            b = (torch.rand(*shape_b, generator=g) > 0.2).float()
            for xname, xd in (("xf32", f32), ("xbf16", b16)):
                xdev = up(x, xd)
                xh = ttnn.to_torch(xdev).double()
                for bname, bd in (("bf32", f32), ("bbf16", b16)):
                    for oname, (op, top) in ops.items():
                        ref = top(xh, b.double())
                        res, y0 = [], None  # per rep: [n_wrong, n_unlike_rep0]
                        for _ in range(reps):
                            poison()
                            bdev = up(b, bd)
                            y = op(xdev, bdev)
                            yh = ttnn.to_torch(y).double()
                            ulp = 2.0 ** -7 if y.dtype == b16 else 2.0 ** -14
                            y0 = yh if y0 is None else y0
                            res.append([int(((yh - ref).abs() > ref.abs() * ulp + 1e-6).sum()),
                                        int((yh != y0).sum())])
                            ttnn.deallocate(y)
                            ttnn.deallocate(bdev)
                        key = f"{oname}_{axis}_{xname}_{bname}_x{list(shape_x)}_b{list(shape_b)}"
                        cases[key] = res
                        s = summary.setdefault(f"{oname}_{axis}_{xname}_{bname}", [0, 0])
                        s[0] += sum(max(r) > 0 for r in res)
                        s[1] += len(res)
                        print(key, res, flush=True)
                ttnn.deallocate(xdev)
    fail = {k: f"{v[0]}/{v[1]} reps wrong or unlike rep 0" for k, v in summary.items() if v[0]}
    out.write_text(json.dumps({"reps": reps, "failing": fail, "summary": summary,
                               "cases": cases}, indent=1) + "\n")
    print("FAILING", json.dumps(fail, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
