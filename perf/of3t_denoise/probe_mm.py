#!/usr/bin/env python3
"""of3t-denoise D257: which ttnn.matmul / ttnn.linear forms are wrong on mixed operand dtypes.

    probe_mm.py --out F.json [--fixed]

`--fixed` issues every matmul through `tt_bio.autograd._matmul` instead of `ttnn.matmul`: the
same table on the helper the backward closures now call.

Every transpose form and both dtype orders, fresh operands, against float64 on the device's
own read-back values. `rel` is over the finite part; `bad` counts non-finite or > 1e30.
"""
import json
import sys
from pathlib import Path

import torch


def main():
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    cfg = ag.precise_config()
    g = torch.Generator().manual_seed(1)
    F, B = ttnn.float32, ttnn.bfloat16
    mm = ag._matmul if "--fixed" in sys.argv else ttnn.matmul
    res = {}
    for ta in (False, True):
        for tb in (False, True):
            for da, db in ((F, F), (B, B), (F, B), (B, F)):
                for od in (None, F):
                    M, K, N = 64, 256, 384
                    A = torch.randn(K, M, generator=g) if ta else torch.randn(M, K, generator=g)
                    Bm = torch.randn(N, K, generator=g) if tb else torch.randn(K, N, generator=g)
                    Ad = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev, dtype=da)
                    Bd = ttnn.from_torch(Bm, layout=ttnn.TILE_LAYOUT, device=dev, dtype=db)
                    Ah, Bh = ttnn.to_torch(Ad).double(), ttnn.to_torch(Bd).double()
                    ref = (Ah.T if ta else Ah) @ (Bh.T if tb else Bh)
                    kw = {"dtype": od} if od is not None else {}
                    C = ttnn.to_torch(mm(Ad, Bd, transpose_a=ta, transpose_b=tb,
                                                  compute_kernel_config=cfg, **kw)).double()
                    bad = (~torch.isfinite(C)) | (C.abs() > 1e30)
                    Cf = torch.where(bad, torch.zeros_like(C), C)
                    key = (f"ta{int(ta)}_tb{int(tb)}_a{str(da).split('.')[-1]}"
                           f"_b{str(db).split('.')[-1]}_out{str(od).split('.')[-1] if od else 'default'}")
                    res[key] = {"bad": int(bad.sum()), "rel": float((Cf - ref).norm() / ref.norm())}
                    print(key, res[key], flush=True)
    # ttnn.linear as the forward calls it: x @ w, no transpose, mixed.
    for da, db in ((F, B), (B, F)):
        x = torch.randn(64, 833, generator=g); w = torch.randn(833, 384, generator=g)
        xd = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=da)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=db)
        ref = ttnn.to_torch(xd).double() @ ttnn.to_torch(wd).double()
        C = ttnn.to_torch(ttnn.linear(xd, wd, compute_kernel_config=cfg)).double()
        key = f"linear_a{str(da).split('.')[-1]}_b{str(db).split('.')[-1]}"
        res[key] = {"rel": float((C - ref).norm() / ref.norm())}
        print(key, res[key], flush=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
