"""`ops.linear` as the models call it: rank-3/4, its 2-D view, and the view with ttnn's own grid.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python perf/bcx_oplin/probe.py out.json [shapes.json]

Per shape and per caller grid (the 11x10 that protenix/openfold3 import at module load, the
device grid `tenstorrent.py` reads after the open, and no grid at all), three arms of one
`ops.linear` call:

  rank         today: the rank-3/4 call as the caller wrote it
  view         `_via2d`: the same call on the (prod(leading), K) view
  view_auto    the view with the caller's core grid dropped, so ttnn picks the program

HiFi4 fp32-acc, bf16 out, 5 x 20 calls interleaved across arms, median, AICLK sampled during
the timed window. Accuracy is graded against a float64 product of the same bf16 operands, never
against another arm; `bits` says which arms return exactly the rank arm's bytes.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import ttnn

from common import Clock, arm
from tt_bio import ops
import tt_bio.tenstorrent as T

SHAPES = [  # (x shape, K, N, bias)
    ((256, 256, 128), 128, 128, False),
    ((1, 256, 256, 128), 128, 128, False),
    ((512, 512, 128), 128, 128, False),
    ((1, 512, 512, 128), 128, 128, True),
    ((512, 512, 128), 128, 512, False),
    ((512, 512, 512), 512, 128, False),
    ((512, 512, 128), 128, 16, False),
    ((1, 512, 384), 384, 1536, True),
    ((1, 512, 1536), 1536, 384, True),
    # esmfold2's outer-product-mean projection (esmfold2.py:1211), 9248 calls per 512 aa fold
    ((1, 32, 512, 1024), 1024, 256, True),
    ((1, 512, 512, 1024), 1024, 128, False),
    ((1, 512, 512, 256), 256, 128, False),
]


def main(out, shapes=SHAPES):
    grid_import = T.CORE_GRID_MAIN
    dev = T.get_device()
    grid_dev = T.CORE_GRID_MAIN
    grids = {f"{grid_import.x}x{grid_import.y}": grid_import}
    grids.setdefault(f"{grid_dev.x}x{grid_dev.y}", grid_dev)
    grids["none"] = None
    ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    rows = []
    for xs, k, n, has_b in shapes:
        xh = torch.randn(xs).bfloat16()
        wh = (torch.randn(k, n) / k ** 0.5).bfloat16()
        bh = torch.randn(n).bfloat16() if has_b else None
        ref = xh.double() @ wh.double() + (bh.double() if has_b else 0)
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        b = (ttnn.from_torch(bh.reshape(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev) if has_b else None)
        for gname, grid in grids.items():
            arms = {"rank": (False, grid), "view": (True, grid), "view_auto": (True, None)}

            def call(on, g):
                with arm(on):
                    return ops.linear(x, w, bias=b, compute_kernel_config=ckc,
                                      dtype=ttnn.bfloat16, core_grid=g)

            row = dict(shape=list(xs), k=k, n=n, bias=has_b, grid=gname)
            outs = {}
            for a, (on, g) in arms.items():
                y = call(on, g)
                row[f"{a}_shape"] = [int(d) for d in y.shape]
                outs[a] = ttnn.to_torch(y).reshape(ref.shape)
                row[f"{a}_rel_l2"] = float((outs[a].double() - ref).norm() / ref.norm())
            row["bits"] = {a: bool(torch.equal(outs["rank"], outs[a])) for a in arms}
            row["same_shape"] = len({tuple(row[f"{a}_shape"]) for a in arms}) == 1
            ts = {a: [] for a in arms}
            with Clock() as clk:
                for _ in range(5):
                    for a, (on, g) in arms.items():
                        call(on, g); ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(20):
                            call(on, g)
                        ttnn.synchronize_device(dev)
                        ts[a].append((time.perf_counter() - t0) / 20 * 1e6)
            for a in arms:
                row[f"{a}_us"] = round(statistics.median(ts[a]), 1)
            row["aiclk"] = clk.stats()
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    # optional second argument: a JSON list of [x shape, K, N, bias] to probe instead
    main(sys.argv[1], *([json.loads(sys.argv[2])] if len(sys.argv) > 2 else []))
