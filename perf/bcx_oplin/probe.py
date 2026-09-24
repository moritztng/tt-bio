"""`ops.linear` as the models call it, rank-3/4 against the 2-D view, on this card.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python perf/bcx_oplin/probe.py out.json

Every shape twice: with the model wrappers' own `core_grid=CORE_GRID_MAIN`, and without a
core grid (ttnn picks). HiFi4 fp32-acc config, bf16 out, both arms through `ops.linear` itself with `_via2d` switched. 5 x 20 calls, median,
AICLK sampled during each timed window. Accuracy is graded against a float64 product of the
same bf16 operands, never against the other arm. Also asserts that both arms return the same
shape, which is what a caller downstream actually depends on.
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
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device

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
]


def main(out):
    dev = get_device()
    ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                           packer_l1_acc=True)
    torch.manual_seed(0)
    rows = []
    for (xs, k, n, has_b), grid in [(c, g) for c in SHAPES for g in (CORE_GRID_MAIN, None)]:
        xh = torch.randn(xs).bfloat16()
        wh = (torch.randn(k, n) / k ** 0.5).bfloat16()
        bh = torch.randn(n).bfloat16() if has_b else None
        ref = xh.double() @ wh.double() + (bh.double() if has_b else 0)
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        w = ttnn.from_torch(wh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        b = (ttnn.from_torch(bh.reshape(1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=dev) if has_b else None)
        call = lambda: ops.linear(x, w, bias=b, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                  core_grid=grid)
        row = dict(shape=list(xs), k=k, n=n, bias=has_b, core_grid=grid is not None)
        outs = {}
        for name, on in (("rank", False), ("view", True)):
            with arm(on):
                y = call()
                outs[name] = y
                row[f"{name}_shape"] = [int(d) for d in y.shape]
                yt = ttnn.to_torch(y).double().reshape(ref.shape)
                row[f"{name}_rel_l2"] = float((yt - ref).norm() / ref.norm())
        for name, on in (("rank", False), ("view", True)):
            row[f"{name}_us"] = []
        with Clock() as clk:
            for _ in range(5):  # interleaved so both arms see the same clock window
                for name, on in (("rank", False), ("view", True)):
                    with arm(on):
                        call(); ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(20):
                            call()
                        ttnn.synchronize_device(dev)
                        row[f"{name}_us"].append((time.perf_counter() - t0) / 20 * 1e6)
        for name in ("rank", "view"):
            row[f"{name}_us"] = round(statistics.median(row[f"{name}_us"]), 1)
        row["ratio"] = round(row["rank_us"] / row["view_us"], 2)
        row["same_shape"] = row["rank_shape"] == row["view_shape"]
        row["bit_identical"] = bool(torch.equal(ttnn.to_torch(outs["rank"]).reshape(ref.shape),
                                                ttnn.to_torch(outs["view"]).reshape(ref.shape)))
        row["aiclk"] = clk.stats()
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
