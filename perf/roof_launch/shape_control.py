#!/usr/bin/env python3
"""Does the per-op launch floor depend on the SHAPE at a fixed tile count, or only on the count?

`launch_sweep.py` fits one ladder per op class on one aspect ratio, [1, R, 768], and the join then
applies that class floor to every call of the class whatever its shape. `ttnn.layer_norm` is 58 %
of the launch term the join adds, and the fold runs it on three shapes this ladder never saw:
[1,512,768], [1,16,512,128] and [1,140,32,128]. If the floor moves with the aspect ratio at the
same tile count, the join's dominant term is carrying a shape it did not measure.

Same trace-replay reading as the sweep, same session discipline, on the fold's own shapes.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from launch_sweep import tiles, trace_point, roofs                            # noqa: E402

# the shapes the 512 aa fold actually launches these two classes on, from op_census_512.json
SHAPES = {
    "layer_norm": [(1, 512, 768), (1, 16, 512, 128), (1, 140, 32, 128), (1, 512, 384),
                   (1, 512, 1536), (1, 1024, 512, 64)],
    "layer_norm_w": [(1, 512, 768), (1, 16, 512, 128), (1, 140, 32, 128), (1, 512, 384)],
    "add": [(1, 512, 768), (1, 140, 32, 128), (1, 16, 512, 512), (1, 512, 512, 128)],
    "multiply_": [(1, 512, 768), (1, 512, 1536), (1, 16, 512, 512), (1, 140, 32, 128)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "shape_control.json")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--sweep", type=Path, default=HERE / "launch_trace_qb2c1.json")
    a = ap.parse_args()

    dev = T.get_device(trace_region_size=1 << 29)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    sweep = json.loads(a.sweep.read_text())

    def t(shape):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def curve(arm, x):
        pts = sorted((q["x_tiles"], q["us"]) for q in sweep["rows"][arm])
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            return y1 + (y1 - y0) / (x1 - x0) * (x - x1)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    out = {"host": platform.node(), "arch": str(dev.arch()), "reps": a.reps, "blocks": a.blocks,
           "reading": "device only, trace replay", "sweep": a.sweep.name,
           "loadavg_before": open("/proc/loadavg").read().split()[:3], "rows": []}
    out["roofs"] = roofs(dev, kc, 5)
    print("cube4096 %.2f TFLOP/s" % out["roofs"]["cube4096_TFLOPs"], flush=True)
    print("%-14s %-22s %7s %9s %9s %8s" % ("arm", "shape", "tiles", "us", "ladder us", "ratio"),
          flush=True)

    for arm, shapes in SHAPES.items():
        for sh in shapes:
            x = tiles(sh)
            xs = t(sh)
            if arm == "layer_norm":
                fn = lambda xs=xs: ttnn.layer_norm(xs, epsilon=1e-5, compute_kernel_config=kc)
                ip = False
            elif arm == "layer_norm_w":
                d = sh[-1]
                w, b = t((1, 1, d // 32, 32)), t((1, 1, d // 32, 32))
                w = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
                b = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
                fn = lambda xs=xs, w=w, b=b: ttnn.layer_norm(
                    xs, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=kc)
                ip = False
            elif arm == "add":
                ys = t(sh)
                fn = lambda xs=xs, ys=ys: ttnn.add(xs, ys)
                ip = False
            else:
                ys = t(sh)
                fn = lambda xs=xs, ys=ys: ttnn.multiply_(xs, ys)
                ip = True
            try:
                us = trace_point(dev, fn, a.reps, a.blocks, ip)
            except Exception as e:                                            # noqa: BLE001
                out["rows"].append({"arm": arm, "shape": list(sh), "x_tiles": x,
                                    "error": "%s: %s" % (type(e).__name__,
                                                         str(e).splitlines()[0][:140])})
                print("%-14s %-22s %7d   REFUSED" % (arm, "x".join(map(str, sh)), x), flush=True)
                continue
            lad = curve(arm, x)
            out["rows"].append({"arm": arm, "shape": list(sh), "x_tiles": x, "us": us,
                                "ladder_us": lad, "ratio": us / lad})
            print("%-14s %-22s %7d %9.2f %9.2f %8.3f"
                  % (arm, "x".join(map(str, sh)), x, us, lad, us / lad), flush=True)
            out["out_path"] = str(a.out)
            a.out.write_text(json.dumps(out, indent=1))

    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    out["roofs_after"] = roofs(dev, kc, 5)
    out["AA_cube_pct"] = 100 * abs(out["roofs_after"]["cube4096_TFLOPs"]
                                   - out["roofs"]["cube4096_TFLOPs"]) / out["roofs"]["cube4096_TFLOPs"]
    print("A/A cube %.2f %%" % out["AA_cube_pct"], flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
