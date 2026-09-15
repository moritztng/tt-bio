#!/usr/bin/env python3
"""The launch floor per (op class, the shape the fold actually launches it on).

`launch_sweep.py` fits one ladder per class on one aspect ratio, [1, R, 768], and `shape_control.py`
showed that is not enough: at a FIXED tile count the floor moves with the shape, and by a lot. A
`ttnn.layer_norm` over 560 tiles costs 12.07 us as [1,140,32,128] against the 17.12 us the [1,R,768]
ladder predicts, and 27.22 us as [1,512,1536] against 18.44. The governing variable is the row
WIDTH -- the reduction length -- and the class ladder holds it fixed at 768, so it prices every
narrow shape too high and every wide shape too low.

So this fits a ladder per shape instead. For each (class, shape) the 512 aa fold launches, it
rebuilds the op at that exact shape, scales the row axis down 8x / 4x / 2x / 1x with the width and
the head layout held at the fold's own, and takes the y-intercept. The row axis is the one that
moves with sequence length, so the intercept is "this op, this shape family, no rows" -- the
program launch and nothing else.

Same trace-replay reading as `launch_sweep.py`: device time, host removed.

The first session measured the 19 highest-call keys and died on `ttnn.reshape([1,768,512])`, which
wedges a Blackhole card (two independent runs, both killed by pid). `--resume` picks the table back
up; `reshape` is refused outright rather than retried, and `--max-tiles` refuses a ladder point
whose tensors are too big to be worth the host time. A refused key costs nothing: it falls back to
the class ladder, which is where it already was.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from launch_sweep import fit, roofs, tiles, trace_point                       # noqa: E402

STEPS = (8, 4, 2, 1)


def row_axis(shape):
    """The axis a longer sequence makes longer: the largest leading dim, else the row dim."""
    lead = [(shape[i], i) for i in range(len(shape) - 2) if shape[i] >= 8]
    if lead:
        return max(lead)[1]
    return len(shape) - 2


def ladder(shape):
    """[shape with the row axis at 1/8, 1/4, 1/2, 1/1], tile-aligned on the row dim."""
    ax = row_axis(shape)
    out = []
    for d in STEPS:
        s = list(shape)
        if ax == len(shape) - 2:
            t = max(1, ceil(shape[ax] / 32) // d)
            s[ax] = 32 * t
        else:
            s[ax] = max(1, shape[ax] // d)
        if tuple(s) not in [tuple(o) for o in out]:
            out.append(tuple(s))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=Path, default=HERE / "fold_shapes.json")
    ap.add_argument("--out", type=Path, default=HERE / "shape_ladder.json")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--max-tiles", type=int, default=131072,
                    help="refuse a ladder point above this output tile count (131072 tiles is "
                         "268 MB in bf16). The keys it refuses are the 16-call giants at the "
                         "bottom of the table; their smaller ladder points still measure.")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop starting new keys after this many seconds and close the table "
                         "properly, with the after-roofs and the after-loadavg. The first session "
                         "was killed instead and published no closing A/A.")
    ap.add_argument("--resume", action="store_true",
                    help="keep the fits already in --out and re-measure only what is missing. "
                         "This box hangs roughly hourly and a ladder row is independent of every "
                         "other row, so a partial table is still a usable table.")
    a = ap.parse_args()

    dev = T.get_device(trace_region_size=1 << 29)
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    want = json.loads(a.shapes.read_text())

    def t(shape, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), layout=layout,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def make(arm, sh, K):
        """(callable, in_place) for one class at one concrete shape."""
        if arm == "layer_norm":
            x = t(sh)
            return (lambda: ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=kc)), False
        if arm == "layer_norm_w":
            x = t(sh)
            d = sh[-1]
            w = ttnn.to_layout(t((1, 1, d // 32, 32)), ttnn.ROW_MAJOR_LAYOUT)
            b = ttnn.to_layout(t((1, 1, d // 32, 32)), ttnn.ROW_MAJOR_LAYOUT)
            return (lambda: ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-5,
                                            compute_kernel_config=kc)), False
        if arm in ("add", "multiply", "add_", "multiply_"):
            x, y = t(sh), t(sh)
            f = {"add": ttnn.add, "multiply": ttnn.multiply,
                 "add_": ttnn.add_, "multiply_": ttnn.multiply_}[arm]
            return (lambda: f(x, y)), arm.endswith("_")
        if arm in ("linear", "matmul"):
            x = t(tuple(sh[:-1]) + (K,))
            w = t((K, sh[-1]))
            f = ttnn.linear if arm == "linear" else ttnn.matmul
            return (lambda: f(x, w, compute_kernel_config=kc,
                              core_grid=T.CORE_GRID_MAIN)), False
        if arm == "permute":
            x = t(tuple(sh[:-2]) + (sh[-1], sh[-2]))
            p = tuple(range(len(sh) - 2)) + (len(sh) - 1, len(sh) - 2)
            return (lambda: ttnn.permute(x, p)), False
        if arm == "reshape":
            n = 1
            for d in sh:
                n *= d
            x = t((1, n // sh[-1], sh[-1]))
            return (lambda: ttnn.reshape(x, tuple(sh))), False
        if arm == "to_layout":
            x = t(sh)
            return (lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)), False
        if arm == "nlp_create_qkv_heads":
            b, h, s, d = sh
            x = t((b, 1, s, 3 * h * d))
            return (lambda: ttnn.experimental.nlp_create_qkv_heads(
                x, num_heads=h, num_kv_heads=h, transpose_k_heads=False)), False
        if arm == "nlp_concat_heads":
            b, _o, s, hd = sh
            x = t((b, 4, s, hd // 4))
            return (lambda: ttnn.experimental.nlp_concat_heads(x)), False
        if arm == "softmax":
            x = t(sh)
            return (lambda: ttnn.softmax(x, dim=-1, compute_kernel_config=kc)), False
        if arm == "cos":
            x = t(sh)
            return (lambda: ttnn.cos(x)), False
        if arm == "to_memory_config_l1":
            x = t(sh)
            return (lambda: ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)), False
        if arm == "transpose":
            x = t(tuple(sh[:-2]) + (sh[-1], sh[-2]))
            return (lambda: ttnn.transpose(x, -2, -1)), False
        if arm == "slice":
            # one tile more on the row axis, sliced straight back off. The cut is tile aligned on
            # purpose: a sub-tile LAST-axis slice of a large DRAM tensor is the known Blackhole
            # wedge, and this ladder is not the place to re-find it.
            big = list(sh)
            big[-2] = sh[-2] + 32
            x = t(tuple(big))
            return (lambda: ttnn.slice(x, tuple(0 for _ in sh), tuple(sh))), False
        if arm == "pad":
            small = list(sh)
            small[-2] = sh[-2] - 32
            if small[-2] < 32:
                raise ValueError("pad: row dim %d leaves no room to pad" % sh[-2])
            x = t(tuple(small))
            pads = tuple((0, 0) for _ in sh[:-2]) + ((0, 32), (0, 0))
            return (lambda: ttnn.pad(x, pads, 0.0)), False
        if arm == "concat":
            # two tile-aligned halves. Last axis where it splits into whole tiles, else the row
            # axis, so the two operands are never a sub-tile shape.
            if sh[-1] % 64 == 0:
                d, half = len(sh) - 1, sh[-1] // 2
            elif sh[-2] % 64 == 0:
                d, half = len(sh) - 2, sh[-2] // 2
            else:
                raise ValueError("concat: %s halves into sub-tile operands" % (sh,))
            s2 = list(sh)
            s2[d] = half
            x, y = t(tuple(s2)), t(tuple(s2))
            return (lambda: ttnn.concat([x, y], dim=d)), False
        if arm == "chunk":
            big = list(sh)
            big[-1] = 2 * sh[-1]
            x = t(tuple(big))
            return (lambda: ttnn.chunk(x, 2, dim=-1)), False
        if arm == "sdpa":
            if len(sh) != 4:
                raise ValueError("sdpa: %s is not (b, h, s, d)" % (sh,))
            q, k, v = t(sh), t(sh), t(sh)
            return (lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False)), False
        raise ValueError("no builder for " + arm)

    out = {"host": platform.node(), "arch": str(dev.arch()), "reps": a.reps, "blocks": a.blocks,
           "reading": "device only, trace replay", "steps": list(STEPS),
           "loadavg_before": open("/proc/loadavg").read().split()[:3],
           "fits": {}, "rows": {}, "refused": {}}
    if a.resume and a.out.exists():
        prev = json.loads(a.out.read_text())
        out["fits"], out["rows"] = prev.get("fits", {}), prev.get("rows", {})
        out["refused"] = prev.get("refused", {})
        out["sessions"] = prev.get("sessions", []) + [
            {"loadavg_before": out["loadavg_before"], "resumed_with": len(out["fits"])}]
        print("resume: %d fits already measured" % len(out["fits"]), flush=True)
    out["roofs"] = roofs(dev, kc, 5)
    print("cube4096 %.2f TFLOP/s" % out["roofs"]["cube4096_TFLOPs"], flush=True)
    print("%-22s %-24s %8s %8s %7s  %s" % ("arm", "shape", "FLOOR", "slope", "r2", "ladder"),
          flush=True)

    t_start = time.time()
    for e in want:
        arm, sh, K = e["arm"], tuple(e["shape"]), e.get("K")
        key = "%s|%s|K%s" % (arm, "x".join(map(str, sh)), K)
        if key in out["fits"]:
            continue
        if a.budget_s and time.time() - t_start > a.budget_s:
            out["stopped_on_budget_after_s"] = time.time() - t_start
            print("budget reached, closing the table", flush=True)
            break
        if arm == "reshape":
            # The builder above reshapes a tensor to the shape it already has, because it derives
            # the input from the output's own element count, and an IDENTITY reshape at [1,768,512]
            # is what hung a card twice and killed two sweeps. Refusing costs the table nothing:
            # `launch_sweep.py`'s class arm reshapes (1,16,48,512) -> (1,768,512), the fold's own
            # output shape, and `launch_trace_qb2c1.json` has it at 22.81 us with no hang, so these
            # 5,080 calls already fall back to a ladder built on their own shape family.
            out["refused"][key] = "skipped: ttnn.reshape wedges this card (2 sightings)"
            print("%-22s %-24s  SKIPPED (reshape wedge)" % (arm, "x".join(map(str, sh))),
                  flush=True)
            a.out.write_text(json.dumps(out, indent=1))
            continue
        pts = []
        for s2 in ladder(sh):
            if tiles(s2) > a.max_tiles:
                out["refused"]["%s@%s" % (key, "x".join(map(str, s2)))] = (
                    "skipped: %d tiles over --max-tiles %d" % (tiles(s2), a.max_tiles))
                continue
            try:
                fn, ip = make(arm, s2, K)
                pts.append({"shape": list(s2), "x_tiles": tiles(s2),
                            "us": trace_point(dev, fn, a.reps, a.blocks, ip)})
            except Exception as ex:                                           # noqa: BLE001
                out["refused"]["%s@%s" % (key, "x".join(map(str, s2)))] = "%s: %s" % (
                    type(ex).__name__, str(ex).splitlines()[0][:150])
        out["rows"][key] = pts
        if len(pts) >= 3:
            a0, b0, r2 = fit([q["x_tiles"] for q in pts], [q["us"] for q in pts])
            lo = min(q["us"] for q in pts)
            floor_us = min(a0 if a0 > 0 else float("inf"), lo)
            out["fits"][key] = {"arm": arm, "shape": list(sh), "calls": e.get("calls"),
                                "fixed_us": a0, "slope_us_per_tile": b0, "r2": r2, "n": len(pts),
                                "min_measured_us": lo, "launch_floor_us": floor_us,
                                "floor_from": "fit intercept" if floor_us == a0
                                              else "measured min"}
            print("%-22s %-24s %8.2f %8.4f %7.4f  %s"
                  % (arm, "x".join(map(str, sh)), floor_us, b0, r2,
                     " ".join("%.1f" % q["us"] for q in pts)), flush=True)
        else:
            print("%-22s %-24s  REFUSED" % (arm, "x".join(map(str, sh))), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    out["loadavg_after"] = open("/proc/loadavg").read().split()[:3]
    out["roofs_after"] = roofs(dev, kc, 5)
    out["AA_cube_pct"] = 100 * abs(out["roofs_after"]["cube4096_TFLOPs"]
                                   - out["roofs"]["cube4096_TFLOPs"]) \
        / out["roofs"]["cube4096_TFLOPs"]
    print("A/A cube %.2f %%   %d fits, %d refused"
          % (out["AA_cube_pct"], len(out["fits"]), len(out["refused"])), flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
