#!/usr/bin/env python3
"""bcx-p10-l1fuse leg 3/4: does the gate reach, and does it move a gradient.

Two questions in one process, arms interleaved `off, on, off` so the `off`/`off2` pair is the
A/A control and a difference between the two offs would disqualify the reading before the on
arm is looked at.

REACH FIRST, because a gate that serves zero has measured nothing and every accuracy number
beside it would be a null dressed as a pass (`a-fused-kernel-can-be-built-accurate-and-called-
zero-times-a-round`). The counters are read at the end of every arm and printed by name.

EQUALITY SECOND. Placement does not change arithmetic -- `bcx-p10-mmroof` measured max absolute
deviation 0 across nine classes moved between buffers -- so both input gradients must come back
`torch.equal` against the off arm. Anything that moves is this gate and not the residency, which
is why the bar here is equality and not a tolerance.

On qb2, never pc card 0: it silently miscomputes ttnn matmuls at a low, location-keyed rate
(memory `pc-card0-512aa-fold-nondeterminism`) and an equality check taken there asserts nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack import stack as S                # noqa: E402
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_p10_mmroof import cells as CE          # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_l1fuse" / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="equal_E.json")
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)

    armed = CE.arm(1, 1, 1, 1)
    print("ARMED %s" % json.dumps(armed), flush=True)

    from tt_bio import fanin_l1 as F
    lv, dev, ref = S.open_all(args)
    clock = S.Clock()

    spec = SH.CELLS["E"]
    n32 = spec["pad"]
    raw_m, raw_z, _, _ = S.inputs(ref, SH.N_HOST, args.seed, ragged=True)
    m0, z0 = SH.pad_like_fold(raw_m, raw_z, n32)
    m0 = m0.repeat(spec["depth"], 1, 1)
    torch.manual_seed(args.seed + 1)
    wm = torch.randn(m0.shape) / m0.numel() ** 0.5
    wz = torch.randn(z0.shape) / z0.numel() ** 0.5
    masks = SH.cell_masks(dev, n32, SH.N_REAL, spec["depth"], spec["masks"])
    SH.set_hifi(True)

    blob = {"stamp": S.stamp(args, clock), "cell": "E", "device_axis": n32, "armed": armed,
            "share": F.FANIN_L1_SHARE, "loadavg_start": os.getloadavg(), "arms": {}}

    for stack in args.stacks.split(","):               # warm both arms: JIT + program cache
        for on in (False, True):
            F.FANIN_L1 = on
            S.block_step(dev, lv, m0, z0, wm, wz, stack, k=1, ckpt=spec["ckpt"], masks=masks)
    F.REACH.clear()

    for stack in args.stacks.split(","):
        grads = {}
        for arm, on in (("off", False), ("on", True), ("off2", False)):
            F.FANIN_L1 = on
            F.REACH.clear()
            r, g = S.block_step(dev, lv, m0, z0, wm, wz, stack, k=1, ckpt=spec["ckpt"],
                                masks=masks)
            grads[arm] = [t.clone() for t in g]
            blob["arms"][f"{stack}|{arm}"] = {
                "flag": on, "reach": F.reach(),
                "wall_fwd": round(r["fwd"], 4), "wall_bwd": round(r["bwd"], 4),
                "aiclk": clock.window(r["spans"]), "loadavg": os.getloadavg()[0]}
            print(json.dumps({"stack": stack, "arm": arm, "reach": F.reach(),
                              "bwd": round(r["bwd"], 3),
                              "aiclk": blob["arms"][f"{stack}|{arm}"]["aiclk"]}), flush=True)

        def cmp(a, b):
            return [{"equal": bool(torch.equal(x, y)),
                     "max_abs": float((x.float() - y.float()).abs().max())}
                    for x, y in zip(grads[a], grads[b])]

        blob[f"{stack}|on_vs_off"] = cmp("on", "off")
        blob[f"{stack}|aa_control"] = cmp("off2", "off")
        print(json.dumps({"stack": stack, "on_vs_off": blob[f"{stack}|on_vs_off"],
                          "aa_control": blob[f"{stack}|aa_control"]}), flush=True)

    blob["loadavg_end"] = os.getloadavg()
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
