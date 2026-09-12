#!/usr/bin/env python3
"""Price the atom-path levers on one settled diffusion step, interleaved, profiler off.

Grabs a real `Diffusion.__call__` out of a 512 aa fold with `b2z2-step-program-fusion`'s probe,
then replays it under each arm in one process, alternating arms inside every block so a drift in
the box lands on all arms equally. The profiler costs 3.81x on this block and all of it in the
gaps, so the step is timed bare.

Arms are module globals, not env vars: the flags are read at import.

    base      shipped
    akw       TT_BIO_ATOM_KEY_WINDOW          the gather instead of the one-hot matmul
    akwpre    + TT_BIO_ATOM_KV_PREPROJ        project on the atom axis, gather the projection

`--parity` additionally runs each arm once and compares the step OUTPUT with torch.equal, with a
negative control that must differ.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2z2_step_fusion"))

ARMS = {"base": (False, False), "akw": (True, False), "akwpre": (True, True)}


def set_arm(T, arm):
    T._ATOM_KEY_WINDOW, T._ATOM_KV_PREPROJ = ARMS[arm]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="base,akw,akwpre")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--parity", action="store_true")
    a = ap.parse_args()
    arms = a.arms.split(",")
    for arm in arms:
        assert arm in ARMS, arm

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import step_probe as SP                    # inserts the paths tt_baseline is found on
    import tt_baseline as B

    SP.OUT_PATH = a.out.with_suffix(".grab.json")
    SP.OUT["env"] = {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "arch": T.arch_name(), "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                   "arms": arms, "reps": a.reps, "blocks": a.blocks,
                   "loadavg_start": open("/proc/loadavg").read().split()[:3]},
           "rows": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    out["env"]["grid"] = str(T.CORE_GRID_MAIN)

    if a.parity:
        ref = {}
        for arm in arms:
            set_arm(T, arm)
            ref[arm] = ttnn.to_torch(g["obj"](*g["args"], **g["kwargs"]))
        par = {}
        for arm in arms[1:]:
            par[f"{arms[0]}_vs_{arm}"] = {
                "bit_exact": bool(torch.equal(ref[arms[0]], ref[arm])),
                "max_abs": float((ref[arms[0]].float() - ref[arm].float()).abs().max())}
        if "akw" in ref and "akwpre" in ref:
            par["akw_vs_akwpre"] = {
                "bit_exact": bool(torch.equal(ref["akw"], ref["akwpre"])),
                "max_abs": float((ref["akw"].float() - ref["akwpre"].float()).abs().max())}
        # negative control: the same arm twice must agree, a perturbed input must not
        set_arm(T, arms[-1])
        again = ttnn.to_torch(g["obj"](*g["args"], **g["kwargs"]))
        par["self_repeat_bit_exact"] = bool(torch.equal(ref[arms[-1]], again))
        par["negative_control_differs"] = bool(not torch.equal(ref[arms[-1]], ref[arms[0]] + 1.0))
        out["parity"] = par
        print("PARITY", json.dumps(par, indent=1), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    fence = SP.make_fence(ttnn, dev)
    for arm in arms:                                    # warm every program cache first
        set_arm(T, arm)
        for _ in range(3):
            g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)

    for blk in range(a.blocks):
        for arm in arms:
            set_arm(T, arm)
            fence()
            t0 = time.perf_counter()
            for _ in range(a.reps):
                g["obj"](*g["args"], **g["kwargs"])
            ttnn.synchronize_device(dev)
            ms = 1e3 * (time.perf_counter() - t0) / a.reps
            out["rows"].append({"block": blk, "arm": arm, "ms": round(ms, 4),
                                "loadavg": open("/proc/loadavg").read().split()[0]})
            print(f"  blk{blk} {arm:8s} {ms:8.4f} ms", flush=True)
            a.out.write_text(json.dumps(out, indent=1))

    med = {arm: round(st.median([r["ms"] for r in out["rows"] if r["arm"] == arm]), 4)
           for arm in arms}
    out["median_ms"] = med
    out["spread_pct"] = {arm: round(100 * (max(v) - min(v)) / st.median(v), 3)
                         for arm in arms
                         for v in [[r["ms"] for r in out["rows"] if r["arm"] == arm]]}
    out["ratio_vs_first"] = {arm: round(med[arms[0]] / med[arm], 5) for arm in arms}
    if "akw" in med and "akwpre" in med:
        out["ratio_akwpre_vs_akw"] = round(med["akw"] / med["akwpre"], 5)
    out["env"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.write_text(json.dumps(out, indent=1))
    print("MEDIAN", json.dumps(med), flush=True)
    print("RATIO", json.dumps(out["ratio_vs_first"]), flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
