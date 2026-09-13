#!/usr/bin/env python3
"""B1 op-level equivalence: the projection column permutation moves no value.

The lever reorders the fused in-projection's four quarters from `[g_a, g_b, p_a, p_b]` to
`[p_a, g_a, p_b, g_b]` and moves the gated channel move's slice offsets with it. Nothing it does
is arithmetic, so the bar is `torch.equal`, not a tolerance.

Both arms run on device through the SAME kernel; only the column order of `xw` and the two
runtime offsets differ. The reference is therefore the shipped layout's own output, not a host
approximation of it.

The rig carries its own negative control: a third arm reads the split layout with the OLD offsets,
which is a genuine per-channel mix-up, and the run fails if that arm comes back equal.

It also times the two arms interleaved (A B B A) so the op ratio the fold win is predicted from is
measured here rather than inherited.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio as _TB
assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
    f"imported tt_bio from {_TB.__file__}, not this tree")
import tt_bio.reblock_permute as RB
from tt_bio.tenstorrent import get_device

MAJOR = ("g_a", "g_b", "p_a", "p_b")
SPLIT = ("p_a", "g_a", "p_b", "g_b")


def lay(quarters: dict, order) -> torch.Tensor:
    return torch.cat([quarters[r] for r in order], dim=-1)


def run(dev, xw_h, p_off, g_off, C, reps):
    xw = ttnn.from_torch(xw_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = RB.reblock_permute_gated(xw, p_off, g_off, C, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(out)
    ttnn.deallocate(out)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = RB.reblock_permute_gated(xw, p_off, g_off, C, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(dev)
        ts.append(1e6 * (time.perf_counter() - t0))
        ttnn.deallocate(o)
    ttnn.deallocate(xw)
    return got, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="298,320,512,640")
    ap.add_argument("--cs", default="32,128")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = get_device()
    torch.manual_seed(0)
    res = {"banks": 8, "cells": [], "reps": a.reps,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    bad = 0
    for N in [int(v) for v in a.ns.split(",")]:
        for C in [int(v) for v in a.cs.split(",")]:
            q = {r: torch.randn(1, N, N, C, dtype=torch.bfloat16) for r in MAJOR}
            cell = {"N": N, "C": C}
            try:
                maj, spl = lay(q, MAJOR), lay(q, SPLIT)
                om = {r: MAJOR.index(r) * C for r in MAJOR}
                os_ = {r: SPLIT.index(r) * C for r in MAJOR}
                cell["off_major"] = {r: om[r] // 32 for r in MAJOR}
                cell["off_split"] = {r: os_[r] // 32 for r in MAJOR}
                # A B B A over the two calls the trimul makes, so box drift cancels.
                ref_a, t_ma = run(dev, maj, om["p_a"], om["g_a"], C, a.reps)
                got_a, t_sa = run(dev, spl, os_["p_a"], os_["g_a"], C, a.reps)
                got_b, t_sb = run(dev, spl, os_["p_b"], os_["g_b"], C, a.reps)
                ref_b, t_mb = run(dev, maj, om["p_b"], om["g_b"], C, a.reps)
                cell["equal_a"] = bool(torch.equal(got_a, ref_a))
                cell["equal_b"] = bool(torch.equal(got_b, ref_b))
                # negative control: the split layout read with the shipped layout's offsets
                ctl, _ = run(dev, spl, om["p_a"], om["g_a"], C, 0)
                cell["control_differs"] = not bool(torch.equal(ctl, ref_a))
                cell["us_major"] = round(st.median(t_ma + t_mb), 1)
                cell["us_split"] = round(st.median(t_sa + t_sb), 1)
                cell["op_ratio"] = round(cell["us_major"] / cell["us_split"], 4)
                ok = cell["equal_a"] and cell["equal_b"] and cell["control_differs"]
            except Exception as e:                                            # noqa: BLE001
                cell["error"] = f"{type(e).__name__}: {e}"[:300]
                ok = False
            bad += not ok
            res["cells"].append(cell)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  N={N:4d} C={C:3d} equal={cell.get('equal_a')}/{cell.get('equal_b')} "
                  f"ctl={cell.get('control_differs')} "
                  f"{cell.get('us_major', 0):8.1f} -> {cell.get('us_split', 0):8.1f} us "
                  f"{cell.get('op_ratio', 0):.4f}x {cell.get('error', '')}", flush=True)
    res["all_ok"] = bad == 0
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n{len(res['cells']) - bad}/{len(res['cells'])} cells bit-exact with a live control")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
