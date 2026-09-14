#!/usr/bin/env python3
"""Paired host-residual attribution: the same region tree, both arms, interleaved in one process.

A single attribution fold gives a decomposition, not a comparison. whglx runs eight workers, and
between this row's baseline fold and its post-port fold the box went from loadavg 8 to 24:
`weighted_rigid_align` read 0.939 ms/call in the first and 10.792 in the second, an 11x move that
has nothing to do with the change. So the before/after host residual has to be taken the same way
the fold ratio is -- ABAB in one process, one device open -- and reported as a paired difference.

The tree, the instrument and its caveats are `perf/b2x_host_residual/host_residual.py`'s
(`time.perf_counter` wall, no `synchronize_device`, no `thread_time`); this reuses them rather
than restating them.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(REPO / "perf" / "b2x_host_residual"))

DEVICE_BRACKETS = ("predict_step/trunk", "predict_step/sampler/denoiser/denoise_device",
                   "predict_step/confidence/pairformer_conf")
FLAG = "TT_BIO_DEVICE_CONDITIONING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=3)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import host_residual as H
    import tt_bio.tenstorrent as T
    import tt_bio.boltz2 as boltz2
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    assert FLAG not in os.environ, f"{FLAG} is set in the environment; this script owns it"
    fix = REPO / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", Path(__file__).resolve().parent / f".msa_{a.size}",
        fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m")
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
                   "size": a.size, "pairs": a.pairs,
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "instrument": "host_residual.Regions, wall, no synchronize_device"},
           "folds": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    print("=== cold fold (discarded) ===", flush=True)
    one_fold()

    for i in range(a.pairs):
        for arm, val in (("on", "1"), ("off", "0")):
            os.environ[FLAG] = val
            reg = H.Regions()
            H.install_regions(reg, state, T, boltz2)
            t0 = time.perf_counter()
            one_fold()
            wall = time.perf_counter() - t0
            reg.remove()
            tree = reg.table()
            device = sum(tree[k]["incl_s"] for k in DEVICE_BRACKETS if k in tree)
            named = sum(v["incl_s"] for p, v in tree.items() if "/" not in p)
            out["folds"].append({
                "i": i, "arm": arm, "wall_s": round(wall, 4),
                "device_bracket_s": round(device, 4),
                "host_residual_s": round(wall - device, 4),
                "unattributed_s": round(wall - named, 4),
                "loadavg": H.loadavg(), "tree": tree})
            print(f"  {arm:3s}[{i}] fold {wall:7.3f}  device {device:7.3f}  "
                  f"host {wall - device:6.3f}  unattr {wall - named:6.3f}  "
                  f"load {H.loadavg()[0]}", flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    os.environ.pop(FLAG, None)

    def med(arm, key):
        return round(st.median([f[key] for f in out["folds"] if f["arm"] == arm]), 4)

    paired = [b["host_residual_s"] - c["host_residual_s"]
              for b, c in zip(out["folds"][1::2], out["folds"][0::2])]
    out["summary"] = {
        "host_residual_on_s": med("on", "host_residual_s"),
        "host_residual_off_s": med("off", "host_residual_s"),
        "paired_host_delta_s": [round(x, 4) for x in paired],
        "paired_host_delta_mean_s": round(st.mean(paired), 4),
        "fold_on_s": med("on", "wall_s"), "fold_off_s": med("off", "wall_s"),
        "fold_ratio": round(med("off", "wall_s") / med("on", "wall_s"), 5)}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1), flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
