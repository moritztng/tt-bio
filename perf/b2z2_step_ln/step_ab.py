#!/usr/bin/env python3
"""Interleaved A/B of the shared `layer_norm(s)` on one settled diffusion step.

The lever is a module global (`tt_bio.tenstorrent._B2_ADALN_SHARED_SNORM`), so both arms live in
ONE process and one grabbed `Diffusion.__call__`: block n of the base is measured between block n
of the arm and block n+1 of it, which is the only way to read a 4 % effect on a shared box where
`_L1_OUT_RUNG` and the other rows' load both drift under a batched measurement.

  --mode time     interleaved base/arm blocks, each `--reps` replays, plus the base's own A/A
                  floor taken from the spread of its own blocks.
  --mode parity   one replay per arm, `torch.equal` and the max abs difference of the step's
                  output, plus a negative control that must differ.
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
FUSION = ROOT / "perf" / "b2z2_step_fusion"
sys.path.insert(0, str(FUSION))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

FLAG = "_B2_ADALN_SHARED_SNORM"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("time", "parity"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    a.out.parent.mkdir(parents=True, exist_ok=True)
    SP.OUT_PATH = a.out
    SP.OUT["env"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "mode": a.mode, "size": a.size,
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "flags": {k: v for k, v in sorted(os.environ.items())
                  if k.startswith(("TT_BIO_", "BOLTZ2_", "B2_"))},
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    SP.dump()

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    call = lambda: g["obj"](*g["args"], **g["kwargs"])          # noqa: E731
    fence = SP.make_fence(ttnn, dev)

    def with_flag(v, fn):
        old = getattr(T, FLAG)
        setattr(T, FLAG, v)
        try:
            return fn()
        finally:
            setattr(T, FLAG, old)

    if a.mode == "parity":
        outs = {}
        for label, v in (("base", False), ("arm", True)):
            o = with_flag(v, call)
            ttnn.synchronize_device(dev)
            outs[label] = ttnn.to_torch(o).float()
        d = (outs["arm"] - outs["base"])
        ref = outs["base"]
        SP.OUT["parity"] = {
            "shape": list(ref.shape),
            "equal": bool(torch.equal(outs["arm"], outs["base"])),
            "max_abs": float(d.abs().max()),
            "rms": float(d.pow(2).mean().sqrt()),
            "ref_rms": float(ref.pow(2).mean().sqrt()),
            "rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "pcc": float(torch.corrcoef(torch.stack([ref.flatten(), outs["arm"].flatten()]))[0, 1]),
        }
        # negative control: two base replays must be identical, or the comparison means nothing
        again = with_flag(False, call)
        ttnn.synchronize_device(dev)
        SP.OUT["parity"]["base_repeat_equal"] = bool(
            torch.equal(ttnn.to_torch(again).float(), outs["base"]))
        print(json.dumps(SP.OUT["parity"], indent=1), flush=True)
        SP.dump()
        print("DONE", a.out, flush=True)
        return 0

    for v in (False, True, False, True):                       # compile + warm both arms
        with_flag(v, call)
    ttnn.synchronize_device(dev)

    seq, walls = [], {"base": [], "arm": []}
    for _ in range(a.blocks):
        for label, v in (("base", False), ("arm", True)):
            fence()
            t0 = time.perf_counter()
            with_flag(v, lambda: [call() for _ in range(a.reps)])
            ttnn.synchronize_device(dev)
            w = 1e3 * (time.perf_counter() - t0) / a.reps
            walls[label].append(round(w, 4))
            seq.append([label, round(w, 4)])
            print(f"  {label:5s} {w:8.4f} ms/step", flush=True)
            SP.OUT["seq"] = seq
            SP.dump()

    med = {k: st.median(v) for k, v in walls.items()}
    spread = {k: (max(v) - min(v)) / st.median(v) for k, v in walls.items()}
    SP.OUT["step"] = {
        "ms": {k: round(v, 4) for k, v in med.items()},
        "all": walls,
        "spread_pct": {k: round(100 * v, 3) for k, v in spread.items()},
        "ratio": round(med["base"] / med["arm"], 5),
        "aa_floor_base": round(max(walls["base"]) / min(walls["base"]), 5),
        "aa_floor_arm": round(max(walls["arm"]) / min(walls["arm"]), 5),
        "reps": a.reps, "blocks": a.blocks,
    }
    SP.dump()
    print(f"  base {med['base']:.4f}  arm {med['arm']:.4f}  RATIO {med['base']/med['arm']:.5f}"
          f"  A/A base {SP.OUT['step']['aa_floor_base']:.5f}", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
