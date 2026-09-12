#!/usr/bin/env python3
"""Interleaved A/B of any set of step levers on one settled diffusion step.

Every lever that lives inside `Diffusion.__call__` is a module global of `tt_bio.tenstorrent`, so
all the arms live in ONE process and one grabbed call: block n of the base is measured between
block n of an arm and block n+1 of it, which is the only way to read a 2-4 % effect on a shared box
where `_L1_OUT_RUNG` and the other rows' load both drift under a batched measurement.

This started as the shared-`layer_norm(s)` harness and is now the harness for all of them, because
the fold-level A/B cannot see a step lever at all: on a contended qb2 the fold's A/A floor is
1.078x and the sampler is 27 % of the fold, so a 1.03x step lever is a 1.008x fold lever and
disappears. `perf/b2z2_layout_tail/tail_ab.py` measures the same quantity for one other flag; this
file measures any of them, and the union of them, in one process against one base.

    step_ab.py --mode time --arms base,LN,LAY,LNLAY --out <json>

  --mode time     interleaved arm blocks, each `--reps` replays, plus the base's own A/A floor
                  taken from the spread of its own blocks.
  --mode parity   one replay per arm, `torch.equal` and the max abs difference of the step's
                  output, plus a negative control that must differ.

Only levers inside the step belong here. `TT_BIO_DEVICE_CONDITIONING` and `TT_BIO_UNFUSED_SILU`
act outside `Diffusion.__call__` — on the host residual and in the Pairformer block — so they
would read as a null here and that null would mean nothing.
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

# arm -> the `tt_bio.tenstorrent` module globals it turns ON. Every arm sets EVERY flag, on for
# itself and off for the rest, so an arm can never inherit what the previous one left behind.
FLAGS = {"LN": "_B2_ADALN_SHARED_SNORM", "LAY": "_HEAD_PAD_TAIL", "AKW": "_ATOM_KEY_WINDOW"}
ARMS = {"base": (), "LN": ("LN",), "LAY": ("LAY",), "AKW": ("AKW",),
        "LNLAY": ("LN", "LAY"), "ALL": ("LN", "LAY", "AKW")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("time", "parity"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--arms", default="base,LN,LAY,LNLAY",
                    help="first must be base; every arm is timed against it")
    a = ap.parse_args()
    arms = a.arms.split(",")
    assert arms[0] == "base", "the first arm is the baseline and must be base"
    for arm in arms:
        assert arm in ARMS, f"unknown arm {arm}"

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

    def with_arm(arm, fn):
        on = set(ARMS[arm])
        old = {}
        for flag, name in FLAGS.items():
            assert hasattr(T, name), f"{name} missing from this checkout; arm {arm} would be a lie"
            old[name] = getattr(T, name)
            setattr(T, name, flag in on)
        try:
            return fn()
        finally:
            for name, v in old.items():
                setattr(T, name, v)

    if a.mode == "parity":
        outs = {}
        for arm in arms:
            o = with_arm(arm, call)
            ttnn.synchronize_device(dev)
            outs[arm] = ttnn.to_torch(o).float()
        ref = outs["base"]
        SP.OUT["parity"] = {"shape": list(ref.shape), "ref_rms": float(ref.pow(2).mean().sqrt()),
                            "arms": {}}
        for arm in arms[1:]:
            d = outs[arm] - ref
            SP.OUT["parity"]["arms"][arm] = {
                "equal": bool(torch.equal(outs[arm], ref)),
                "max_abs": float(d.abs().max()),
                "rms": float(d.pow(2).mean().sqrt()),
                "rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                "pcc": float(torch.corrcoef(
                    torch.stack([ref.flatten(), outs[arm].flatten()]))[0, 1])}
        # negative control: two base replays must be identical, or the comparison means nothing
        again = with_arm("base", call)
        ttnn.synchronize_device(dev)
        SP.OUT["parity"]["base_repeat_equal"] = bool(
            torch.equal(ttnn.to_torch(again).float(), ref))
        print(json.dumps(SP.OUT["parity"], indent=1), flush=True)
        SP.dump()
        print("DONE", a.out, flush=True)
        return 0

    for arm in arms + arms:                                    # compile + warm every arm twice
        with_arm(arm, call)
    ttnn.synchronize_device(dev)

    seq, walls = [], {arm: [] for arm in arms}
    for _ in range(a.blocks):
        for arm in arms:
            fence()
            t0 = time.perf_counter()
            with_arm(arm, lambda: [call() for _ in range(a.reps)])
            ttnn.synchronize_device(dev)
            w = 1e3 * (time.perf_counter() - t0) / a.reps
            walls[arm].append(round(w, 4))
            seq.append([arm, round(w, 4), round(os.getloadavg()[0], 2)])
            print(f"  {arm:6s} {w:8.4f} ms/step   load {os.getloadavg()[0]:5.2f}", flush=True)
            SP.OUT["seq"] = seq
            SP.dump()

    med = {k: st.median(v) for k, v in walls.items()}
    SP.OUT["step"] = {
        "ms": {k: round(v, 4) for k, v in med.items()},
        "all": walls,
        "spread_pct": {k: round(100 * (max(v) - min(v)) / st.median(v), 3)
                       for k, v in walls.items()},
        "ratio": {k: round(med["base"] / v, 5) for k, v in med.items()},
        # The floor: the base arm against itself, worst block over best. Any arm ratio inside it
        # is not a reading.
        "aa_floor_base": round(max(walls["base"]) / min(walls["base"]), 5),
        "reps": a.reps, "blocks": a.blocks, "arms": arms,
    }
    SP.dump()
    floor = SP.OUT["step"]["aa_floor_base"]
    print(f"\n  base {med['base']:.4f} ms/step   A/A floor {floor:.5f}", flush=True)
    for arm in arms[1:]:
        r = med["base"] / med[arm]
        print(f"  {arm:6s} {med[arm]:8.4f} ms  RATIO {r:.5f}  "
              f"{'ABOVE' if r > floor else 'inside'} the floor", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
