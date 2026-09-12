#!/usr/bin/env python3
"""The padded-width tail, measured on the diffusion step it lives in, and proved bit-exact.

`TT_BIO_HEAD_PAD_TAIL` replaces the four programs that strip the SDPA output's zero pad lanes
(slice, permute, a reshape that merges a 48-row axis, permute) with one `nlp_concat_heads`, and
lets the gate projection and the output projection carry the pad lanes instead. The pad lanes are
exactly zero, so it is an identity -- which is the whole claim, and `--mode parity` is what proves
it.

The step grab, the fence and the timing block are `perf/b2z2_step_fusion/step_probe.py`'s, imported
rather than copied: this row's instrument is that row's instrument plus an arm switch. Because the
lever builds its padded weights lazily and keeps the unpadded ones, ONE process holds both arms and
the A/B is interleaved rather than batched, as the campaign's rule 3 requires.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSION = ROOT / "perf" / "b2z2_step_fusion" / "step_probe.py"

_spec = importlib.util.spec_from_file_location("_step_probe", FUSION)
SP = importlib.util.module_from_spec(_spec)
sys.modules["_step_probe"] = SP
_spec.loader.exec_module(SP)

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def set_arm(T, on: bool):
    T._HEAD_PAD_TAIL = bool(on)


def eligible(T):
    """How many AttentionPairBias sites the property gate actually accepts, and at what width."""
    import gc
    seen, widths = 0, {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, T.AttentionPairBias) and getattr(obj, "pad_tail", False):
                seen += 1
                widths[(obj.n_heads, obj.head_dim, obj.padded_head_dim)] = \
                    widths.get((obj.n_heads, obj.head_dim, obj.padded_head_dim), 0) + 1
        except ReferenceError:
            continue
    return seen, {str(k): v for k, v in widths.items()}


def mode_parity(ttnn, torch, T, dev, g):
    call = lambda: g["obj"](*g["args"], **g["kwargs"])           # noqa: E731

    def run(on):
        set_arm(T, on)
        out = call()
        ttnn.synchronize_device(dev)
        t = ttnn.to_torch(out) if isinstance(out, ttnn.Tensor) else out
        return t.clone()

    base = run(False)
    base2 = run(False)
    arm = run(True)
    OUT["parity"] = {
        "aa_base_deterministic": bool(torch.equal(base, base2)),
        "bit_exact": bool(torch.equal(base, arm)),
        "max_abs": float((base.float() - arm.float()).abs().max()),
        "shape": list(base.shape),
    }
    dump()

    # Negative control: break one REAL row of the padded output projection. If the comparison above
    # is reading the tail at all, this must move the answer.
    import gc
    victim = None
    for obj in gc.get_objects():
        try:
            if isinstance(obj, T.AttentionPairBias) and getattr(obj, "_pad_tail_weights", None):
                victim = obj
                break
        except ReferenceError:
            continue
    if victim is not None:
        gw, ow = victim._pad_tail_weights
        broken = ttnn.to_torch(ow)
        broken[0, :] = 0.0                      # a real lane, not a pad lane
        victim._pad_tail_weights = (gw, ttnn.from_torch(
            broken, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ow.dtype))
        ctrl = run(True)
        OUT["parity"]["negative_control_differs"] = not bool(torch.equal(base, ctrl))
        OUT["parity"]["negative_control_max_abs"] = float((base.float() - ctrl.float()).abs().max())
        victim._pad_tail_weights = (gw, ow)
    set_arm(T, False)
    dump()
    print(f"  parity {OUT['parity']}", flush=True)


def mode_ab(ttnn, T, dev, g, reps, blocks):
    fence = SP.make_fence(ttnn, dev)
    call = lambda: g["obj"](*g["args"], **g["kwargs"])           # noqa: E731
    for on in (False, True):                                     # warm BOTH arms, build the weights
        set_arm(T, on)
        for _ in range(3):
            call()
    ttnn.synchronize_device(dev)
    fence()

    walls: dict = {"off": [], "on": []}
    order = []
    for b in range(blocks):
        for on in (False, True) if b % 2 == 0 else (True, False):   # alternate lead, no position bias
            set_arm(T, on)
            t0 = time.perf_counter()
            for _ in range(reps):
                call()
            ttnn.synchronize_device(dev)
            w = (time.perf_counter() - t0) / reps
            walls["on" if on else "off"].append(1e3 * w)
            order.append(("on" if on else "off", round(1e3 * w, 4)))
        fence()
    set_arm(T, False)

    med = {k: st.median(v) for k, v in walls.items()}
    spread = {k: (max(v) - min(v)) / st.median(v) for k, v in walls.items()}
    # A/A floor: the base arm against itself, first half vs second half of its own blocks.
    off = walls["off"]
    half = len(off) // 2
    aa = st.median(off[:half]) / st.median(off[half:]) if half else float("nan")
    OUT["ab"] = {
        "ms_off": round(med["off"], 4), "ms_on": round(med["on"], 4),
        "ratio": round(med["off"] / med["on"], 5),
        "delta_ms_per_step": round(med["off"] - med["on"], 4),
        "aa_floor": round(aa, 5),
        "spread_pct": {k: round(100 * v, 3) for k, v in spread.items()},
        "walls": {k: [round(x, 4) for x in v] for k, v in walls.items()},
        "order": order, "reps": reps, "blocks": blocks,
    }
    dump()
    print(f"  off {med['off']:.4f} ms  on {med['on']:.4f} ms  ratio {med['off']/med['on']:.5f}x  "
          f"A/A {aa:.5f}", flush=True)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("parity", "ab", "both"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SP.OUT = OUT
    SP.OUT_PATH = OUT_PATH

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "mode": a.mode, "size": a.size,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "flags": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith("TT_BIO_") or k.startswith("B2_")},
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    set_arm(T, False)
    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)
    n, widths = eligible(T)
    OUT["eligible_sites"] = {"n": n, "widths": widths}
    print(f"  eligible AttentionPairBias sites: {n} {widths}", flush=True)
    dump()

    if a.mode in ("parity", "both"):
        mode_parity(ttnn, torch, T, dev, g)
    if a.mode in ("ab", "both"):
        mode_ab(ttnn, T, dev, g, a.reps, a.blocks)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
