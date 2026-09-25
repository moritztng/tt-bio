#!/usr/bin/env python3
"""bcx-bytes: can a binary op take its operands' dtypes as they come, bit for bit?

Every fp32 island on the AF2 tape is written as typecast-then-op: `add_grad` widens both
gradients before summing them, `_residual` widens both addends, the softmax tail widens the
scores before the bias add. Each widening reads bf16 and writes fp32, 6 B per element, only to
feed an op that reads the value again. If `ttnn.add` with bf16 (or mixed) operands and an fp32
output produces the same bits as the typecast form, the typecast is pure traffic.

Each arm is compared with its typecast form bit for bit and with a float64 reference, over
shapes the real block runs (one Evoformer pair tensor, one MSA-track tensor, the attention
scores) and over inputs built to hit rounding ties. Device seconds per call are the median of
synced reps, one process, AICLK sampled from the card's sysfs node during them.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_bytes"


def main():
    import ttnn
    from perf.bcx_stack.stack import Clock
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock()
    f32, bf = ttnn.float32, ttnn.bfloat16
    TL = ttnn.TILE_LAYOUT

    def up(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=TL, device=dev)

    def down(t):
        return ttnn.to_torch(t).float()

    def timed(fn, reps=15):
        fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(out)
        return float(np.median(ts)), (t0, time.perf_counter())

    torch.manual_seed(0)
    blob = {"cases": []}
    spans = []
    for shape in ([1, 256, 256, 128], [1, 256, 256, 64], [81, 4, 256, 256]):
        for kind in ("randn", "ties"):
            a = torch.randn(shape).bfloat16()
            if kind == "ties":
                # b = a * 2^-8 puts the exact sum halfway between two bf16 neighbours often
                b = (a.float() * 2.0 ** -8).bfloat16()
            else:
                b = (torch.randn(shape) * 0.3).bfloat16()
            c = torch.randn(shape) * 0.01                      # an fp32 accumulator
            A, B, C = up(a, bf), up(b, bf), up(c, f32)
            f64 = {"a+b": a.double() + b.double(), "c+b": c.double() + b.double()}
            arms = {
                # fan-in, first contribution meets second: both widened, summed in fp32
                "a+b f32 [ref]": lambda: ttnn.add(ttnn.typecast(A, f32), ttnn.typecast(B, f32)),
                "a+b f32 [mixed]": lambda: ttnn.add(A, B, dtype=f32),
                # fan-in, fp32 accumulator meets a bf16 contribution
                "c+b f32 [ref]": lambda: ttnn.add(C, ttnn.typecast(B, f32)),
                "c+b f32 [mixed]": lambda: ttnn.add(C, B),
                "c+b f32 [mixed dtype]": lambda: ttnn.add(C, B, dtype=f32),
                # `_residual`: widen both, add, narrow once
                "a+b bf16 [ref]": lambda: ttnn.typecast(
                    ttnn.add(ttnn.typecast(A, f32), ttnn.typecast(B, f32)), bf),
                "a+b bf16 [mixed]": lambda: ttnn.typecast(ttnn.add(A, B, dtype=f32), bf),
                "a+b bf16 [plain]": lambda: ttnn.add(A, B),
                # accumulator narrowed at the closure: add then narrow, or narrow in the add
                "c+b bf16 [ref]": lambda: ttnn.typecast(ttnn.add(C, ttnn.typecast(B, f32)), bf),
                "c+b bf16 [mixed]": lambda: ttnn.add(C, B, dtype=bf),
            }
            res = {}
            for name, fn in arms.items():
                try:
                    out = fn()
                    ttnn.synchronize_device(dev)
                    res[name] = (down(out), str(out.dtype))
                    ttnn.deallocate(out)
                except Exception as e:                        # an arm the op refuses
                    res[name] = (None, f"refused: {str(e).splitlines()[0][:160]}")
            for name, fn in arms.items():
                v, dt = res[name]
                case = {"shape": shape, "kind": kind, "arm": name, "out_dtype": dt}
                if v is not None:
                    ref = res[name.split(" [")[0] + " [ref]"][0]
                    truth = f64[name.split(" ")[0]]
                    case["identical_to_ref"] = bool(torch.equal(v, ref))
                    case["mismatch_frac"] = float((v != ref).float().mean())
                    case["rel_l2_vs_f64"] = float((v.double() - truth).norm() / truth.norm())
                    case["ms"], sp = timed(fn)
                    case["ms"] *= 1e3
                    spans.append(sp)
                blob["cases"].append(case)
                print(json.dumps(case), flush=True)
            for t in (A, B, C):
                ttnn.deallocate(t)
    blob["aiclk"] = clock.window(spans)
    clock.stop()
    print("aiclk", blob["aiclk"])
    (OUT / "dtype_probe.json").write_text(json.dumps(blob, indent=1))


if __name__ == "__main__":
    main()
