#!/usr/bin/env python3
"""Where bf16 runs out of EXPONENT in the denoiser backward, as a ladder rather than a point.

`LEDGER K5` / commit `b3fa222ef` established the mechanism and it is not the one this row's
DONE_CHECK probe was written for: on REAL captured conditioning every stage of the denoiser
backward returns non-finite gradients with norms around 1e39 against bf16's maximum 3.39e38,
while the same block on random inputs of order 0.5 is clean. So bf16 does not drift and it does
not stall -- it overflows outright, and the failure is exponent range, not mantissa.

A single point leaves the actionable question open: how much headroom does the fp32 default
actually buy, and how far from the cliff is a site that stays bf16? This sweeps the input scale
over the whole block, in fp32 where every stage stays finite, and reports the largest gradient
magnitude anywhere in the backward at each scale. bf16's overflow threshold is a horizontal
line across that curve, so the crossing is read off rather than guessed, and the ratio between
the crossing scale and the scale real conditioning presents is the headroom.

Run in fp32 deliberately: the point is to watch the magnitude the backward PRODUCES, and a bf16
arm cannot report a number it cannot represent.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dit

BF16_MAX = 3.3895313892515355e38      # (2 - 2^-7) * 2^127
FP32_MAX = 3.4028234663852886e38      # same exponent range; bf16 is fp32's exponent, fp32's is not the lever


def aiclk():
    root = Path("/sys/class/tenstorrent")
    return {p.name.split("!")[1]: int((p / "tt_aiclk").read_text().strip())
            for p in sorted(root.glob("tenstorrent!*"))}


def held_nodes():
    import glob, os
    return sorted({int(os.readlink(f).rsplit("/", 1)[1]) for f in glob.glob("/proc/self/fd/*")
                   if os.path.islink(f) and os.readlink(f).startswith("/dev/tenstorrent/")})


def run(scale, ttnn, ag, tt, nt, blocks, seed, seed_scale=1.0):
    """One rung: every activation scaled by `scale`, blocks chained, backward seeded, and the
    largest |gradient| anywhere reported along with each weight's own maximum."""
    rng = np.random.default_rng(seed)
    raw = dit.params(rng, nt)
    for k in dit.INPUTS:
        raw[k] = raw[k] * scale
    precise = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def up(v):
        return ttnn.from_torch(torch.from_numpy(v).to(torch.float32), dtype=ttnn.float32,
                               layout=ttnn.TILE_LAYOUT, device=tt.get_device())

    t = {k: ag.Tensor(up(v), requires_grad=(k in dit.WEIGHTS or k in dit.INPUTS))
         for k, v in raw.items()}
    a = t["a"]
    for _ in range(blocks):
        a = dit.block_tape(ag, dict(t, a=a), nt, cfg=precise, bwcfg=precise, attn_cfg=None)
    fwd_max = float(np.abs(ttnn.to_torch(a.value).to(torch.float64).numpy()).max())
    seed_arr = rng.standard_normal((nt, dit.C)) * seed_scale
    a.backward(seed=up(seed_arr))
    ttnn.synchronize_device(tt.get_device())

    per = {}
    for k in dit.WEIGHTS + dit.INPUTS:
        g = t[k].grad
        if g is None:
            continue
        v = ttnn.to_torch(g).to(torch.float64).numpy()
        per[k] = {"max_abs": float(np.abs(v).max()),
                  "finite": bool(np.isfinite(v).all())}
    worst = max(per.values(), key=lambda r: r["max_abs"])

    def worst_of(_p, _w=None):
        return max(r["max_abs"] for r in _p.values())

    seed_max = float(np.abs(seed_arr).max())
    return {"scale": scale, "seed_scale": seed_scale, "seed_max_abs": seed_max,
            "fwd_max_abs": fwd_max,
            # The chain's GAIN: how much the 24-block backward multiplies the magnitude of the
            # gradient handed to it. This is the number that decides overflow, because the
            # incoming seed is the loss's gradient and is not ours to choose -- so the usable
            # criterion is `seed_max * gain > bf16_max`, not a statement about activations.
            "gain": (worst_of(per) / seed_max) if seed_max > 0 else None,
            "grad_max_abs": worst["max_abs"],
            "all_finite_fp32": all(r["finite"] for r in per.values()),
            "would_overflow_bf16": bool(worst["max_abs"] > BF16_MAX),
            "bf16_headroom_decades": (float(np.log10(BF16_MAX / worst["max_abs"]))
                                      if worst["max_abs"] > 0 else None),
            "per_tensor_max_abs": {k: r["max_abs"] for k, r in sorted(
                per.items(), key=lambda x: -x[1]["max_abs"])[:6]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=dit.BLOCKS)
    ap.add_argument("--scales", default="1,2,4,8,16,32,64")
    ap.add_argument("--seed-scales", default="",
                    help="sweep the BACKWARD seed instead of the inputs. The block is layer-norm "
                         "dominated and therefore scale-invariant in its inputs, so the seed is "
                         "the axis that actually reaches the exponent range.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    tt.get_device()

    rows = []
    sweep = ([(1.0, float(x)) for x in args.seed_scales.split(",") if x]
             if args.seed_scales else
             [(float(x), 1.0) for x in args.scales.split(",") if x])
    for s, ss in sweep:
        r = run(s, ttnn, ag, tt, args.nt, args.blocks, args.seed, ss)
        rows.append(r)
        print(f"in x{s:<6g} seed x{ss:<10g} gain {r['gain']:.3e}"
              f"  grad_max {r['grad_max_abs']:.3e}"
              f"  bf16_headroom {r['bf16_headroom_decades']:+.2f} decades"
              f"  overflows_bf16={r['would_overflow_bf16']}", flush=True)

    # The crossing, read off the ladder rather than asserted. Growth is measured between
    # consecutive rungs so the extrapolation is the data's own slope, not an assumed power law.
    cross = next((r["scale"] for r in rows if r["would_overflow_bf16"]), None)
    slope = None
    if len(rows) >= 2 and rows[0]["grad_max_abs"] > 0:
        lo, hi = rows[0], rows[-1]
        axis = "seed_scale" if args.seed_scales else "scale"
        if hi[axis] != lo[axis]:
            slope = (np.log10(hi["grad_max_abs"] / lo["grad_max_abs"])
                     / np.log10(hi[axis] / lo[axis]))
    rep = {"nt": args.nt, "blocks": args.blocks, "seed": args.seed,
           "bf16_max": BF16_MAX, "fp32_max": FP32_MAX,
           "device_nodes_held": held_nodes(), "aiclk": aiclk(),
           "rows": rows,
           "swept": "seed" if args.seed_scales else "inputs",
           "first_scale_overflowing_bf16": cross,
           "median_gain": float(np.median([r["gain"] for r in rows if r["gain"]])),
           "seed_magnitude_that_overflows_bf16":
               float(BF16_MAX / np.median([r["gain"] for r in rows if r["gain"]])),
           "grad_decades_per_decade_of_input": (float(slope) if slope is not None else None)}
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
