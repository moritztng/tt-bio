#!/usr/bin/env python3
"""Is `AF2PairBlock.rne_residual` buying a ROUNDING MODE that a bf16 add can also buy?

`_residual` is wide for one stated reason (`tt_bio/af2.py`): `ttnn.add` breaks ties away from
zero on its bfloat16 datapath and torch/JAX round ties to even, and that bias is the whole of
this trunk's error growth (52 failed taps of 94 and 0.084555 of i_pTM, down to 9 and 0.002605).
It is NOT wide for dynamic range. So the question leg 2 asks is whether the rounding can be had
without the bytes.

The wide path today reads and writes float32 three times per residual add:

    typecast(x, f32) + typecast(update, f32) -> add_ -> typecast(out, bf16)

The candidate is one op: `ttnn.add` on bf16 operands with the repo's own kernel config, which
carries `fp32_dest_acc_en=True`. That accumulates in the fp32 DEST register and packs once to
bfloat16, so the arithmetic is the same fp32 sum with the same single final rounding. If the
packer rounds ties to even, the wide detour is buying nothing the config does not already buy,
and narrowing it is a byte win at no accuracy cost.

Arms, all on the same inputs at the trunk's own shapes and magnitudes:

    add_bare      `ttnn.add_(z, u)`, no config -- reproduces the known half-away result
    add_cfg       `ttnn.add(z, u, ...)` with `af2.compute_kernel_config()`, bf16 out
    wide          today's `_residual`: f32 typecasts, f32 add, typecast back
    torch         `z + u` in bfloat16, the reference

Each arm is scored against the exact fp32 sum narrowed under each candidate rule, so the output
says which rounding each arm took rather than only how far apart they are.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROUNDING_MODES = ("rne", "half_away", "truncate")


def _round(exact: torch.Tensor, mode: str) -> torch.Tensor:
    """`exact` (float32) narrowed to bfloat16 under one candidate rule. `rne` is torch's."""
    if mode == "rne":
        return exact.to(torch.bfloat16)
    bits = exact.view(torch.int32).to(torch.int64)
    if mode == "half_away":
        bits = bits + 0x8000
    return (bits & ~0xFFFF).to(torch.int32).view(torch.float32).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--ratios", default="1.0,0.3,0.1,0.03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rne_probe.json")
    args = ap.parse_args()

    import ttnn
    from tt_bio import af2
    from tt_bio.tenstorrent import get_device

    device = get_device()
    cfg = af2.compute_kernel_config()
    rows, cols, ch = args.rows, args.cols, args.channels

    def up(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    def down(t):
        return torch.Tensor(ttnn.to_torch(t)).to(torch.bfloat16)

    rowsout = []
    for ratio in [float(x) for x in args.ratios.split(",")]:
        g = torch.Generator().manual_seed(args.seed)
        z = torch.randn(1, rows, cols, ch, generator=g).to(torch.bfloat16)
        u = (ratio * torch.randn(1, rows, cols, ch, generator=g)).to(torch.bfloat16)
        exact = z.float() + u.float()
        arms = {}

        zt, ut = up(z), up(u)
        arms["add_bare"] = down(ttnn.add_(zt, ut))
        ttnn.deallocate(zt)

        # `ttnn.add` takes NO compute_kernel_config on this build, so fp32 accumulation is
        # not something the op can be asked for: the only way to get it is to hand it fp32
        # tensors. What IS reachable is bf16 OPERANDS with an fp32 RESULT, which keeps the
        # reads narrow and moves the widening to the one place it may be needed.
        zt, ut = up(z), up(u)
        wide_out = ttnn.add(zt, ut, dtype=ttnn.float32)
        arms["add_f32out"] = down(ttnn.typecast(wide_out, ttnn.bfloat16))
        ttnn.deallocate(wide_out), ttnn.deallocate(zt), ttnn.deallocate(ut)

        # today's wide path, reproduced verbatim from AF2PairBlock._residual
        zt, ut = up(z), up(u)
        wide = ttnn.typecast(zt, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        other = ttnn.typecast(ut, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wide = ttnn.add_(wide, other)
        arms["wide"] = down(ttnn.typecast(wide, ttnn.bfloat16))
        ttnn.deallocate(wide), ttnn.deallocate(other)

        arms["torch"] = (z + u).to(torch.bfloat16)

        rec = {"ratio": ratio, "shape": [rows, cols, ch], "elements": int(z.numel()), "arms": {}}
        ref = arms["torch"]
        scale = exact.abs().reshape(-1).clamp(min=1e-30)
        for name, got in arms.items():
            err = (got.float() - exact).reshape(-1) / scale
            rec["arms"][name] = {
                "vs_rule": {m: int((got != _round(exact, m)).sum()) for m in ROUNDING_MODES},
                "differs_from_torch": int((got != ref).sum()),
                "mean_signed_rel": float(err.mean()),
                "rms_rel": float(err.square().mean().sqrt()),
                "exact_frac": float((err == 0).float().mean()),
            }
        rowsout.append(rec)
        print(json.dumps({"ratio": ratio, **{k: {"vs_rne": v["vs_rule"]["rne"],
                                                 "vs_half_away": v["vs_rule"]["half_away"],
                                                 "differs_from_torch": v["differs_from_torch"],
                                                 "mean_signed_rel": round(v["mean_signed_rel"], 9)}
                                             for k, v in rec["arms"].items()}}), flush=True)

    out = pathlib.Path(ROOT / "perf" / "bcx_p10_bfp8" / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import os
    import time
    out.write_text(json.dumps({
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "utc": time.strftime("%FT%TZ", time.gmtime()), "loadavg": os.getloadavg(),
        "kernel_config": "af2.compute_kernel_config(): HiFi4, fp32_dest_acc_en, packer_l1_acc",
        "rows": rowsout}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
