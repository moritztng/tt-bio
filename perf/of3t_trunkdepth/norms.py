#!/usr/bin/env python3
"""The cheap statistic that makes the correlation testable: what the activations and the
incoming cotangent actually WEIGH at each captured boundary.

Every block in the trunk runs identical code, so a single-block backward error that moves 136x
between block 0 and block 47 is a property of the tensors at that depth and not of the program.
This reads those tensors off the captured boundaries and reports their norms MASKED -- the
crop-64 boundary carries 56 real token positions and 8 pad, and a pad-inclusive norm is not an
activation figure. The padded value is printed beside the masked one, never instead of it.

No device, no model, no gradient: it opens the boundary files and measures them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def norms(t: torch.Tensor, real: torch.Tensor, pair: bool) -> dict:
    t = t.double()
    m = t[:, real][:, :, real] if pair else t[:, real]
    return {"masked": float(m.norm()), "padded": float(t.norm()),
            "masked_share_of_squared": float((m.norm() ** 2) / (t.norm() ** 2))
            if float(t.norm()) > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", default="/home/ttuser/of3t_trunkdepth/cap_ladder")
    ap.add_argument("--blocks", default="0,8,16,23,32,40,47")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--out", default="perf/of3t_trunkdepth/NORMS.json")
    a = ap.parse_args()
    ks = [int(x) for x in a.blocks.split(",") if x != ""]
    out = {"what": __doc__.strip().splitlines()[0], "cap": a.cap, "crop": a.crop, "depths": {}}
    for k in ks:
        cap = torch.load(Path(a.cap) / f"block{k}_boundary.pt", map_location="cpu",
                         weights_only=False)
        s_in, z_in = cap["args"][0], cap["args"][1]
        sm = None
        for key, v in (cap["kwargs"] or {}).items():
            if torch.is_tensor(v) and "single" in key:
                sm = v
        if sm is None:
            raise SystemExit(f"block {k}: no single mask on the boundary, cannot mask")
        c = a.crop
        s_in, z_in = s_in[:, :c], z_in[:, :c, :c]
        s_out, z_out = cap["out"][0][:, :c], cap["out"][1][:, :c, :c]
        cot_s, cot_z = cap["cot"][0][:, :c], cap["cot"][1][:, :c, :c]
        sm = sm[:, :c]
        real = (sm.reshape(-1) > 0)
        row = {"tokens": int(c), "real_tokens": int(real.sum()),
               "pad_fraction_single": 1.0 - float(real.sum()) / float(real.numel()),
               "pad_fraction_pair": 1.0 - (float(real.sum()) / float(real.numel())) ** 2,
               "s_in": norms(s_in, real, False), "z_in": norms(z_in, real, True),
               "s_out": norms(s_out, real, False), "z_out": norms(z_out, real, True),
               "cot_s_out": norms(cot_s, real, False), "cot_z_out": norms(cot_z, real, True)}
        out["depths"][str(k)] = row
        print(f"block {k:2d}  |s_in| {row['s_in']['masked']:.6e}  |z_in| {row['z_in']['masked']:.6e}"
              f"  |ds_out| {row['cot_s_out']['masked']:.6e}"
              f"  |dz_out| {row['cot_z_out']['masked']:.6e}"
              f"   (padded |s_in| {row['s_in']['padded']:.6e})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
