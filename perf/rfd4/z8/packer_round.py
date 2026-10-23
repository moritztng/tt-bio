"""How the packer rounds an fp32 DEST value into bfp8_b (and bf16): round to nearest, or truncate.

    TT_VISIBLE_DEVICES=<card> timeout 300 python perf/rfd4/z8/packer_round.py --out <json>

zprec_cpu.py shows the answer decides whether the pair stream can be stored bfp8: rounded to nearest, z reads 0.023
rel RMS vs float64 at 128 tokens (bar 0.05); truncated, 0.127. Each route below packs known fp32 values on the card
and the bits read back are compared with zprec_cpu.bfp8's two emulations, element for element.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

from tt_bio import tenstorrent as T  # noqa: E402
from zprec_cpu import bfp8  # noqa: E402


def bf16_modes(t):
    """fp32 -> bf16 as round-to-nearest-even, ties-away and truncation, back in float64."""
    u = t.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    lo, hi = u & 0xFFFF, u >> 16
    trunc = hi
    away = hi + (lo >= 0x8000).long()
    even = hi + ((lo > 0x8000) | ((lo == 0x8000) & (hi & 1).bool())).long()
    f = lambda h: (h << 16).to(torch.int64).to(torch.int32).view(torch.float32).double()  # noqa: E731
    return {"rne": f(even), "away": f(away), "trunc": f(trunc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    # [1, 1, 256, 384], channel-last like z: block scales vary by 2^-8 .. 2^4 along the rows and inside each block
    x = (torch.randn(1, 1, 256, 384) * torch.logspace(-8, 4, 384, base=2.0)[torch.randperm(384)]).float()
    y = (torch.randn_like(x) * 0.01).float()
    dev = T.get_device()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    xd, yd = up(x), up(y)
    exact = (x.double() + y.double()).float().double()          # the fp32 sum DEST holds
    routes = {
        "typecast_bfp8": (ttnn.typecast(xd, ttnn.bfloat8_b), x.double()),
        "add_out_bfp8": (ttnn.add(xd, yd, dtype=ttnn.bfloat8_b), exact),
        "typecast_bf16": (ttnn.typecast(xd, ttnn.bfloat16), x.double()),
        "add_out_bf16": (ttnn.add(xd, yd, dtype=ttnn.bfloat16), exact),
    }
    res = {}
    for name, (t, src) in routes.items():
        got = ttnn.to_torch(t).double().reshape(src.shape)
        if "bfp8" in name:
            emu = {"round": bfp8(src, -1), "trunc": bfp8(src, -1, trunc=True),
                   "round_of_bf16": bfp8(src.float().bfloat16().double(), -1),
                   "trunc_of_bf16": bfp8(src.float().bfloat16().double(), -1, trunc=True)}
        else:
            emu = bf16_modes(src)
        res[name] = {k: {"equal_frac": (got == v).double().mean().item(),
                         "max_abs_diff": (got - v).abs().max().item()}
                     for k, v in emu.items()}
        # truncation shrinks magnitudes: a mean |got| - |src| near -half a step is the tell, round-to-nearest sits at 0
        res[name]["magnitude_bias_rel"] = ((got.abs() - src.abs()).mean() / src.abs().mean()).item()
        res[name]["got_vs_src_rel_rms"] = ((got - src).pow(2).mean().sqrt() / src.pow(2).mean().sqrt()).item()
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
