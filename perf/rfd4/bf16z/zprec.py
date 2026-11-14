"""z and s of the released trunk against float64, RFD4's default lever set vs the same plus `--lever` (`pair_bf16z`).

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/bf16z/zprec.py --out <json> [--n 256] [--batch 1] [--cycles 1] [--lever NAME]

The golden trunk inputs (tests/test_rfd4_trunk_golden.py) tiled along the token axis to `--n` tokens with no padded
token, so every fused program the lever changes (outer_residual, pair_tail, pair_relu) serves; resid_precision.py's
golden case is padded and runs none of them. Graded against a float64 reference (bar 0.05 rel_rms, as the golden
test), never against another approximation; the bf16-mixed floor is printed beside it. `--batch B` runs B copies of the
input as one trunk call (pair_batch's route) and grades all of them against the one reference.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import torch  # noqa: E402

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import rfd4_reference as REF  # noqa: E402
import test_rfd4_trunk_golden as G  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.rfd4 import trunk as TR  # noqa: E402


def tile(t, n, dims):
    for d in dims:
        reps = [1] * t.dim()
        reps[d] = -(-n // t.shape[d])
        t = t.repeat(*reps).narrow(d, 0, n)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--lever", default="pair_bf16z")
    a = ap.parse_args()
    weights = {k: v.double() for k, v in torch.load(G.WEIGHTS, map_location="cpu", weights_only=False)["weights"].items()}
    (x, z, cond, mask), _ = G._golden("MultiheadAttnAndTransition__nn_transformer_layers_1__call0.pt")
    m = mask[0].bool()
    x, z, cond = x[:, m], z[:, m][:, :, m], cond[:, m]     # the real tokens only, then tiled
    n = a.n
    x, z, cond = tile(x, n, [1]), tile(z, n, [1, 2]), tile(cond, n, [1])
    mask = torch.ones(1, n, dtype=torch.float64)
    ref_s, ref_z = REF.trunk(weights, x, z, [cond] * a.cycles, mask, 14, n_heads=G.HEADS)
    wb = {kk: REF.bf16(v) for kk, v in weights.items()}
    fl_s, fl_z = REF.trunk(wb, x, z, [cond] * a.cycles, mask, 14, n_heads=G.HEADS, rnd=REF.bf16_mixed)
    rel = lambda g, r: ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()  # noqa: E731
    B = a.batch
    rb = lambda t: t.repeat(B, *[1] * (t.dim() - 1))  # noqa: E731
    ref_s, ref_z = rb(ref_s), rb(ref_z)
    dev = T.get_device()
    res = {"tokens": n, "batch": B, "cycles": a.cycles, "bar": G.BAR, "floor": {"s": rel(fl_s, ref_s), "z": rel(fl_z, ref_z)}}
    base = ",".join(sorted(TR.NORMAL_LEVERS))
    outs = {}
    for name, spec in (("base", base), (a.lever, base + "," + a.lever)):
        os.environ["TT_BIO_LEVERS"] = spec
        mod = TR.Trunk(T.WeightScope({kk: v.float() for kk, v in weights.items()}), TR.compute_kernel_config(), 14,
                       G.HEADS, residual_dtype=ttnn.float32)
        zd = mod.pair_residual_dtype
        masks = TR.trunk_masks(rb(mask), dev, residual_dtype=ttnn.float32)
        s, zt = mod(G._dev(rb(x), dev, ttnn.float32), G._dev(rb(z), dev, zd), [G._dev(rb(cond), dev)] * a.cycles, masks)
        gs, gz = ttnn.to_torch(s).double(), ttnn.to_torch(zt).double()
        outs[name] = (gs, gz)
        res[name] = {"pair_dtype": str(zd), "s": rel(gs, ref_s), "z": rel(gz, ref_z)}
        print(name, res[name], flush=True)
    res[a.lever + "_vs_base"] = {"s": rel(outs[a.lever][0], outs["base"][0]), "z": rel(outs[a.lever][1], outs["base"][1])}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
