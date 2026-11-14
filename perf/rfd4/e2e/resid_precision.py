"""The released trunk, 14 blocks x 6 recycles on golden inputs, against float64: fp32 pair stream
(the release config) and the bf16 pair stream probe, side by side, s and z.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/resid_precision.py --out <json>

Same case and grading as tests/test_rfd4_trunk_golden.py::test_released_trunk_14x6_recycles (bar 0.05
rel_rms vs float64); the bf16-mixed floor is printed beside it.
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=6)
    a = ap.parse_args()
    weights = {k: v.double() for k, v in torch.load(G.WEIGHTS, map_location="cpu", weights_only=False)["weights"].items()}
    (x, z, cond, mask), _ = G._golden("MultiheadAttnAndTransition__nn_transformer_layers_1__call0.pt")
    mask = mask.double()
    k, n = x.shape[1], G._bucket(x.shape[1])
    ref_s, ref_z = REF.trunk(weights, x, z, [cond] * a.cycles, mask, 14, n_heads=G.HEADS)
    wb = {kk: REF.bf16(v) for kk, v in weights.items()}
    fl_s, fl_z = REF.trunk(wb, x, z, [cond] * a.cycles, mask, 14, n_heads=G.HEADS, rnd=REF.bf16_mixed)
    m, pm = mask.bool(), (mask[:, :, None] * mask[:, None, :]).bool()
    rel = lambda g, r: ((g - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item()  # noqa: E731
    dev = T.get_device()
    res = {"tokens": k, "bucket": n, "cycles": a.cycles, "bar": G.BAR,
           "floor": {"s": rel(fl_s[m], ref_s[m]), "z": rel(fl_z[pm], ref_z[pm])}}
    outs = {}
    for name, zd in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        mod = TR.Trunk(T.WeightScope({kk: v.float() for kk, v in weights.items()}), TR.compute_kernel_config(), 14,
                       G.HEADS, residual_dtype=ttnn.float32, pair_residual_dtype=zd)
        masks = TR.trunk_masks(G._pad(mask, n, [1]), dev, residual_dtype=ttnn.float32)
        ct = G._dev(G._pad(cond, n, [1]), dev)
        s, zt = mod(G._dev(G._pad(x, n, [1]), dev, ttnn.float32), G._dev(G._pad(z, n, [1, 2]), dev, zd),
                    [ct] * a.cycles, masks)
        gs = ttnn.to_torch(s).double()[:, :k]
        gz = ttnn.to_torch(zt).double()[:, :k, :k]
        outs[name] = (gs, gz)
        res[name] = {"s": rel(gs[m], ref_s[m]), "z": rel(gz[pm], ref_z[pm])}
        print(name, res[name], flush=True)
    res["bf16_vs_fp32"] = {"s": rel(outs["bf16"][0][m], outs["fp32"][0][m]),
                           "z": rel(outs["bf16"][1][pm], outs["fp32"][1][pm])}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
