"""outer_residual with a bfloat8_b pair stream (`pair_bfp8z`) against float64: z' = z + pj4 * pi4, the sum packed bfp8.

    TT_VISIBLE_DEVICES=<card> timeout 300 python perf/rfd4/z8/or_check.py --out <json>

The card's output is compared element for element with zprec_cpu.bfp8 of the float64 sum (round and trunc), and its
rel RMS vs float64 with the bf16 arm's.
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
from tt_bio.outer_residual import outer_residual  # noqa: E402
from zprec_cpu import bfp8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    torch.manual_seed(0)
    n, c = a.n, 128
    z = torch.randn(1, n, n, c) * torch.logspace(-3, 1, c, base=2.0)[torch.randperm(c)]
    pi, pj = torch.randn(1, n, c).bfloat16(), torch.randn(1, n, 1, c).bfloat16()
    dev = T.get_device()
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    res = {"n": n, "c": c}
    for name, dt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
        zd = up(z.float(), dt)
        zin = ttnn.to_torch(zd).double()                     # z as the card holds it
        ref = zin + pj.double() * pi.double()[:, None]       # float64 sum, [1, n(i), n(j), c]: pj4[b, i] * pi[b, j]
        out = outer_residual(zd, up(pi, ttnn.bfloat16), up(pj, ttnn.bfloat16))
        assert out is not None, name
        got = ttnn.to_torch(out[0]).double()
        r = {"rel_rms_vs_f64": ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()}
        if name == "bfp8":
            for k, tr in (("round", False), ("trunc", True)):
                e = bfp8(ref, -1, tr)
                r[k] = {"equal_frac": (got == e).double().mean().item(), "max_abs_diff": (got - e).abs().max().item()}
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
