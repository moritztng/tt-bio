#!/usr/bin/env python3
"""What the four no-config forward sites cost NOW that the backward is corrected.

`of3t-d116` named four `ttnn.softmax` calls that reach the kernel with no
`compute_kernel_config` and attributed 93 % of the softmax row-sum deficit to that one
missing argument. Two things have changed since, and they pull in opposite directions:

  * `TT_BIO_SOFTMAX_BW_RENORM` now ships ON (`of3t-d56-renorm`), and the renormalised rule
    `y*(g - sum(g*y)/sum(y))` is the EXACT vjp of `c*softmax(x)` for any per-row `c`. So the
    row-sum deficit -- the 93 % -- is absorbed exactly, however large it is. Whatever the
    config was worth through that channel, it is now worth nothing.
  * What it cannot absorb is `y`'s DIRECTIONAL error: the card's row is not a scalar multiple
    of the true softmax row. That error reaches the forward and the backward alike, and no
    backward rule can reach it.

So this table re-prices each site under both rules. `d_gain_shipped` is what the config was
worth in the backward before the repair, `d_gain_renorm` is what it is worth after.

Shapes and dtypes are not guessed: they are the ones a shipped OpenFold3 fold on ubq actually
executes, recorded by a `ttnn.softmax` spy riding on PYTHONPATH through the spawned worker
(`perf/of3t_d116_verify/SITES_CENSUS.json`).

Every number is against a float64 `torch.softmax` on the SAME values the card was handed, so
what is scored is the operation and never the input rounding.
"""
import argparse
import json
import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio                                                    # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT), (
    f"tt_bio came from {_tt_bio.__file__}, not {_ROOT}")

import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio.autograd import precise_config                                  # noqa: E402
from tt_bio.tenstorrent import get_device                                   # noqa: E402

# site token -> the shape and storage dtype the fold runs it at, and how many calls per fold.
SITES = [
    ("openfold3.diffusion_transformer", "openfold3_diffusion_transformer.py:211",
     (1, 16, 96, 96), "float32", 4800),
    ("openfold3.atom_transformer", "openfold3_atom_transformer.py:186",
     (1, 19, 4, 32, 128), "float32", 1200),
    ("openfold3.atom_transformer [host prep]", "openfold3_atom_transformer.py:186",
     (1, 19, 4, 32, 128), "bfloat16", 3),
    # AttentionPairBias: the token exists and defaults off, but the shipped OF3 fold takes the
    # sibling branch at tenstorrent.py:8645, which passes a config. Priced at the trunk shape
    # the taped backward runs it at, because that is where it would bite if it fired.
    ("AttentionPairBias site_softmax", "tenstorrent.py:8604",
     (1, 16, 384, 384), "bfloat16", 0),
    # ConfidenceHeadsDevice._reduce: a bare no-config call, not on the registry at all.
    ("ConfidenceHeadsDevice._reduce", "tenstorrent.py:13082",
     (1, 384, 384, 64), "bfloat16", 0),
]

DT = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16}


def rel_l2(a, b):
    return float(torch.linalg.vector_norm((a - b).double())
                 / torch.linalg.vector_norm(b.double()))


def rms_rowsum(t):
    return float(t.sum(-1).pow(2).mean().sqrt())


def measure(dev, shape, dtype_name, std, seed):
    torch.manual_seed(seed)
    x = (torch.randn(*shape) * std).to(torch.bfloat16)
    g = torch.randn(*shape).to(torch.bfloat16)
    dt = DT[dtype_name]
    x_tt = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    g_tt = ttnn.from_torch(g, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    x64, g64 = x.double(), g.double()
    p = torch.softmax(x64, dim=-1)
    dx_ref = p * (g64 - (g64 * p).sum(-1, keepdim=True))

    arms = {}
    for nm, cfg in (("none", None), ("precise_config()", precise_config())):
        kw = {} if cfg is None else {"compute_kernel_config": cfg}
        y_tt = ttnn.softmax(x_tt, dim=-1, **kw)
        y = ttnn.to_torch(y_tt).double()
        rows = y.sum(-1, keepdim=True)
        inner = (g64 * y).sum(-1, keepdim=True)
        dx_ship = y * (g64 - inner)
        dx_rn = y * (g64 - inner / rows)
        # and on the card, which is where the rule actually runs
        i_tt = ttnn.sum(ttnn.multiply(g_tt, y_tt), dim=-1, keepdim=True)
        dx_ship_dev = ttnn.to_torch(
            ttnn.multiply(y_tt, ttnn.subtract(g_tt, i_tt))).double()
        r_tt = ttnn.sum(y_tt, dim=-1, keepdim=True, compute_kernel_config=precise_config())
        dx_rn_dev = ttnn.to_torch(
            ttnn.multiply(y_tt, ttnn.subtract(g_tt, ttnn.divide(i_tt, r_tt)))).double()
        ttnn.deallocate(y_tt)
        arms[nm] = {
            "rowsum_mean": float(rows.mean()),
            "rowsum_rmsdev": float((rows - 1).pow(2).mean().sqrt()),
            "y_rel_l2": rel_l2(y, p),
            "dx_rel_l2": {"shipped_rule_f64": rel_l2(dx_ship, dx_ref),
                          "renorm_rule_f64": rel_l2(dx_rn, dx_ref),
                          "shipped_device": rel_l2(dx_ship_dev, dx_ref),
                          "renorm_device": rel_l2(dx_rn_dev, dx_ref)},
            "dx_rowsum_rms": {"reference": rms_rowsum(dx_ref),
                              "shipped_rule_f64": rms_rowsum(dx_ship),
                              "renorm_rule_f64": rms_rowsum(dx_rn),
                              "shipped_device": rms_rowsum(dx_ship_dev),
                              "renorm_device": rms_rowsum(dx_rn_dev)},
        }
    n, pr = arms["none"], arms["precise_config()"]
    gains = {
        "forward": n["y_rel_l2"] / pr["y_rel_l2"],
        "d_gain_shipped": (n["dx_rel_l2"]["shipped_device"]
                           / pr["dx_rel_l2"]["shipped_device"]),
        "d_gain_renorm": (n["dx_rel_l2"]["renorm_device"]
                          / pr["dx_rel_l2"]["renorm_device"]),
        "rowsum": n["rowsum_rmsdev"] / pr["rowsum_rmsdev"],
    }
    # A/A: the same call twice must be bit-identical, or an arm's reading is not its arm's
    y1 = ttnn.to_torch(ttnn.softmax(x_tt, dim=-1))
    y2 = ttnn.to_torch(ttnn.softmax(x_tt, dim=-1))
    # break control: the exact reference scored against itself with the softmax axis
    # reversed. It must read O(1) or rel_l2 is not reading the thing it is asked to read.
    brk = rel_l2(dx_ref, dx_ref.flip(-1))
    return {"arms": arms, "what_the_config_buys": gains,
            "AA_bit_identical": bool(torch.equal(y1, y2)),
            "break_control_reference_flipped": brk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_d116_verify/SITES.json")
    ap.add_argument("--seeds", default="20260921,7")
    a = ap.parse_args()
    dev = get_device()
    t0 = time.time()
    rows = []
    for token, site, shape, dtn, calls in SITES:
        for seed in (int(s) for s in a.seeds.split(",")):
            for std in (3.0, 12.0):
                r = measure(dev, shape, dtn, std, seed)
                rows.append({"token": token, "site": site, "shape": list(shape),
                             "storage": dtn, "calls_per_of3_ubq_fold": calls,
                             "std": std, "seed": seed, **r})
                gz = r["what_the_config_buys"]
                print("%-42s %-22s std=%-5s seed=%-9d fwd %.2fx  bwd(shipped) %.2fx  "
                      "bwd(renorm) %.2fx" % (token, shape, std, seed, gz["forward"],
                                             gz["d_gain_shipped"], gz["d_gain_renorm"]),
                      flush=True)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": os.uname().nodename, "what": __doc__.strip().splitlines()[0],
           "seconds": round(time.time() - t0, 1), "cases": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
