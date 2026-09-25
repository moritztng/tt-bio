#!/usr/bin/env python3
"""What the softmax backward's row-sum renormalisation is worth against a CONFIGURED forward.

`TT_BIO_SOFTMAX_BW_RENORM` ships ON (ask 9629). Its value depends on the forward it corrects:
`of3t-d116` measured 4.0x on device against a no-config forward at [1,16,384,384], and a wash
against `precise_config()` (3.6605e-03 against 3.7217e-03), because the extra reduction's own
error is the size of the leak it removes. `of3t-fwdkcfg` configures two forward sites, so the
lever has to be re-priced at the shape those sites actually run.

Four arms, the full 2x2, all on device, all scored against a float64 reference evaluated on the
SAME values the card was handed, so what is scored is the rule and never the input rounding:

    forward none    x  renorm off / on
    forward precise x  renorm off / on

Two statistics, because the one everyone reaches for is blind to this defect (d116): the rel L2
of dx barely moves, and the ROW SUM of dx is the quantity that sees it. The true softmax
backward has row sums at float64 zero; a leak there multiplies the mean of k downstream and
lands in dq as a term the gradient does not contain.

Controls: a float64-against-float64 A/A at exactly 0, and a break control (column-permuted
reference) that must move the reading, or the harness has tested nothing.
"""
import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from perf import clocksample                                                # noqa: E402
from tt_bio import autograd as ag                                           # noqa: E402
from tt_bio.autograd import precise_config                                  # noqa: E402


def rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((got.double() - ref.double()))
                 / torch.linalg.vector_norm(ref.double()))


def rms_row_sum(t: torch.Tensor) -> float:
    return float(t.double().sum(dim=-1).pow(2).mean().sqrt())


def reference(x: torch.Tensor, g: torch.Tensor):
    """The exact softmax backward in float64: y = softmax(x), dx = y*(g - sum(g*y))."""
    y = torch.softmax(x.double(), dim=-1)
    return y, y * (g.double() - (g.double() * y).sum(dim=-1, keepdim=True))


def arm(dev, x_t, g_t, cfg, renorm: bool):
    prev = ag.SOFTMAX_BW_RENORM
    ag.SOFTMAX_BW_RENORM = renorm
    try:
        y = ttnn.softmax(x_t, dim=-1, compute_kernel_config=cfg)
        inner = ag.softmax_bw_inner(y, g_t, dim=-1)
        dx = ttnn.multiply(y, ttnn.subtract(g_t, inner))
        out = (ttnn.to_torch(y).float(), ttnn.to_torch(dx).float())
        ttnn.deallocate(y); ttnn.deallocate(inner); ttnn.deallocate(dx)
        return out
    finally:
        ag.SOFTMAX_BW_RENORM = prev


def run_shape(dev, shape, spread, seed):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.float32) * spread
    g = torch.randn(*shape, dtype=torch.float32)
    y64, dx64 = reference(x, g)

    x_t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    g_t = ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    cfg = precise_config()

    rows = {}
    for fwd, c in (("none", None), ("precise", cfg)):
        for rn in (False, True):
            y, dx = arm(dev, x_t, g_t, c, rn)
            rows["fwd=%s renorm=%s" % (fwd, "on" if rn else "off")] = {
                "y_rel_l2_vs_f64": rel_l2(y, y64),
                "y_mean_row_sum": float(y.double().sum(dim=-1).mean()),
                "dx_rel_l2_vs_f64": rel_l2(dx, dx64),
                "dx_rms_row_sum": rms_row_sum(dx),
            }
    ttnn.deallocate(x_t); ttnn.deallocate(g_t)

    # Controls. The A/A is the reference against itself and must be exactly 0; the break
    # control permutes the reference's last axis and must move the reading by orders.
    perm = dx64[..., torch.randperm(dx64.shape[-1])]
    rows["CONTROL float64 A/A"] = {"dx_rel_l2_vs_f64": rel_l2(dx64.float(), dx64),
                                   "dx_rms_row_sum": rms_row_sum(dx64)}
    rows["CONTROL permuted reference"] = {"dx_rel_l2_vs_f64": rel_l2(perm.float(), dx64),
                                          "dx_rms_row_sum": rms_row_sum(perm)}
    return {"shape": list(shape), "dtype": "fp32", "spread": spread, "seed": seed, "arms": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_fwdkcfg/RENORM_REPRICE_qb2c0.json")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--shapes", default="1x16x128x128,1x16x384x384")
    ap.add_argument("--spreads", default="8.0,12.0")
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    dev = get_device()

    results = []
    with clocksample.during(period=2.0) as clk:
        for s in a.shapes.split(","):
            shape = tuple(int(v) for v in s.split("x"))
            for spread in (float(v) for v in a.spreads.split(",")):
                for seed in (20260921, 7):
                    r = run_shape(dev, shape, spread, seed)
                    results.append(r)
                    print(s, "spread", spread, "seed", seed, flush=True)
                    for k, v in r["arms"].items():
                        print("   %-28s dx_rel %.4e  dx_row_sum %.4e" %
                              (k, v["dx_rel_l2_vs_f64"], v["dx_rms_row_sum"]), flush=True)

    out = {"what": "the softmax backward renormalisation re-priced against a configured forward",
           "host": socket.gethostname(), "card": a.card, "board": "p300c",
           "reference": "float64 torch softmax and its exact vjp, on the same values the card got",
           "clock_aiclk_during": clk.summary(), "clock_line": clk.line(0),
           "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(clk.line(0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
