"""J2: what the fused softmax backward costs and how accurate it is, at our real shapes.

Four arms, all producing the SAME tensor -- the softmax backward `ds` -- so they are
comparable end to end rather than verb by verb:

    composed_renorm   the shipped path: multiply, sum, sum, divide, subtract, multiply
                      (`autograd.softmax_bw_inner` + its caller's two verbs)   = 6 verbs
    composed_plain    the same with SOFTMAX_BW_RENORM off                      = 4 verbs
    moreh             ttnn.moreh_softmax_backward(y, g, dim=-1)                = 1 verb
    moreh_renorm      moreh on y/rowsum, rescaled by rowsum: the shipped
                      expression exactly, through the fused kernel             = 4 verbs

Arms are interleaved and the order reverses every round, because an ordered sweep charges
the first arm for compile and hands the last one a warm cache. Every arm is warmed before
any timing. `ttnn.synchronize_device` bounds every timed region on both sides.

Accuracy is scored in the same run against two float64 references, because they answer
different questions:

    vs_expr   ds from the SAME y the arm was handed, in float64. Grades the arithmetic.
    vs_true   ds from the float64 softmax of x. Grades the VJP the model actually wants,
              which is A46 clause 1's subject. `--fd` validates this reference against
              central finite differences on a small case.

The AICLK is sampled DURING the timed work. A timing without one is not a measurement.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import ttnn

from perf import clocksample
from tt_bio.autograd import precise_config


def f64_softmax(x: torch.Tensor) -> torch.Tensor:
    xd = x.to(torch.float64)
    m = xd.amax(dim=-1, keepdim=True)
    e = (xd - m).exp()
    return e / e.sum(dim=-1, keepdim=True)


def f64_softmax_bw(y64: torch.Tensor, g64: torch.Tensor) -> torch.Tensor:
    """`dx = y (g - sum_j g_j y_j)`, the softmax VJP, in float64."""
    return y64 * (g64 - (g64 * y64).sum(dim=-1, keepdim=True))


# --- the four arms, each returning ds ------------------------------------------------------

# The op takes `dim` as a uint32_t (`moreh_softmax_backward.hpp`), so -1 does not name the
# last axis to it. Every shape this harness runs is rank 4.
_AX = 3


def _composed(y, g, cfg, renorm):
    inner = ttnn.sum(ttnn.multiply(g, y), dim=-1, keepdim=True)
    if renorm:
        inner = ttnn.divide(inner, ttnn.sum(y, dim=-1, keepdim=True,
                                            compute_kernel_config=cfg))
    return ttnn.multiply(y, ttnn.subtract(g, inner))


def _moreh_renorm(y, g, cfg):
    """`y (g - sum(g y)/s)` with s = rowsum(y), through the fused kernel.

    moreh(y/s, g) = (y/s)(g - sum(g y)/s), so multiplying by s returns the shipped
    expression exactly. Four verbs where the composed path spends six.
    """
    s = ttnn.sum(y, dim=-1, keepdim=True, compute_kernel_config=cfg)
    return ttnn.multiply(ttnn.moreh_softmax_backward(ttnn.divide(y, s), g, _AX), s)


ARMS = {
    "composed_renorm": lambda y, g, cfg: _composed(y, g, cfg, True),
    "composed_plain": lambda y, g, cfg: _composed(y, g, cfg, False),
    "moreh": lambda y, g, cfg: ttnn.moreh_softmax_backward(y, g, _AX),
    "moreh_renorm": _moreh_renorm,
}
VERBS = {"composed_renorm": 6, "composed_plain": 4, "moreh": 1, "moreh_renorm": 4}


def rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    d = got.astype(np.float64) - ref
    n = np.sqrt((ref * ref).sum())
    return float(np.sqrt((d * d).sum()) / n) if n else float("nan")


def run_shape(device, shape, dtype, iters, rounds, spread, exact_fwd, seed=0):
    """One rung. `exact_fwd` picks which forward produced y: the host float64 softmax
    rounded to the card (what a training tape runs, `autograd._exact_softmax_raw`) or the
    device `ttnn.softmax` (what `exact_training(False)` runs). The two differ in ROW SUM,
    which is the whole renorm question."""
    torch.manual_seed(seed)
    x_host = torch.randn(*shape, dtype=torch.float32) * spread
    g_host = torch.randn(*shape, dtype=torch.float32)
    tt_dtype = ttnn.float32 if dtype == "fp32" else ttnn.bfloat16
    cfg = precise_config()

    x = ttnn.from_torch(x_host, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(g_host, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    y_true64 = f64_softmax(x_host)
    if exact_fwd:
        y = ttnn.from_torch(y_true64.float(), dtype=tt_dtype, layout=ttnn.TILE_LAYOUT,
                            device=device)
    else:
        y = ttnn.softmax(x, dim=-1, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)

    # the y the arms are actually handed, read back so both references use it
    y_dev64 = ttnn.to_torch(y).double()
    g_dev64 = ttnn.to_torch(g).double()
    row_sum = y_dev64.sum(dim=-1)
    ref_expr = f64_softmax_bw(y_dev64, g_dev64).numpy()
    ref_true = f64_softmax_bw(y_true64, g_dev64).numpy()

    acc_expr, acc_true, per_call = {}, {}, {a: [] for a in ARMS}

    # Warm and score before any timing. An arm the op REFUSES at this dtype is recorded and
    # dropped rather than crashing the rung: `moreh_softmax_backward` takes bfloat16 and
    # bfloat8_b only (`moreh_softmax_backward_device_operation.cpp:80`), and "which arms are
    # even available at which dtype" is a result, not an error.
    refused = {}
    for name, fn in list(ARMS.items()):
        try:
            ds = fn(y, g, cfg)
            ttnn.synchronize_device(device)
        except RuntimeError as e:
            refused[name] = str(e).split("info:")[-1].strip().split("\n")[0]
            continue
        got = ttnn.to_torch(ds).float().numpy()
        acc_expr[name] = rel_l2(got, ref_expr)
        acc_true[name] = rel_l2(got, ref_true)
        ttnn.deallocate(ds)

    order = [a for a in ARMS if a not in refused]
    per_call = {a: [] for a in order}
    for r in range(rounds):
        for name in (order if r % 2 == 0 else order[::-1]):
            fn = ARMS[name]
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                ds = fn(y, g, cfg)
                ttnn.deallocate(ds)
            ttnn.synchronize_device(device)
            per_call[name].append((time.perf_counter() - t0) / iters)

    for t in (x, g, y):
        ttnn.deallocate(t)
    med = {k: 1e3 * statistics.median(v) for k, v in per_call.items() if v}
    return {
        "refused_at_this_dtype": refused,
        "shape": list(shape), "dtype": dtype, "spread": spread,
        "forward": "host_f64_rounded" if exact_fwd else "ttnn.softmax",
        "iters": iters, "rounds": rounds, "verbs": VERBS,
        "row_sum_max_abs_dev_from_one": float((row_sum - 1.0).abs().max()),
        "rel_l2_vs_expr_f64": acc_expr,
        "rel_l2_vs_true_f64": acc_true,
        "ms_per_call_median": med,
        "ms_per_call_min": {k: 1e3 * min(v) for k, v in per_call.items() if v},
        "ms_per_call_rounds": {k: [1e3 * t for t in v] for k, v in per_call.items() if v},
        "speedup_vs_composed_renorm": {k: med["composed_renorm"] / v for k, v in med.items()},
    }


def finite_difference_check(shape=(1, 1, 32, 64), spread=4.0, eps=1e-5, dirs=8, seed=0):
    """Validate `f64_softmax_bw` itself against central differences, in float64, host only.

    DIRECTIONAL, not per-entry. `f(x) = sum(g * softmax(x))` has a gradient whose entries
    span many magnitudes -- a softmax row's small entries carry gradients near zero -- so a
    per-entry relative error is dominated by the conditioning of the smallest entry and
    reads 1.3e-2 on a reference that is in fact exact. Along a random unit direction v,
    `(f(x + eps v) - f(x - eps v)) / 2 eps` against `<grad, v>` is one well-conditioned
    scalar, and it is the same claim. A46 clause 1 asks for the reference to be validated
    rather than asserted.
    """
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.float64) * spread
    g = torch.randn(*shape, dtype=torch.float64)
    analytic = f64_softmax_bw(f64_softmax(x), g)
    worst = 0.0
    for _ in range(dirs):
        v = torch.randn(*shape, dtype=torch.float64)
        v /= v.norm()
        num = ((g * f64_softmax(x + eps * v)).sum()
               - (g * f64_softmax(x - eps * v)).sum()) / (2 * eps)
        ana = (analytic * v).sum()
        worst = max(worst, abs(float(num - ana)) / max(abs(float(ana)), 1e-12))
    return {"directions": dirs, "eps": eps, "worst_rel_err": worst, "shape": list(shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_softbw/softmax_bw_cost_pc0.json")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--host", default="pc")
    ap.add_argument("--board", default="p150a")
    ap.add_argument("--shapes", default="")
    a = ap.parse_args()

    # The shapes the softmax backward is actually taken at. TRI_ATT on an OF3T crop-384
    # pairformer is [1, H, 384, 384]; the 16-head rung keeps this row comparable with
    # perf/of3t_softmax and perf/of3t_f64softmax, which both used it; the 512 rung is the
    # Boltz-2 512 aa fold's tri-att, which is the same shared path.
    shapes = [
        ((1, 4, 384, 384), "fp32"),
        ((1, 16, 384, 384), "fp32"),
        ((1, 16, 384, 384), "bf16"),
        ((1, 16, 512, 512), "fp32"),
    ]
    if a.shapes:
        shapes = []
        for s in a.shapes.split(","):
            dims, _, dt = s.partition("@")
            shapes.append((tuple(int(v) for v in dims.split("x")), dt or "fp32"))

    fd = finite_difference_check()
    print("FD check of the float64 reference:", json.dumps(fd), flush=True)

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    device = get_device()
    results = []
    with clocksample.during(period=2.0) as clk:
        for shape, dtype in shapes:
            for exact_fwd in (True, False):
                r = run_shape(device, shape, dtype, a.iters, a.rounds, 8.0, exact_fwd)
                r["clock"] = clk.summary()
                results.append(r)
                print(shape, dtype, r["forward"],
                      "rowdev=%.3e" % r["row_sum_max_abs_dev_from_one"],
                      json.dumps({k: round(v, 4) for k, v in r["ms_per_call_median"].items()}),
                      json.dumps({k: "%.2e" % v for k, v in r["rel_l2_vs_true_f64"].items()}),
                      flush=True)
    clock = clk.summary()

    out = {"host": a.host, "card": a.card, "board": a.board,
           "clock_aiclk_during": clock, "clock_line": clk.line(0),
           "float64_reference_fd_check": fd, "results": results}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(clk.line(0))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
