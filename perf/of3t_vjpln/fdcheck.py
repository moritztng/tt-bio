#!/usr/bin/env python3
"""Validate `refvjp.py` against CENTRAL FINITE DIFFERENCES in float64, before any device arm.

A VJP written against a dispatcher, or with two operand roles swapped, tests nothing, and no
output comparison downstream says so: the arm fires, the numbers move, and the verdict is about
the wrong object. So each VJP is checked here against the only reference that does not share its
own algebra -- differences of the forward itself.

THE TEST. For a function f and a cotangent g, the VJP claims
    <g, f'(theta) v>  ==  <VJP(g), v>   for every direction v.
The left side is evaluated by central differences, `<g, (f(theta + h v) - f(theta - h v))/(2h)>`,
which is second-order accurate and, unlike a one-sided difference, does not confuse a first
derivative with a curvature term. Each operand is driven ALONE as well as all together, because
a role mix-up between `dw` and `dbias` is invisible when every direction is excited at once.

Two independent references, and they answer different questions:
  FD    -- differences of the forward. The one the brief requires, and the one that cannot share
           an algebra error with the VJP.
  AUTOGRAD -- `torch.autograd.grad` on the same forward. Exact to roundoff, so it separates "the
           VJP is wrong" from "h was badly chosen"; a disagreement here is always the VJP.

WHAT THIS DOES NOT PROVE. Both references differentiate the forward THIS MODULE writes. They
cannot tell you that forward is the one the device computes. That question is answered on device
in `census.py`, which compares `refvjp.layer_norm_forward` and `refvjp.linear_forward` against
the shipped `out_v` of every live firing.

h is 1e-6 relative to each operand's own RMS. In float64 central differences the truncation term
falls as h^2 and the cancellation term grows as eps/h, so the floor is about eps^(2/3) ~ 4e-11
and a well-formed VJP should land near it. Anything at 1e-3 or 1 is a formula error, not h.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refvjp as R                                                     # noqa: E402

torch.manual_seed(20260922)
EPS_LN = 1e-5


def _dot(a, b):
    return float((a.reshape(-1) * b.reshape(-1)).sum())


def _pair(vjp, fwd, thetas, g, names, h_rel=1e-6):
    """One (function, operand-subset) check. Returns FD and autograd agreements."""
    live = [i for i, t in enumerate(thetas) if t is not None]
    rows = []
    subsets = [[i] for i in live] + ([live] if len(live) > 1 else [])
    for sub in subsets:
        vs = [torch.randn_like(t) if (t is not None and i in sub) else None
              for i, t in enumerate(thetas)]
        rhs = 0.0
        for i in sub:
            if vjp[i] is not None:
                rhs += _dot(vjp[i], vs[i])
        hs = [(h_rel * float(thetas[i].pow(2).mean().sqrt()) if thetas[i] is not None else 0.0)
              for i in range(len(thetas))]
        plus = [(t + hs[i] * vs[i]) if (t is not None and vs[i] is not None) else t
                for i, t in enumerate(thetas)]
        minus = [(t - hs[i] * vs[i]) if (t is not None and vs[i] is not None) else t
                 for i, t in enumerate(thetas)]
        # one h per operand, so the directional derivative is sum_i h_i * d/dtheta_i . v_i;
        # divide each side by its own h by scaling v instead. Simplest correct form: scale the
        # direction by h and use a single step of size 1.
        d = (fwd(*plus) - fwd(*minus))
        lhs_raw = _dot(g, d) / 2.0
        # lhs_raw = sum_i h_i <g, J_i v_i>; rhs must be scaled the same way
        rhs_scaled = 0.0
        for i in sub:
            if vjp[i] is not None:
                rhs_scaled += hs[i] * _dot(vjp[i], vs[i])
        fd_rel = abs(lhs_raw - rhs_scaled) / max(abs(rhs_scaled), 1e-300)

        # autograd on the same forward
        ths = [(t.clone().requires_grad_(True) if t is not None else None) for t in thetas]
        out = fwd(*ths)
        gr = torch.autograd.grad(out, [ths[i] for i in live], g, allow_unused=True)
        ag = {live[k]: gr[k] for k in range(len(live))}
        ag_rel = 0.0
        for i in live:
            a, b = ag.get(i), vjp[i]
            if a is None and b is None:
                continue
            if a is None or b is None:
                ag_rel = float("inf")
                continue
            nb = float(torch.linalg.vector_norm(b.reshape(-1)))
            ag_rel = max(ag_rel, float(torch.linalg.vector_norm(
                (a - b).reshape(-1))) / max(nb, 1e-300))
        rows.append({"operands": [names[i] for i in sub],
                     "fd_rel_err": fd_rel, "autograd_rel_err": ag_rel})
    return rows


def check_layer_norm(shape, c, with_gamma=True, with_beta=True):
    x = torch.randn(*shape, dtype=torch.float64) * 3.0 + 0.7
    gamma = torch.randn(c, dtype=torch.float64) if with_gamma else None
    beta = torch.randn(c, dtype=torch.float64) if with_beta else None
    g = torch.randn(*shape, dtype=torch.float64)
    vjp = list(R.layer_norm_vjp(x, gamma, beta, g, EPS_LN))
    fwd = lambda a, b, cc: R.layer_norm_forward(a, b, cc, EPS_LN)      # noqa: E731
    return _pair(vjp, fwd, [x, gamma, beta], g, ["x", "gamma", "beta"])


def check_linear(xshape, cin, cout, with_bias=True, drop_rank=False):
    x = torch.randn(*xshape, dtype=torch.float64)
    w = torch.randn(cin, cout, dtype=torch.float64)
    bias = torch.randn(cout, dtype=torch.float64) if with_bias else None
    gshape = list(xshape[:-1]) + [cout]
    if drop_rank:
        gshape = gshape[1:]                # ttnn normalises (1,N,N,c) -> (N,N,c)
    g = torch.randn(*gshape, dtype=torch.float64)
    vjp = list(R.linear_vjp(x, w, bias, g))
    def fwd(a, b, cc):
        out = R.linear_forward(a, b, cc)
        return out.reshape(gshape) if drop_rank else out
    return _pair(vjp, fwd, [x, w, bias], g, ["x", "w", "bias"])


def main() -> int:
    cases = []
    # shapes taken from of3t-blk4544's own backward census at padded 64 (COVERAGE_64.json),
    # scaled down on the token axis so this runs on a host in seconds; the arithmetic under
    # test does not depend on the extent.
    for nm, rows in (
        ("layer_norm [1,N,N,128]", check_layer_norm((1, 9, 9, 128), 128)),
        ("layer_norm [N,N,128]", check_layer_norm((9, 9, 128), 128)),
        ("layer_norm [1,N,384]", check_layer_norm((1, 9, 384), 384)),
        ("layer_norm no beta", check_layer_norm((1, 9, 9, 128), 128, with_beta=False)),
        ("layer_norm no gamma/beta", check_layer_norm((1, 9, 9, 128), 128, False, False)),
        ("linear [1,N,N,128]->512", check_linear((1, 9, 9, 128), 128, 512)),
        ("linear [N,N,128]->384", check_linear((9, 9, 128), 128, 384)),
        ("linear [1,N,N,128]->16 no bias", check_linear((1, 9, 9, 128), 128, 16, False)),
        ("linear [1,N,1536]->384", check_linear((1, 9, 1536), 1536, 384)),
        ("linear rank-normalised g", check_linear((1, 9, 9, 128), 128, 128, True, True)),
    ):
        for r in rows:
            cases.append({"case": nm, **r})

    worst_fd = max(c["fd_rel_err"] for c in cases)
    worst_ag = max(c["autograd_rel_err"] for c in cases)
    print("%-34s %-22s %12s %12s" % ("case", "operands", "fd_rel", "autograd_rel"))
    for c in cases:
        print("%-34s %-22s %12.3e %12.3e"
              % (c["case"], ",".join(c["operands"]), c["fd_rel_err"], c["autograd_rel_err"]))
    print()
    print("WORST central-FD relative disagreement   %.6e" % worst_fd)
    print("WORST autograd relative disagreement     %.6e" % worst_ag)
    ok = worst_fd < 1e-7 and worst_ag < 1e-12
    print("VERDICT:", "PASS" if ok else "FAIL")
    out = {"h_relative": 1e-6, "eps_layer_norm": EPS_LN,
           "worst_fd_rel_err": worst_fd, "worst_autograd_rel_err": worst_ag,
           "pass_bar": {"fd": 1e-7, "autograd": 1e-12}, "pass": ok, "cases": cases}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", sys.argv[1])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
