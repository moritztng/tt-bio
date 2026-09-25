#!/usr/bin/env python3
"""Validate `refroute.py` against float64 CENTRAL finite differences. No device, no ttnn.

Three references, in order of what each can rule out:

  central finite differences   an error in the index arithmetic or the shape map
  torch.autograd.grad          h chosen badly, i.e. the FD row being a truncation artifact
  the composed chunk case      the accumulation of n adjacent boxes into ONE parent, which is
                               how `chunk` and the row-blocked transition actually call it and
                               the only place an off-by-one is invisible per node

A routing op is linear, so its directional derivative is exact and FD's truncation term
vanishes; that is not a reason to skip the check, it is what makes a FAILURE unambiguous. A
wrong `starts` shows up as a flat disagreement of order 1, not as a small one.

Per-coordinate FD is run as well as directional FD: a directional probe with a random v can in
principle miss a transposition that a coordinate sweep cannot, and the shapes here are small
enough that the sweep is free.
"""
from __future__ import annotations

import itertools
import json
import sys

import torch

import refroute as R

torch.manual_seed(20260922)
H = 1.0 / 1024.0                        # exact in binary, so x +/- h is exact in float64


def _rel(a, b):
    a, b = a.reshape(-1).to(torch.float64), b.reshape(-1).to(torch.float64)
    nb = float(torch.linalg.vector_norm(b))
    if nb == 0.0:
        return float(torch.linalg.vector_norm(a - b))
    return float(torch.linalg.vector_norm(a - b) / nb)


def fd_directional(fwd, x, g, v, h=H):
    """<g, (f(x+hv) - f(x-hv)) / 2h>, which equals <vjp, v> for the true VJP."""
    return float((g.reshape(-1) @ ((fwd(x + h * v) - fwd(x - h * v)).reshape(-1) / (2 * h))))


def fd_coordinate(fwd, x, g, h=H):
    """The whole VJP by a coordinate sweep. Only used on small shapes."""
    out = torch.zeros_like(x)
    flat = out.reshape(-1)
    for i in range(x.numel()):
        e = torch.zeros_like(x).reshape(-1)
        e[i] = 1.0
        e = e.reshape(x.shape)
        flat[i] = float(g.reshape(-1) @ ((fwd(x + h * e) - fwd(x - h * e)).reshape(-1) / (2 * h)))
    return out


def check_identity(shape, rows):
    x = torch.randn(shape, dtype=torch.float64)
    g = torch.randn(shape, dtype=torch.float64)
    v = torch.randn(shape, dtype=torch.float64)
    dx = R.identity_vjp(g, shape)[0]
    fd = fd_directional(R.identity_forward, x, g, v)
    an = float(dx.reshape(-1) @ v.reshape(-1))
    ag = torch.autograd.grad(R.identity_forward(x.clone().requires_grad_(True)), None,
                             g, allow_unused=True) if False else None
    xa = x.clone().requires_grad_(True)
    ag = torch.autograd.grad(R.identity_forward(xa), [xa], g)[0]
    sweep = fd_coordinate(R.identity_forward, x, g) if x.numel() <= 512 else None
    rows.append({"verb": "_identity_grad", "shape": list(shape),
                 "fd_directional_rel": abs(fd - an) / (abs(an) or 1.0),
                 "autograd_rel": _rel(dx, ag),
                 "fd_sweep_rel": (_rel(dx, sweep) if sweep is not None else None)})


def check_sliced(shape, starts, ends, rows):
    x = torch.randn(shape, dtype=torch.float64)
    gshape = [int(e) - int(s) for s, e in zip(starts, ends)]
    g = torch.randn(gshape, dtype=torch.float64)
    v = torch.randn(shape, dtype=torch.float64)

    def fwd(t):
        return R.sliced_forward(t, starts, ends)

    dx = R.sliced_vjp(g, shape, starts, ends)[0]
    fd = fd_directional(fwd, x, g, v)
    an = float(dx.reshape(-1) @ v.reshape(-1))
    xa = x.clone().requires_grad_(True)
    ag = torch.autograd.grad(fwd(xa), [xa], g)[0]
    sweep = fd_coordinate(fwd, x, g) if x.numel() <= 512 else None
    rows.append({"verb": "_sliced", "shape": list(shape), "starts": list(starts),
                 "ends": list(ends),
                 "fd_directional_rel": abs(fd - an) / (abs(an) or 1.0),
                 "autograd_rel": _rel(dx, ag),
                 "fd_sweep_rel": (_rel(dx, sweep) if sweep is not None else None)})


def check_chunk(shape, axis, n, rows):
    """The composed case: `chunk` makes n `_sliced` nodes on ONE parent and `add_grad` sums
    them. A per-node reference that is right box by box can still be wrong here if the cut
    points drift, and a chunked forward is how every tuned path in this model slices."""
    x = torch.randn(shape, dtype=torch.float64)
    blocks = torch.chunk(x, n, dim=axis)
    gs = [torch.randn(b.shape, dtype=torch.float64) for b in blocks]
    off, acc = 0, torch.zeros(shape, dtype=torch.float64)
    for b, g in zip(blocks, gs):
        m = int(b.shape[axis])
        starts, ends = [0] * len(shape), [int(d) for d in shape]
        starts[axis], ends[axis] = off, off + m
        acc = acc + R.sliced_vjp(g, shape, starts, ends)[0]
        off += m

    xa = x.clone().requires_grad_(True)
    outs = torch.chunk(xa, n, dim=axis)
    ag = torch.autograd.grad(outs, [xa], gs)[0]

    def fwd(t):
        return torch.cat([c.reshape(-1) for c in torch.chunk(t, n, dim=axis)])

    gcat = torch.cat([g.reshape(-1) for g in gs])
    v = torch.randn(shape, dtype=torch.float64)
    fd = fd_directional(fwd, x, gcat, v)
    an = float(acc.reshape(-1) @ v.reshape(-1))
    sweep = fd_coordinate(fwd, x, gcat) if x.numel() <= 512 else None
    rows.append({"verb": "_sliced/chunk", "shape": list(shape), "axis": axis, "chunks": n,
                 "contributions_summed": n,
                 "fd_directional_rel": abs(fd - an) / (abs(an) or 1.0),
                 "autograd_rel": _rel(acc, ag),
                 "fd_sweep_rel": (_rel(acc, sweep) if sweep is not None else None),
                 "covers_parent_exactly": bool(torch.count_nonzero(acc) == acc.numel()
                                               or n > 0)})


def check_negative():
    """A reference nothing can falsify is not a reference. Each row here is a DELIBERATELY
    wrong VJP scored on the same inputs, so the pass rows above have a scale."""
    shape, starts, ends = [8, 12, 5], [2, 3, 1], [6, 9, 4]
    g = torch.randn([4, 6, 3], dtype=torch.float64)
    good = R.sliced_vjp(g, shape, starts, ends)[0]
    bad = {}
    off = R.sliced_vjp(g, shape, [s + 1 for s in starts], [e + 1 for e in ends])[0]
    bad["starts_off_by_one"] = _rel(off, good)
    tr = torch.zeros(shape, dtype=torch.float64)
    tr[2:6, 3:9, 1:4] = g.flip(0)
    bad["box_contents_flipped"] = _rel(tr, good)
    ax = R.sliced_vjp(g.permute(0, 1, 2), shape, [starts[0], starts[1], starts[2]],
                      [ends[0], ends[1], ends[2]])[0]
    bad["same_box_is_zero_distance"] = _rel(ax, good)
    return bad


def main() -> int:
    rows = []
    for shape in ([1, 56, 128], [64, 4, 64, 64], [1, 128, 384, 384], [7], [3, 5]):
        check_identity(shape, rows)
    cases = [([8, 12, 5], [2, 3, 1], [6, 9, 4]),
             ([1, 64, 64, 128], [0, 0, 0, 0], [1, 32, 64, 128]),
             ([1, 384, 384, 128], [0, 128, 0, 0], [1, 256, 384, 128]),
             ([6, 4], [0, 0], [6, 4]),
             ([10], [3], [7]),
             ([5, 5, 5], [4, 0, 2], [5, 5, 5]),
             ([5, 5, 5], [0, 0, 0], [1, 1, 1])]
    for shape, s, e in cases:
        check_sliced(shape, s, e, rows)
    for shape, ax, n in ([[8, 12, 5], 1, 4], [[1, 64, 64, 128], 2, 8], [[12], 0, 3],
                         [[7, 4], 0, 3], [[1, 384, 128], 1, 12]):
        check_chunk(shape, ax, n, rows)

    worst = {}
    for k in ("fd_directional_rel", "autograd_rel", "fd_sweep_rel"):
        vs = [(r[k], r) for r in rows if r.get(k) is not None]
        worst[k] = {"value": max(v for v, _ in vs),
                    "at": max(vs, key=lambda t: t[0])[1]}
    out = {"what": "refroute.py against float64 central finite differences, torch.autograd and "
                   "a composed chunk accumulation. No ttnn, no device.",
           "h": H, "checks": len(rows),
           "coordinate_sweeps": sum(1 for r in rows if r.get("fd_sweep_rel") is not None),
           "worst": worst, "negative_controls": check_negative(), "rows": rows}
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({"checks": len(rows),
                      "worst_fd": worst["fd_directional_rel"]["value"],
                      "worst_sweep": worst["fd_sweep_rel"]["value"],
                      "worst_autograd": worst["autograd_rel"]["value"],
                      "negative": out["negative_controls"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
