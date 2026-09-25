#!/usr/bin/env python3
"""J1: the LayerNorm backward, `ttnn.moreh_layer_norm_backward` against today's composed closure.

WHAT THE SURVEY GOT WRONG, established by reading the source before spending device time.
`of3t-bwsurvey` sized J1 as "adapt `ttml::metal::layernorm_bw`, which computes the per-row
dgamma/dbeta partials in the kernel and leaves one reduction". It does compute them, but
`layernorm_bw_device_operation.cpp:91-109` gives `dgamma_components` and `dbeta_components` the
INPUT's shape, and `layernorm_bw.cpp:29` then reduces them on the host side with the same
`ttnn::sum` over the flattened leading axis that `_sum_leading` already runs. So tt-train's kernel
does not remove the reduction J1 was briefed to remove -- it removes the norm maths around it.

`ttnn.moreh_layer_norm_backward` does remove it. It ships in the 0.68 wheel, it is bound to Python
(`moreh_layer_norm_backward_nanobind.cpp:20`), and its gamma/beta half reduces over every leading
coordinate inside its own kernel, returning gamma's shape. It was ruled out earlier for needing
`mean` and `rstd` that `ttnn.layer_norm` does not return -- but this backward recomputes both
anyway, so they are handed in.

Both halves are separately optional in that op, and they are parallelised on DIFFERENT axes,
which is why this probe grades them apart:

    input_grad      `split_work_to_cores(grid, num_outer)`  -> one row-tile per core, all cores
    gamma_beta_grad `split_work_to_cores(grid, num_inner)`  -> one WIDTH-tile per core, so
                                                               C = 128 uses 4 cores, C = 384 uses 12

So the dx half should win on both roofs and the dgamma/dbeta half is core-starved by
construction. Arms:

    composed    today's closure out of `tt_bio/autograd.py`, verb-counted
    moreh_dx    moreh for dx, `_sum_leading` for dgamma/dbeta
    moreh_all   moreh for all three

A46.1: the VJP is what is graded. The float64 reference here is validated by central finite
differences on a scalar loss before any arm is scored against it, so a reference that is itself
wrong cannot pass an arm.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "of3t_lnbw"

# (name, shape, C) at the crop-384 trunk. The pair track is c_z = 128 over 384 x 384 tokens,
# which is the [147456, 128] flattening the survey priced; the single track is c_s = 384.
SHAPES = [
    ("fd_small", [1, 2, 64, 128]),
    ("pair_384", [1, 384, 384, 128]),
    ("single_384", [1, 1, 384, 384]),
    ("pair_64", [1, 64, 64, 128]),
]

#: Above this volume the finite-difference validation is skipped and the SAME reference
#: function carries over. The expression does not depend on the shape, so validating it on
#: `fd_small` validates it everywhere; running 144 float64 forwards over 18.9 M elements to
#: re-confirm that would cost minutes of device-session time and settle nothing new.
FD_MAX_VOLUME = 1 << 20

VERBS = ("mean", "subtract", "multiply", "add", "rsqrt", "sum", "reshape", "to_layout",
         "slice", "add_", "deallocate", "moreh_layer_norm_backward", "typecast",
         "to_memory_config", "zeros", "empty", "rsub", "neg")


class Count:
    """Verb calls per ttnn name, for the arm currently running."""

    def __init__(self, ttnn):
        self.ttnn, self.n, self._saved = ttnn, {}, {}

    def __enter__(self):
        for v in VERBS:
            f = getattr(self.ttnn, v, None)
            if f is None:
                continue
            self._saved[v] = f

            def wrap(f=f, v=v):
                def g(*a, **k):
                    self.n[v] = self.n.get(v, 0) + 1
                    return f(*a, **k)
                return g
            setattr(self.ttnn, v, wrap())
        return self

    def __exit__(self, *e):
        for v, f in self._saved.items():
            setattr(self.ttnn, v, f)
        return False

    @property
    def total(self):
        return sum(self.n.values())


def f64_reference(x, g, gamma, beta, eps, *, check=True):
    """dx, dgamma, dbeta in float64, plus the finite-difference validation of all three.

    The loss is <g, layer_norm(x)>, whose gradient in x is exactly the cotangent a tape hands
    this closure, so one scalar covers every output element rather than a sampled row.
    """
    xd, gd, ga, be = (t.double() for t in (x, g, gamma, beta))

    def fwd(x_, ga_, be_):
        mu = x_.mean(-1, keepdim=True)
        v = ((x_ - mu) ** 2).mean(-1, keepdim=True)
        return (x_ - mu) / (v + eps).sqrt() * ga_ + be_

    def loss(x_, ga_, be_):
        return (gd * fwd(x_, ga_, be_)).sum()

    xv = xd.clone().requires_grad_(True)
    gv = ga.clone().requires_grad_(True)
    bv = be.clone().requires_grad_(True)
    loss(xv, gv, bv).backward()
    dx, dgamma, dbeta = xv.grad.clone(), gv.grad.clone(), bv.grad.clone()

    # Central differences on the same float64 loss, and both sides of the comparison are
    # FLOAT64. Built as `torch.tensor(list_of_floats)` they come out float32, every difference
    # below 1e-7 relative rounds to zero, and the check reads a flat 0.0 for all three tensors
    # while testing nothing -- which is what it did on the first run of this file.
    #
    # h is 1e-4 of the tensor's own mean magnitude. Smaller is not better here: the loss is
    # O(1) and a dx entry is O(1e-5), so at h = 1e-6 the difference of the two losses is
    # 1e-11 relative and cancellation, not truncation, sets the floor.
    fd = {}
    rng = np.random.default_rng(0)
    if not check:
        return dx, dgamma, dbeta, {"skipped": "validated on fd_small, same expression"}
    for nm, ten, an in (("dx", xv, dx), ("dgamma", gv, dgamma), ("dbeta", bv, dbeta)):
        flat = ten.detach().reshape(-1)
        idx = rng.choice(flat.numel(), size=min(24, flat.numel()), replace=False)
        h = float(flat.abs().mean()) * 1e-4 + 1e-12
        num, ana = [], an.reshape(-1)[idx].tolist()
        for i in idx:
            up, dn = flat.clone(), flat.clone()
            up[i] += h
            dn[i] -= h
            a = [up.reshape(ten.shape) if nm == "dx" else xd,
                 up.reshape(ten.shape) if nm == "dgamma" else ga,
                 up.reshape(ten.shape) if nm == "dbeta" else be]
            b = [dn.reshape(ten.shape) if nm == "dx" else xd,
                 dn.reshape(ten.shape) if nm == "dgamma" else ga,
                 dn.reshape(ten.shape) if nm == "dbeta" else be]
            num.append(float((loss(*a) - loss(*b)) / (2 * h)))
        num_t = torch.tensor(num, dtype=torch.float64)
        ana_t = torch.tensor(ana, dtype=torch.float64)
        fd[nm] = float((num_t - ana_t).norm() / max(ana_t.norm(), 1e-300))
    return dx, dgamma, dbeta, fd


def _cotenants():
    """Every process holding the card, so a timing can be thrown out rather than believed."""
    r = subprocess.run(["fuser", "-v", "/dev/tenstorrent/0"], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()


def rel(got, ref):
    return float((got.double() - ref).norm() / max(float(ref.norm()), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "probe.json"))
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--dtypes", default="bfloat16,float32")
    ap.add_argument("--only", default="", help="comma-separated cell names; default all")
    ap.add_argument("--arms", default="", help="comma-separated arm names; default all")
    args = ap.parse_args()

    import ttnn
    from perf.bcx_stack.stack import Clock, sysfs_node
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    clock = Clock(dt=0.05)
    DT = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
    eps = 1e-5

    def up(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    # ---------------------------------------------------------------- the three arms
    def stats(xv, g, eps, cfg):
        """mean and rstd, two-pass. Every arm needs them and no arm gets them free:
        `ttnn.layer_norm` returns neither and the forward is production's."""
        mean = ttnn.mean(xv, dim=-1, keepdim=True)
        centered = ttnn.subtract(xv, mean)
        var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                        compute_kernel_config=cfg)
        rstd = ttnn.rsqrt(ttnn.add(var, eps))
        return mean, centered, rstd

    def composed(xv, g, gamma_v, beta_shape, cfg):
        """`autograd.layer_norm`'s closure, verbatim in structure."""
        mean, centered, rstd = stats(xv, g, eps, cfg)
        norm = ttnn.multiply(centered, rstd)
        dgamma = ag._sum_leading(ttnn.multiply(g, norm), gamma_v.shape)
        dbeta = ag._sum_leading(g, beta_shape)
        dnorm = ttnn.multiply(g, gamma_v)
        dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
        dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1, keepdim=True)
        dx = ttnn.multiply(ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                         ttnn.multiply(norm, dn_norm_mean)), rstd)
        return dx, dgamma, dbeta

    def moreh(xv, g, gamma_v, beta_shape, cfg, *, gb):
        """moreh for dx, and for dgamma/dbeta when ``gb``.

        ALL THREE outputs are preallocated, `input_grad` included.
        `moreh_layer_norm_backward.cpp:64-72` only computes a term whose output tensor was handed
        in and pushes `std::nullopt` otherwise -- it does not allocate for you, and the
        `compute_output_specs` fallback that looks like it does is only reached once the prim has
        already been invoked. Left at None the op returns `[None, ...]` and the next verb throws
        `'NoneType' object has no attribute 'storage_type'`, which is how this read on the first
        device run. `ttnn.empty` and not `ttnn.zeros`: these are outputs, the kernel writes every
        element, and a zero-fill of the dx buffer is a second full-size write per node.
        """
        mean, centered, rstd = stats(xv, g, eps, cfg)
        dxo = ttnn.empty(list(xv.shape), dtype=g.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
        gg = bg = None
        if gb:
            gg = ttnn.empty(list(gamma_v.shape), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=dev)
            bg = ttnn.empty(list(beta_shape), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        r = ttnn.moreh_layer_norm_backward(g, xv, mean, rstd, 1, gamma=gamma_v,
                                           input_grad=dxo, gamma_grad=gg, beta_grad=bg,
                                           compute_kernel_config=cfg)
        dx = r[0]
        if gb:
            return dx, r[1], r[2]
        norm = ttnn.multiply(centered, rstd)
        dgamma = ag._sum_leading(ttnn.multiply(g, norm), gamma_v.shape)
        dbeta = ag._sum_leading(g, beta_shape)
        return dx, dgamma, dbeta

    ARMS = {"composed": composed,
            "moreh_dx": lambda *a: moreh(*a, gb=False),
            "moreh_all": lambda *a: moreh(*a, gb=True)}

    blob = {"host": socket.gethostname(), "pci": sysfs_node()[1],
            "visible": os.environ.get("TT_VISIBLE_DEVICES"),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
            "eps": eps, "reps": args.reps, "cotenants": _cotenants(), "cells": []}
    gen = torch.Generator().manual_seed(20260925)

    only = [x for x in args.only.split(",") if x]
    keep = [x for x in args.arms.split(",") if x]
    for name, shape in SHAPES:
        if only and name not in only:
            continue
        C = shape[-1]
        x = torch.randn(shape, generator=gen)
        g = torch.randn(shape, generator=gen) * 1e-2
        gamma = torch.randn([C], generator=gen) * 0.1 + 1.0
        beta = torch.randn([C], generator=gen) * 0.1
        vol = int(np.prod(shape))
        ref_dx, ref_dg, ref_db, fd = f64_reference(x, g, gamma, beta, eps,
                                                   check=vol <= FD_MAX_VOLUME)
        cell = {"name": name, "shape": shape, "C": C, "fd_check": fd, "arms": {}}
        print(f"[{name}] fd check {fd}", flush=True)

        for dtname in args.dtypes.split(","):
            dt = DT[dtname]
            xv, gv = up(x, dt), up(g, dt)
            gam = up(gamma.reshape([1, 1, 1, C]), dt)
            bshape = [1, 1, 1, C]
            cfg = ag.precise_config()
            for arm, fn in ARMS.items():
                if keep and arm not in keep:
                    continue
                key = f"{arm}/{dtname}"
                try:
                    with Count(ttnn) as c:
                        dx, dg, db = fn(xv, gv, gam, bshape, cfg)
                        ttnn.synchronize_device(dev)
                    acc = {"rel_dx": rel(ttnn.to_torch(dx).float(), ref_dx),
                           "rel_dgamma": rel(ttnn.to_torch(dg).float().reshape(C),
                                             ref_dg.reshape(C)),
                           "rel_dbeta": rel(ttnn.to_torch(db).float().reshape(C),
                                            ref_db.reshape(C)),
                           "verbs": c.total, "verbs_by": dict(sorted(c.n.items()))}
                    for t in (dx, dg, db):
                        ttnn.deallocate(t)
                except Exception as e:                                   # noqa: BLE001
                    cell["arms"][key] = {"error": f"{type(e).__name__}: {e}"[:400]}
                    print(f"  {key}: FAILED {type(e).__name__}: {e}"[:300], flush=True)
                    continue

                ts, t_a = [], time.time()
                if args.reps == 0:
                    # Accuracy only. A p150a shared with another row's arms cannot produce a
                    # timing that means anything, and correctness does not care about the
                    # co-tenant, so the two halves of this probe are separable and are
                    # separated rather than reported together with a caveat.
                    cell["arms"][key] = acc
                    print(f"  {key}: verbs {acc['verbs']}  dx {acc['rel_dx']:.2e} "
                          f"dg {acc['rel_dgamma']:.2e} db {acc['rel_dbeta']:.2e}", flush=True)
                    continue
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    outs = fn(xv, gv, gam, bshape, cfg)
                    ttnn.synchronize_device(dev)
                    ts.append(time.perf_counter() - t0)
                    for t in outs:
                        ttnn.deallocate(t)
                acc.update({"min_ms": min(ts) * 1e3,
                            "median_ms": float(statistics.median(ts)) * 1e3,
                            "aiclk": clock.window([(t_a, time.time())]),
                            "load1": os.getloadavg()[0]})
                cell["arms"][key] = acc
                print(f"  {key}: {acc['median_ms']:.3f} ms  verbs {acc['verbs']}  "
                      f"dx {acc['rel_dx']:.2e} dg {acc['rel_dgamma']:.2e} "
                      f"db {acc['rel_dbeta']:.2e} clk {acc['aiclk']}", flush=True)
            for t in (xv, gv, gam):
                ttnn.deallocate(t)
        blob["cells"].append(cell)

    clock.stop()
    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
