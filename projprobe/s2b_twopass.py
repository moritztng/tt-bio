#!/usr/bin/env python3
"""S2 part 1b -- the TRUE Catmull-Smith two-pass interpolant, which is the kernel's actual form.

s2_interpolant.py measured the separable family's FLOOR: z-collapse then a true 2D bilinear in-plane
sample, 1.16-1.21x RELION trilinear's own error. The kernel does not do a 2D bilinear. Section 4.3's
stage 2 is two separable 1D passes, and a two-pass affine resample is a THIRD interpolant: its
intermediate lives on the mixed lattice (x_out, Y_source), so the x-fraction differs between the two
y-neighbours in a way a 2D bilinear's does not.

Catmull-Smith, order A (resample X first, pivot v1). For output (x, y),
    Xt = u0*x + v0*y,   Yt = u1*x + v1*y
Pass 1 builds the intermediate on (x, Y) for integer Y: invert the second equation for y and
substitute, y = (Y - u1*x)/v1, giving
    I(x, Y) = lerp_X( W(., Y) at alpha*x + beta*Y ),   alpha = u0 - v0*u1/v1,  beta = v0/v1
Pass 2 resamples along Y at the true target: out = lerp_Y( I(x, .) at u1*x + v1*y ).
Order B exchanges the axes, pivot u0.

ALL FOUR CANDIDATES ARE TRIED, and that is not an optimisation. Order A needs |v1| and order B needs
|u0|, and for a rotation submatrix both can be small at once. The other half of Catmull-Smith is that
relabelling the OUTPUT axes x <-> y is free -- it is a transpose of the output tile, a native tile op --
and it swaps u3 with v3, giving two more candidates with pivots u1 and v0. Taking the largest of the
four is the bottleneck rule applied properly. A first version of this arm tried only two and hit a
1.06 max relative error on exactly the orientations where neither pivot was usable.

Mean, median and p95 are all reported, because a handful of near-degenerate orientations can move a
mean a long way and the mean alone would not say whether that had happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NATOM, SIGMA, NORIENT, SHELLS = 300, 1.0, 64, 8


def atoms(P, rng):
    v = rng.normal(size=(NATOM, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = (P / 4.0) * rng.uniform(size=(NATOM, 1)) ** (1.0 / 3.0)
    return v * r, rng.uniform(0.5, 1.5, size=NATOM)


def F(pos, f, P, q):
    qs = q.reshape(-1, 3) / P
    ph = -2.0 * np.pi * (qs @ pos.T)
    env = np.exp(-2.0 * np.pi ** 2 * SIGMA ** 2 * np.sum(qs * qs, axis=1))
    val = (np.cos(ph) + 1j * np.sin(ph)) @ f.astype(np.complex128)
    return (val * env).reshape(q.shape[:-1])


def rand_rot(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def trilinear(pos, f, P, p):
    p0 = np.floor(p).astype(np.int64)
    fr = p - p0
    out = np.zeros(p.shape[:-1], dtype=np.complex128)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((1 - fr[..., 0]) if dx == 0 else fr[..., 0]) \
                    * ((1 - fr[..., 1]) if dy == 0 else fr[..., 1]) \
                    * ((1 - fr[..., 2]) if dz == 0 else fr[..., 2])
                out += w * F(pos, f, P, (p0 + np.array([dx, dy, dz])).astype(np.float64))
    return out


def make_W(pos, f, P, u3, v3):
    """Stage 1: the plane's value at an integer (X, Y) lattice column, linear along z."""
    M = np.array([[u3[0], v3[0]], [u3[1], v3[1]]])
    ab = np.array([u3[2], v3[2]]) @ np.linalg.inv(M)
    a, b = ab[0], ab[1]

    def W(X, Y):
        Z = a * X + b * Y
        Z0 = np.floor(Z)
        t = Z - Z0
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        return ((1 - t) * F(pos, f, P, np.stack([X, Y, Z0], -1))
                + t * F(pos, f, P, np.stack([X, Y, Z0 + 1], -1)))
    return W


def sep2d(W, uu, vv, u3, v3):
    Xt, Yt = u3[0] * uu + v3[0] * vv, u3[1] * uu + v3[1] * vv
    X0, Y0 = np.floor(Xt), np.floor(Yt)
    fx, fy = Xt - X0, Yt - Y0
    acc = 0.0
    for dy in (0, 1):
        wy = (1 - fy) if dy == 0 else fy
        for dx in (0, 1):
            wx = (1 - fx) if dx == 0 else fx
            acc = acc + wx * wy * W(X0 + dx, Y0 + dy)
    return acc


def twopass(W, uu, vv, u3, v3):
    """Two-pass with the best of the four Catmull-Smith candidates. Returns (values, pivot)."""
    cands = [("A", 0, abs(v3[1])), ("B", 0, abs(u3[0])),
             ("A", 1, abs(u3[1])), ("B", 1, abs(v3[0]))]
    order, swap, piv = max(cands, key=lambda c: c[2])
    # Swapping the output axis labels swaps the two basis vectors AND the two output coordinates.
    a3, b3 = (v3, u3) if swap else (u3, v3)
    s_uu, s_vv = (vv, uu) if swap else (uu, vv)
    u0, u1, v0, v1 = a3[0], a3[1], b3[0], b3[1]

    if order == "A":
        alpha, beta = u0 - v0 * u1 / v1, v0 / v1
        Yr = u1 * s_uu + v1 * s_vv
        Y0 = np.floor(Yr)
        ty = Yr - Y0
        out = 0.0
        for dy in (0, 1):
            Y = Y0 + dy
            wy = (1 - ty) if dy == 0 else ty
            # Xr depends on Y. That dependence is exactly what makes two-pass a third interpolant.
            Xr = alpha * s_uu + beta * Y
            X0 = np.floor(Xr)
            tx = Xr - X0
            wxy = (1 - tx) * W(X0, Y) + tx * W(X0 + 1, Y)
            out = out + wy * wxy
        return out, piv

    gamma, delta = u1 / u0, v1 - v0 * u1 / u0
    Xr = u0 * s_uu + v0 * s_vv
    X0 = np.floor(Xr)
    tx = Xr - X0
    out = 0.0
    for dx in (0, 1):
        X = X0 + dx
        wx = (1 - tx) if dx == 0 else tx
        Yr = gamma * X + delta * s_vv
        Y0 = np.floor(Yr)
        ty = Yr - Y0
        wxy = (1 - ty) * W(X, Y0) + ty * W(X, Y0 + 1)
        out = out + wx * wxy
    return out, piv


def shell_l2(err, ref, rad, nsh):
    out = []
    edges = np.linspace(0, rad.max() + 1e-9, nsh + 1)
    for i in range(nsh):
        m = (rad >= edges[i]) & (rad < edges[i + 1])
        out.append(None if m.sum() < 8
                   else float(np.linalg.norm(err[m]) / max(np.linalg.norm(ref[m]), 1e-300)))
    return out


def stats(v):
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}


def main():
    rng = np.random.default_rng(7)
    res = {"natom": NATOM, "sigma": SIGMA, "norient": NORIENT, "candidates": 4, "boxes": {}}
    for N in (256, 384, 512):
        P = 2 * N
        pos, f = atoms(P, rng)
        rmax = N // 2
        step = max(1, rmax // 40)
        g = np.arange(-rmax, rmax + 1, step)
        ux, vy = np.meshgrid(g, g, indexing="ij")
        rad2 = ux ** 2 + vy ** 2
        keep = rad2 <= rmax ** 2
        uu, vv = ux[keep].astype(np.float64), vy[keep].astype(np.float64)
        rad = np.sqrt(rad2[keep]).astype(np.float64)
        acc = {"tri": [], "sep2d": [], "twopass": []}
        sh = {"tri": [], "sep2d": [], "twopass": [], "twopass_vs_tri": []}
        pivots = []
        for o in range(NORIENT):
            A = rand_rot(rng)
            u3, v3 = 2.0 * A[:, 0], 2.0 * A[:, 1]
            perm = np.argsort(np.abs(A[:, 2]))
            u3p, v3p, posp = u3[perm], v3[perm], pos[:, perm]
            p = uu[:, None] * u3[None, :] + vv[:, None] * v3[None, :]
            truth = F(pos, f, P, p)
            W = make_W(posp, f, P, u3p, v3p)
            tp, piv = twopass(W, uu, vv, u3p, v3p)
            pivots.append(piv)
            got = {"tri": trilinear(pos, f, P, p), "sep2d": sep2d(W, uu, vv, u3p, v3p),
                   "twopass": tp}
            nt = np.linalg.norm(truth)
            for k, v in got.items():
                acc[k].append(float(np.linalg.norm(v - truth) / nt))
                sh[k].append(shell_l2(v - truth, truth, rad, SHELLS))
            sh["twopass_vs_tri"].append(shell_l2(got["twopass"] - got["tri"], truth, rad, SHELLS))
        agg = lambda L: [float(np.mean([r[i] for r in L if r[i] is not None])) for i in range(SHELLS)]
        res["boxes"][str(N)] = {
            "npoints": int(keep.sum()),
            "pivot_min": float(np.min(pivots)), "pivot_mean": float(np.mean(pivots)),
            "rel_l2_vs_truth": {k: stats(v) for k, v in acc.items()},
            "ratio_mean_over_tri": {k: float(np.mean(acc[k]) / np.mean(acc["tri"]))
                                    for k in ("sep2d", "twopass")},
            "ratio_median_over_tri": {k: float(np.median(acc[k]) / np.median(acc["tri"]))
                                      for k in ("sep2d", "twopass")},
            "shell_rel_l2": {k: agg(v) for k, v in sh.items()},
        }
        print(f"box {N} (P={P}, {int(keep.sum())} pts x {NORIENT} orientations, "
              f"min pivot {np.min(pivots):.3f}, mean pivot {np.mean(pivots):.3f})")
        for k, lab in (("tri", "RELION trilinear "), ("sep2d", "separable 2D     "),
                       ("twopass", "two-pass (kernel)")):
            r = "" if k == "tri" else f"  = {np.mean(acc[k])/np.mean(acc['tri']):.3f}x tri (mean), " \
                                     f"{np.median(acc[k])/np.median(acc['tri']):.3f}x (median)"
            print(f"   {lab} mean {np.mean(acc[k]):.4e}  median {np.median(acc[k]):.4e}  "
                  f"p95 {np.percentile(acc[k],95):.4e}  max {np.max(acc[k]):.4e}{r}")
        print("   shells two-pass: " + " ".join(f"{x:.2e}" for x in agg(sh["twopass"])))
        json.dump(res, open(HERE / "s2b_twopass4.json", "w"), indent=1)

    print("\n--- S2 part 1b verdict: the kernel's actual interpolant, 4-candidate pivoting ---")
    for N in (256, 384, 512):
        b = res["boxes"][str(N)]
        print(f"box {N}: two-pass {b['ratio_mean_over_tri']['twopass']:.3f}x trilinear on the mean, "
              f"{b['ratio_median_over_tri']['twopass']:.3f}x on the median "
              f"(separable-2D floor {b['ratio_mean_over_tri']['sep2d']:.3f}x)")
    json.dump(res, open(HERE / "s2b_twopass4.json", "w"), indent=1)


main()
