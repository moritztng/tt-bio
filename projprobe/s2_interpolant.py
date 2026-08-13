#!/usr/bin/env python3
"""S2 part 1 -- is the separable interpolant as accurate as RELION's trilinear? Host only, fp64.

State doc section 8: the separable form is NOT RELION's trilinear. RELION evaluates the 8 corners of
the one cell containing (xp, yp, zp); the separable form interpolates along z at each neighbouring
lattice column and then in-plane. Section 8 argues they agree except across a floor(z) boundary and
that the separable one may be the better interpolant. That is an argument, not evidence.

The test is built so that a TRUE continuum value exists, which is stronger than comparing the two
interpolants to each other. The reference volume is a sum of Gaussian atoms, so its Fourier
transform is analytic:
    F(q) = sum_j f_j exp(-2 pi i q.r_j) exp(-2 pi^2 sigma^2 |q|^2)
evaluable at any real coordinate. Atom positions fill a sphere of radius P/4 in a padded box of
P = 2N, which is RELION's padding_factor = 2: the object occupies half the padded box, so the
Fourier volume's correlation length is about 2 lattice units. That is why RELION pads, and it also
means trilinear's own absolute error at padding_factor 2 is percent-level rather than negligible --
so the quantity that matters is the RATIO of the separable form's error to trilinear's, not either
one alone.

THE AXIS PERMUTATION IS PART OF THE MEASUREMENT, NOT AN OPTIMISATION. Section 4.3 requires three
axis-permuted volume copies so the plane normal's largest component is always z; without it the 2x2
in-plane system is near-singular for about a third of orientations, (a, b) blow up, and the measured
error is dominated by an artifact the design explicitly excludes. A first run of this screen omitted
the permutation and reported a 2.01 max relative error at box 256 for exactly that reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NATOM = 300
SIGMA = 1.0          # Gaussian atom width, padded-box voxels
NORIENT = 64
SHELLS = 8


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
    """RELION's LIN_INTERP: the 8 corners of the single cell containing p, in fp64."""
    p0 = np.floor(p).astype(np.int64)
    fr = p - p0
    out = np.zeros(p.shape[:-1], dtype=np.complex128)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((1 - fr[..., 0]) if dx == 0 else fr[..., 0]) \
                    * ((1 - fr[..., 1]) if dy == 0 else fr[..., 1]) \
                    * ((1 - fr[..., 2]) if dz == 0 else fr[..., 2])
                c = p0 + np.array([dx, dy, dz])
                out += w * F(pos, f, P, c.astype(np.float64))
    return out


def separable(pos, f, P, uu, vv, u3, v3):
    """z-collapse on the volume's own (X, Y) lattice, then true 2D bilinear in-plane.

    Stage 1: W(X, Y) = the plane's value at integer lattice column (X, Y), linear along z, with
             z = a*X + b*Y from the 2x2 in-plane solve.
    Stage 2: bilinear sample of W at the affine image of the output pixel.
    Caller has already applied the axis permutation, so |a|, |b| <= 1 by construction.
    """
    Xt = u3[0] * uu + v3[0] * vv
    Yt = u3[1] * uu + v3[1] * vv
    M = np.array([[u3[0], v3[0]], [u3[1], v3[1]]])
    ab = np.array([u3[2], v3[2]]) @ np.linalg.inv(M)
    a, b = ab[0], ab[1]

    def W(X, Y):
        Z = a * X + b * Y
        Z0 = np.floor(Z)
        t = Z - Z0
        q0 = np.stack([X.astype(np.float64), Y.astype(np.float64), Z0], -1)
        q1 = np.stack([X.astype(np.float64), Y.astype(np.float64), Z0 + 1], -1)
        return (1 - t) * F(pos, f, P, q0) + t * F(pos, f, P, q1)

    X0, Y0 = np.floor(Xt), np.floor(Yt)
    fx, fy = Xt - X0, Yt - Y0
    acc = 0.0
    for dy in (0, 1):
        wy = (1 - fy) if dy == 0 else fy
        for dx in (0, 1):
            wx = (1 - fx) if dx == 0 else fx
            acc = acc + wx * wy * W(X0 + dx, Y0 + dy)
    return acc, abs(a), abs(b)


def shell_l2(err, ref, rad, nsh):
    out = []
    edges = np.linspace(0, rad.max() + 1e-9, nsh + 1)
    for i in range(nsh):
        m = (rad >= edges[i]) & (rad < edges[i + 1])
        if m.sum() < 8:
            out.append(None)
            continue
        out.append(float(np.linalg.norm(err[m]) / max(np.linalg.norm(ref[m]), 1e-300)))
    return out


def main():
    rng = np.random.default_rng(7)
    res = {"natom": NATOM, "sigma": SIGMA, "norient": NORIENT, "shells": SHELLS,
           "note": "true separable two-pass (Catmull-Smith) arm is OWED, not measured here",
           "boxes": {}}
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
        acc = {"tri": [], "sep2d": []}
        sh = {"tri": [], "sep2d": [], "sep2d_vs_tri": []}
        abmax = []
        for o in range(NORIENT):
            A = rand_rot(rng)
            # RELION: xp = (Ainv[0,0]*x + Ainv[0,1]*y) * padding_factor, padding_factor = 2.
            u3, v3 = 2.0 * A[:, 0], 2.0 * A[:, 1]
            n = A[:, 2]
            # The section-4.3 axis permutation: largest |normal component| becomes z. Applied to the
            # basis vectors AND to the atom coordinates, so the dot product q.r -- and therefore the
            # truth -- is unchanged. This is a relabelling of axes, not a change of problem.
            perm = np.argsort(np.abs(n))
            u3p, v3p, posp = u3[perm], v3[perm], pos[:, perm]
            p = uu[:, None] * u3[None, :] + vv[:, None] * v3[None, :]
            truth = F(pos, f, P, p)
            tri = trilinear(pos, f, P, p)
            sep, aa, bb = separable(posp, f, P, uu, vv, u3p, v3p)
            abmax.append(max(aa, bb))
            nt = np.linalg.norm(truth)
            for k, v in (("tri", tri), ("sep2d", sep)):
                acc[k].append(float(np.linalg.norm(v - truth) / nt))
                sh[k].append(shell_l2(v - truth, truth, rad, SHELLS))
            sh["sep2d_vs_tri"].append(shell_l2(sep - tri, truth, rad, SHELLS))
        agg = lambda L: [float(np.mean([r[i] for r in L if r[i] is not None])) for i in range(SHELLS)]
        res["boxes"][str(N)] = {
            "npoints": int(keep.sum()), "step": int(step),
            "max_abs_a_or_b": float(np.max(abmax)),
            "rel_l2_vs_truth": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))}
                                for k, v in acc.items()},
            "ratio_sep_over_tri": float(np.mean(acc["sep2d"]) / np.mean(acc["tri"])),
            "shell_rel_l2": {k: agg(v) for k, v in sh.items()},
        }
        print(f"box {N} (P={P}, {int(keep.sum())} pts x {NORIENT} orientations, "
              f"max|a|,|b| = {np.max(abmax):.3f})")
        for k, lab in (("tri", "RELION trilinear"), ("sep2d", "separable       ")):
            print(f"   {lab} vs truth: rel L2 mean {np.mean(acc[k]):.4e}  max {np.max(acc[k]):.4e}")
        print(f"   ratio separable / RELION = {np.mean(acc['sep2d'])/np.mean(acc['tri']):.3f}x")
        for k in ("tri", "sep2d", "sep2d_vs_tri"):
            print(f"   shells {k:12s}: " + " ".join(f"{x:.2e}" for x in agg(sh[k])))
        json.dump(res, open(HERE / "s2_interpolant.json", "w"), indent=1)

    print("\n--- S2 part 1 verdict ---")
    for N in (256, 384, 512):
        r = res["boxes"][str(N)]["ratio_sep_over_tri"]
        print(f"box {N}: separable costs {r:.3f}x RELION trilinear's own error against the "
              f"continuum -> {'matches the incumbent' if r <= 1.05 else 'worse than the incumbent'}")
    json.dump(res, open(HERE / "s2_interpolant.json", "w"), indent=1)


main()
