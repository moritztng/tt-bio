#!/usr/bin/env python3
"""How wrong is the shear interpolant on RELION's own inputs, and does it move RELION's answer?

Every interpolant figure in this lineage is a synthetic-volume study: projprobe/s2_interpolant.py
builds a Gaussian-atom volume so a true continuum value exists, and s2p2_fsc.py FSCs a whole
project/backproject pipeline against a known truth. Both are the right test for "which interpolant
is more accurate in principle". Neither answers "how far does the shear move RELION's coarse score
on RELION's own padded model, at RELION's own operating point", which is the input state doc 4.11's
transfer function needs and explicitly does not have: it measured how much reference error the
argmin tolerates (3%), not how much the shear actually makes.

Inputs come from a live relion_refine_mpi call, dumped through the bridge (TT_RELION_DUMP), so the
model, the orientation set, the image, the CTF/noise weights and RELION's own diff2 are all real.

trilinear  = tt_bio.cryoem.relion._project, already verified bit-identical to RELION's answer on
             all 4,452 particles (state doc 4.5).
separable  = the z-collapse-then-in-plane-bilinear form of projprobe/s2_interpolant.py, which is the
             interpolant tt_bio/kernels/fslice implements. Axis-permuted so the plane normal's
             largest component is z, per that file's section 4.3 requirement.

Scope: this measures the INTERPOLANT, on the host, in fp32. It is not the device kernel and says
nothing about the device's own arithmetic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-acc-backend")
from tt_bio.cryoem.relion import _project

torch.set_grad_enabled(False)
DUMP = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/ttuser/relion-scratch/p8/call.2735016.npz")


def load(p):
    d = np.load(p)
    g = d["geom"].tolist()
    k = dict(zip(["mdlX", "mdlY", "mdlZ", "mdlInitY", "mdlInitZ", "maxR", "maxR2_padded",
                  "padding_factor", "imgX", "imgY", "No", "Nt", "P"], g))
    return d, k


def sample(mdl, k, X, Y, Z):
    """One integer lattice point of the padded half-volume, with the Hermitian pair and zero fill.

    RELION stores x >= 0 only. A negative-x lattice point is the conjugate of its antipode, which is
    the same rule project3Dmodel applies to the continuous coordinate before it interpolates.
    """
    mdlX, mdlY, mdlZ = k["mdlX"], k["mdlY"], k["mdlZ"]
    iY, iZ = k["mdlInitY"], k["mdlInitZ"]
    neg = X < 0
    X = np.where(neg, -X, X)
    Y = np.where(neg, -Y, Y)
    Z = np.where(neg, -Z, Z)
    ok = ((X < mdlX) & (Y - iY >= 0) & (Y - iY < mdlY) & (Z - iZ >= 0) & (Z - iZ < mdlZ))
    idx = np.where(ok, (Z - iZ) * mdlX * mdlY + (Y - iY) * mdlX + X, 0)
    v = mdl[idx]
    r = np.where(ok, v[..., 0], 0.0)
    i = np.where(ok, v[..., 1], 0.0)
    return r, np.where(neg, -i, i)


def separable(mdl, k, x, y, e):
    """z-collapse on the volume's own lattice, then in-plane bilinear. Per orientation."""
    pf = float(k["padding_factor"])
    u3 = np.array([e[0], e[3], e[6]], dtype=np.float64) * pf
    v3 = np.array([e[1], e[4], e[7]], dtype=np.float64) * pf
    n = np.cross(u3, v3)
    perm = np.argsort(np.abs(n))            # perm[2] is the largest normal component
    up, vp = u3[perm], v3[perm]
    M = np.array([[up[0], vp[0]], [up[1], vp[1]]])
    a, b = np.array([up[2], vp[2]]) @ np.linalg.inv(M)

    Xt = up[0] * x + vp[0] * y
    Yt = up[1] * x + vp[1] * y
    X0, Y0 = np.floor(Xt), np.floor(Yt)
    fx, fy = Xt - X0, Yt - Y0

    inv = np.argsort(perm)                  # permuted axis -> original axis
    acc_r = np.zeros_like(x)
    acc_i = np.zeros_like(x)
    for dy in (0, 1):
        wy = (1.0 - fy) if dy == 0 else fy
        Y = (Y0 + dy)
        for dx in (0, 1):
            wx = (1.0 - fx) if dx == 0 else fx
            X = (X0 + dx)
            Z = a * X + b * Y
            Z0 = np.floor(Z)
            tz = Z - Z0
            for dz, wz in ((0.0, 1.0 - tz), (1.0, tz)):
                c = [X.astype(np.int64), Y.astype(np.int64), (Z0 + dz).astype(np.int64)]
                o = [c[inv[0]], c[inv[1]], c[inv[2]]]
                r, i = sample(mdl, k, o[0], o[1], o[2])
                acc_r += wx * wy * wz * r
                acc_i += wx * wy * wz * i
    return acc_r, acc_i, abs(a), abs(b)


def one(path):
    d, k = load(path)
    mdl_np = d["mdl"]
    mdl_t = torch.from_numpy(mdl_np.copy())
    eul = d["eul"].astype(np.float64)
    No, Nt, P = k["No"], k["Nt"], k["P"]
    imgX, imgY, maxR = k["imgX"], k["imgY"], k["maxR"]

    pix = np.arange(P)
    xi = (pix % imgX).astype(np.float64)
    yi = pix // imgX
    yf = np.where(yi > maxR, yi - imgY, yi).astype(np.float64)

    # RELION's radius mask is a property of the pixel, not of the interpolant, so both arms carry it.
    pf = float(k["padding_factor"])
    xp = np.einsum("o,p->op", eul[:, 0], xi) + np.einsum("o,p->op", eul[:, 1], yf)
    yp = np.einsum("o,p->op", eul[:, 3], xi) + np.einsum("o,p->op", eul[:, 4], yf)
    zp = np.einsum("o,p->op", eul[:, 6], xi) + np.einsum("o,p->op", eul[:, 7], yf)
    inside = np.trunc((xp * pf) ** 2 + (yp * pf) ** 2 + (zp * pf) ** 2) <= k["maxR2_padded"]

    t = torch
    xt = torch.from_numpy(xi.astype(np.float32))
    yt = torch.from_numpy(yf.astype(np.float32))
    et = torch.from_numpy(d["eul"].copy())
    tri_r, tri_i = _project(t, mdl_t, xt, yt, et, k["mdlX"], k["mdlY"], k["mdlZ"],
                            k["mdlInitY"], k["mdlInitZ"], k["maxR2_padded"], k["padding_factor"])
    tri_r = tri_r.numpy().astype(np.float64)
    tri_i = tri_i.numpy().astype(np.float64)

    sep_r = np.zeros_like(tri_r)
    sep_i = np.zeros_like(tri_i)
    ab = []
    for o in range(No):
        r, i, aa, bb = separable(mdl_np, k, xi, yf, eul[o])
        sep_r[o] = np.where(inside[o], r, 0.0)
        sep_i[o] = np.where(inside[o], i, 0.0)
        ab.append((aa, bb))
    ab = np.array(ab)

    # Reference-level error, the input 4.11's transfer function takes.
    num = np.sqrt(((sep_r - tri_r) ** 2 + (sep_i - tri_i) ** 2).sum())
    den = np.sqrt((tri_r ** 2 + tri_i ** 2).sum())
    eps_rms = float(num / den)
    per_pix = np.sqrt((sep_r - tri_r) ** 2 + (sep_i - tri_i) ** 2)
    mag = np.sqrt(tri_r ** 2 + tri_i ** 2)
    sel = mag > 0
    rel_pix = per_pix[sel] / mag[sel]
    eps_med = float(np.median(rel_pix))

    # Score level, and the only thing that decides an orientation: does the argmin move.
    w = d["w"].astype(np.float64)
    img_r, img_i = d["img_r"].astype(np.float64), d["img_i"].astype(np.float64)
    tx, ty = d["tx"].astype(np.float64), d["ty"].astype(np.float64)
    ph = xi[None, :] * tx[:, None] + yf[None, :] * ty[:, None]
    s, c = np.sin(ph), np.cos(ph)
    sh_r = c * img_r - s * img_i
    sh_i = c * img_i + s * img_r

    def score(rr, ri):
        dr = rr[:, None, :] - sh_r[None, :, :]
        di = ri[:, None, :] - sh_i[None, :, :]
        return ((dr * dr + di * di) * w).sum(-1)

    d2_tri = score(tri_r, tri_i)
    d2_sep = score(sep_r, sep_i)
    rel_d2 = float(np.median(np.abs(d2_sep - d2_tri) / np.abs(d2_tri)))
    a_tri = int(np.argmin(d2_tri))
    a_sep = int(np.argmin(d2_sep))
    o_tri, o_sep = a_tri // Nt, a_sep // Nt

    # RELION's own diff2 for this call, as an anchor that the trilinear arm is the real thing.
    ref = d["diff2"].astype(np.float64).reshape(No, Nt)
    anchor = float(np.abs(d2_tri - ref).max() / np.abs(ref).max())

    gap = np.sort(d2_tri.ravel())
    gap_rel = float((gap[1] - gap[0]) / abs(gap[0]))

    out = {
        "dump": str(path), "No": No, "Nt": Nt, "P": P,
        "padding_factor": k["padding_factor"],
        "shear_slope_a_median": float(np.median(ab[:, 0])),
        "shear_slope_b_median": float(np.median(ab[:, 1])),
        "shear_slope_max": float(ab.max()),
        "eps_reference_rms": eps_rms,
        "eps_reference_median_per_pixel": eps_med,
        "rel_change_diff2_median": rel_d2,
        "argmin_tri": [o_tri, a_tri % Nt], "argmin_sep": [o_sep, a_sep % Nt],
        "argmin_moved": a_tri != a_sep,
        "orientation_moved": o_tri != o_sep,
        "best_to_second_relative_gap": gap_rel,
        "trilinear_vs_relion_own_diff2_rel": anchor,
    }
    return out


def main():
    """One row per particle. A single particle cannot answer the argmin question, because the
    score perturbation and the best-to-second gap are the same order, so whether the argmin
    survives is a coin flip on any one of them."""
    paths = sorted(Path(DUMP).parent.glob("call.*.npz")) if Path(DUMP).is_file() else []
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        paths = sorted(Path(sys.argv[1]).glob("call.*.npz"))
    rows = []
    for q in paths:
        r = one(q)
        rows.append(r)
        print("%-18s eps=%.4f  d(diff2)=%.3e  gap=%.3e  argmin_moved=%s  ori_moved=%s"
              % (q.name, r["eps_reference_rms"], r["rel_change_diff2_median"],
                 r["best_to_second_relative_gap"], r["argmin_moved"], r["orientation_moved"]),
              flush=True)
    n = len(rows)
    summ = {
        "n_particles": n,
        "padding_factor": rows[0]["padding_factor"] if n else None,
        "eps_reference_rms_median": float(np.median([r["eps_reference_rms"] for r in rows])),
        "eps_reference_rms_min": float(np.min([r["eps_reference_rms"] for r in rows])),
        "eps_reference_rms_max": float(np.max([r["eps_reference_rms"] for r in rows])),
        "rel_change_diff2_median": float(np.median([r["rel_change_diff2_median"] for r in rows])),
        "best_to_second_gap_median": float(np.median([r["best_to_second_relative_gap"] for r in rows])),
        "argmin_moved": int(sum(r["argmin_moved"] for r in rows)),
        "orientation_moved": int(sum(r["orientation_moved"] for r in rows)),
        "trilinear_vs_relion_own_diff2_rel_max":
            float(np.max([r["trilinear_vs_relion_own_diff2_rel"] for r in rows])),
    }
    print(json.dumps({"summary": summ, "rows": rows}, indent=2))
    return summ


if __name__ == "__main__":
    main()
