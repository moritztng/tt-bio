#!/usr/bin/env python3
"""S2 part 2: what the separable interpolant costs in ANGSTROMS at gold-standard FSC 0.143.

NOISE IS PART OF THE EXPERIMENT, not a garnish. FSC 0.143 is a noise-driven criterion: it marks
where signal falls below noise. Run noiseless, both pipelines recover the volume essentially
exactly and the FSC never crosses 0.143 at all -- the first run of this script returned the
Nyquist shell for both arms and discriminated nothing. With noise the interpolation error
competes with it, which is the regime a real refinement lives in.

The SAME noise realisation is given to both arms. A paired comparison is the only fair one
here; independent draws would put the difference between the interpolants underneath the
difference between the noise fields.

The accuracy gate, owed since section 15.3 and the only one that can still turn the design down.
Section 8 set it: more than 0.05 A of lost resolution at box 256 is a NO-GO for the separable form, and
the fallback is the exact-trilinear diagonal-matmul variant at 19x the compute.

Section 15.3 measured the interpolant against a continuum and got 1.23-1.27x RELION trilinear's own
error, flat across frequency shells. Flat is the property FSC depends on, but a ratio of per-slice
errors is not Angstroms and cannot be converted into them by argument. This runs the experiment.

The pipeline mirrors what a refinement actually does, and mirrors it symmetrically, which matters:
RELION projects AND backprojects with trilinear, so its interpolation errors partially cancel. The
comparison is therefore pipeline against pipeline -- trilinear/trilinear against separable/separable --
not interpolant against interpolant. Doing it asymmetrically would flatter neither side honestly.

  1. a real-space volume of Gaussian atoms, FFT'd to a padded Fourier volume (padding_factor 2)
  2. project N orientations with each interpolant
  3. backproject each set with the SAME interpolant, accumulating values and weights
  4. reconstruct by dividing out the weights
  5. FSC of each reconstruction against the true volume, and the 0.143 crossing in Angstroms
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128     # box
NORIENT = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
APIX = 1.0                                             # Angstrom per pixel
SNR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # per-pixel signal-to-noise
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 97
NATOM, SIGMA = 400, 1.5
PAD = 2


def build_volume(P, rng):
    """Gaussian atoms in the central half of a padded box, FFT'd. The padding is RELION's."""
    real = np.zeros((P, P, P), dtype=np.float32)
    c = P // 2
    R = P // 4
    v = rng.normal(size=(NATOM, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pos = c + v * (R * rng.uniform(size=(NATOM, 1)) ** (1.0 / 3.0))
    amp = rng.uniform(0.5, 1.5, size=NATOM)
    rad = int(np.ceil(3 * SIGMA))
    ax = np.arange(-rad, rad + 1)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    g = np.exp(-(gx ** 2 + gy ** 2 + gz ** 2) / (2 * SIGMA ** 2)).astype(np.float32)
    for (px, py, pz), a in zip(pos, amp):
        ix, iy, iz = int(round(px)), int(round(py)), int(round(pz))
        real[ix - rad:ix + rad + 1, iy - rad:iy + rad + 1, iz - rad:iz + rad + 1] += a * g
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(real))).astype(np.complex64)


def slice_coords(rmax, A, pad):
    g = np.arange(-rmax, rmax + 1)
    ux, vy = np.meshgrid(g, g, indexing="ij")
    keep = ux ** 2 + vy ** 2 <= rmax ** 2
    uu, vv = ux[keep].astype(np.float64), vy[keep].astype(np.float64)
    u3, v3 = pad * A[:, 0], pad * A[:, 1]
    p = uu[:, None] * u3[None, :] + vv[:, None] * v3[None, :]
    return p, uu, vv, u3, v3


def tri_weights(p, P):
    """RELION LIN_INTERP: the 8 corners of the cell containing p, with their weights."""
    c = P // 2
    q = p + c
    q0 = np.floor(q).astype(np.int64)
    fr = q - q0
    idx = np.empty((8, len(q)), dtype=np.int64)
    wt = np.empty((8, len(q)))
    n = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((1 - fr[:, 0]) if dx == 0 else fr[:, 0]) \
                    * ((1 - fr[:, 1]) if dy == 0 else fr[:, 1]) \
                    * ((1 - fr[:, 2]) if dz == 0 else fr[:, 2])
                cc = q0 + np.array([dx, dy, dz])
                np.clip(cc, 0, P - 1, out=cc)
                idx[n] = (cc[:, 0] * P + cc[:, 1]) * P + cc[:, 2]
                wt[n] = w
                n += 1
    return idx, wt


def sep_weights(uu, vv, u3, v3, P):
    """The kernel's interpolant: z-collapse on the (X, Y) lattice, then a two-pass in-plane resample.

    Expressed as sparse weights on the SAME lattice so it can be projected and backprojected with the
    identical machinery as trilinear. The axis permutation of section 4.3 and the four-candidate
    Catmull-Smith pivot of section 15.3 are both applied, since without them the form is not the one
    the kernel implements.
    """
    c = P // 2
    n = np.cross(u3, v3)
    perm = np.argsort(np.abs(n))
    a3, b3 = u3[perm], v3[perm]
    M = np.array([[a3[0], b3[0]], [a3[1], b3[1]]])
    ab = np.array([a3[2], b3[2]]) @ np.linalg.inv(M)
    aa, bb = ab[0], ab[1]

    cands = [("A", 0, abs(b3[1])), ("B", 0, abs(a3[0])),
             ("A", 1, abs(a3[1])), ("B", 1, abs(b3[0]))]
    order, swap, _ = max(cands, key=lambda t: t[2])
    p3, q3 = (b3, a3) if swap else (a3, b3)
    su, sv = (vv, uu) if swap else (uu, vv)
    u0, u1, v0, v1 = p3[0], p3[1], q3[0], q3[1]

    inv = np.argsort(perm)          # permuted axis -> original axis
    out_i, out_w = [], []

    def emit(X, Y, wxy):
        """Stage 1: split the (X, Y) column's weight between floor(z) and floor(z)+1."""
        Z = aa * X + bb * Y
        Z0 = np.floor(Z)
        t = Z - Z0
        for dz, wz in ((0.0, 1 - t), (1.0, t)):
            cc = (np.stack([X, Y, Z0 + dz], -1)[:, inv] + c).astype(np.int64)
            np.clip(cc, 0, P - 1, out=cc)
            out_i.append((cc[:, 0] * P + cc[:, 1]) * P + cc[:, 2])
            out_w.append(wxy * wz)

    if order == "A":
        alpha, beta = u0 - v0 * u1 / v1, v0 / v1
        Yr = u1 * su + v1 * sv
        Y0 = np.floor(Yr)
        ty = Yr - Y0
        for dy, wy in ((0.0, 1 - ty), (1.0, ty)):
            Y = Y0 + dy
            Xr = alpha * su + beta * Y
            X0 = np.floor(Xr)
            tx = Xr - X0
            for dx, wx in ((0.0, 1 - tx), (1.0, tx)):
                emit(X0 + dx, Y, wy * wx)
    else:
        gam, dlt = u1 / u0, v1 - v0 * u1 / u0
        Xr = u0 * su + v0 * sv
        X0 = np.floor(Xr)
        tx = Xr - X0
        for dx, wx in ((0.0, 1 - tx), (1.0, tx)):
            X = X0 + dx
            Yr = gam * X + dlt * sv
            Y0 = np.floor(Yr)
            ty = Yr - Y0
            for dy, wy in ((0.0, 1 - ty), (1.0, ty)):
                emit(X, Y0 + dy, wx * wy)
    return np.array(out_i), np.array(out_w)


def rand_rot(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def fsc(a, b, P, rmax):
    c = P // 2
    ax = np.arange(P) - c
    R = np.sqrt(ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)
    sh = np.round(R / PAD).astype(np.int64)          # shells in UNPADDED frequency units
    m = sh <= rmax
    sh, A, B = sh[m], a[m], b[m]
    num = np.bincount(sh, np.real(A * np.conj(B)), rmax + 1)
    da = np.bincount(sh, np.abs(A) ** 2, rmax + 1)
    db = np.bincount(sh, np.abs(B) ** 2, rmax + 1)
    return num / np.sqrt(np.maximum(da * db, 1e-30))


def resolution(f, N):
    """First shell where FSC drops below 0.143, linearly interpolated, in Angstrom."""
    for k in range(1, len(f)):
        if f[k] < 0.143:
            t = (f[k - 1] - 0.143) / max(f[k - 1] - f[k], 1e-12)
            kk = (k - 1) + t
            return (N * APIX) / max(kk, 1e-9), kk
    return (N * APIX) / max(len(f) - 1, 1), float(len(f) - 1)


def main():
    P = PAD * N
    rmax = N // 2
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    V = build_volume(P, rng)
    print(f"box {N}, padded {P}, {NORIENT} orientations, SNR {SNR}, volume built in "
          f"{time.time()-t0:.1f}s", flush=True)
    Vf = V.reshape(-1)

    acc = {k: np.zeros(P ** 3, dtype=np.complex128) for k in ("tri", "sep")}
    wsum = {k: np.zeros(P ** 3) for k in ("tri", "sep")}
    for o in range(NORIENT):
        A = rand_rot(rng)
        p, uu, vv, u3, v3 = slice_coords(rmax, A, PAD)
        # Both arms get the SAME noise field, so the comparison is paired.
        npts = p.shape[0]
        noise = None
        for kind in ("tri", "sep"):
            if kind == "tri":
                idx, wt = tri_weights(p, P)
            else:
                idx, wt = sep_weights(uu, vv, u3, v3, P)
            # Project: the sample is the weighted sum over its lattice neighbours.
            val = (Vf[idx] * wt).sum(0)
            if noise is None:
                sig = np.sqrt(np.mean(np.abs(val) ** 2))
                sd = sig / np.sqrt(max(SNR, 1e-9)) / np.sqrt(2.0)
                noise = rng.normal(0, sd, npts) + 1j * rng.normal(0, sd, npts)
            val = val + noise
            # Backproject with the SAME weights -- the adjoint. Standard gridding accumulates the
            # weighted value and the weight, then divides.
            fl = idx.reshape(-1)
            contrib = (wt * val[None, :]).reshape(-1)
            acc[kind] += np.bincount(fl, contrib.real, P ** 3) \
                + 1j * np.bincount(fl, contrib.imag, P ** 3)
            wsum[kind] += np.bincount(fl, wt.reshape(-1), P ** 3)
        if (o + 1) % 300 == 0:
            print(f"  {o+1}/{NORIENT}  ({time.time()-t0:.0f}s)", flush=True)

    res = {"box": N, "pad": P, "norient": NORIENT, "apix": APIX, "snr": SNR, "arms": {}}
    print()
    for kind in ("tri", "sep"):
        rec = (acc[kind] / np.maximum(wsum[kind], 1e-9)).reshape(P, P, P).astype(np.complex128)
        f = fsc(rec, V.astype(np.complex128), P, rmax)
        r, kk = resolution(f, N)
        res["arms"][kind] = {"resolution_A": r, "shell": kk, "fsc": f.tolist()}
        print(f"{kind:4s}: FSC 0.143 at shell {kk:6.2f} -> {r:7.3f} A", flush=True)
    d = res["arms"]["sep"]["resolution_A"] - res["arms"]["tri"]["resolution_A"]
    res["delta_A"] = d
    res["gate_A"] = 0.05
    res["pass"] = bool(abs(d) <= 0.05)
    print(f"\nseparable minus trilinear: {d:+.4f} A   (gate: 0.05 A)  -> "
          f"{'PASS' if abs(d) <= 0.05 else 'FAIL'}", flush=True)
    json.dump(res, open(HERE / f"s2p2_fsc_box{N}_snr{SNR}_s{SEED}.json", "w"), indent=1)
    ft, fs = res["arms"]["tri"]["fsc"], res["arms"]["sep"]["fsc"]
    print("  shell : FSC tri / FSC sep, upper half of the spectrum")
    for k in range(len(ft) // 2, len(ft), max(1, len(ft) // 16)):
        print(f"   {k:4d} : {ft[k]:6.3f} / {fs[k]:6.3f}")


main()
