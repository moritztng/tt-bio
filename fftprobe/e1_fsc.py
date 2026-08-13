#!/usr/bin/env python3
"""E1 -- what does the FFT's precision cost in resolution? The gate on the whole cryo-EM program.

Section 3 measured the wall: every fp32 value crossing the Tensix FPU is truncated to about 11
mantissa bits, so the best end-to-end accuracy an FPU-based FFT can reach on Blackhole is 3.5e-4
relative L2, against cuFFT's 2.5e-7. Whether 3.5e-4 is acceptable is not a hardware question, so it
is answered on CPU, and it is answered in the only unit that matters to a structural biologist:
Angstroms of resolution at gold-standard FSC = 0.143.

This is the fallback protocol the design doc specifies, not a real `relion_refine` run: a synthetic
map, projections at known orientations, direct Fourier inversion, and a gold-standard FSC between
two independently reconstructed half sets. It is weaker evidence than real refinement because it
omits the iterative estimator's error accumulation -- a real refinement feeds each iteration's
errors into the next iteration's orientation assignment, and this does not. That understates the
cost of a given error level, so a FAIL here is conclusive and a PASS here is necessary but not
sufficient. Stated up front because it bounds what the number can be used for.

Every forward and inverse transform in the reconstruction path is perturbed, which is where the
hardware's error would actually enter. The perturbation is complex Gaussian scaled to a target
relative L2 and applied to the transform OUTPUT, matching the measured error's broadly distributed
shape (section 3.3: max-abs/max-ref tracks rel L2 within 40% in every arm, so the error is not
concentrated in a few coefficients).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BOX = int(os.environ.get("E1_BOX", "128"))
NPROJ = int(os.environ.get("E1_NPROJ", "8000"))
SNR = float(os.environ.get("E1_SNR", "0.1"))
PIXEL_A = float(os.environ.get("E1_PIXEL_A", "1.5"))     # Angstroms per pixel
SEED = 0

# The arms, as fixed in advance by the design doc.
ARMS = {
    "arm0_exact": 0.0,          # control
    "arm1_3.5e-4": 3.5e-4,      # path C+, the best FPU-based accuracy measured
    "arm2_1.5e-3": 1.5e-3,      # path A / path C without the split-bf16 matmul
    "arm3_4.1e-2": 4.1e-2,      # bf16, to price the 2x bandwidth win
}

_rng_perturb = np.random.default_rng(12345)


def perturb(F, rel):
    """Complex Gaussian noise on a transform output, scaled to a target relative L2."""
    if rel == 0.0:
        return F
    n = _rng_perturb.standard_normal(F.shape) + 1j * _rng_perturb.standard_normal(F.shape)
    n *= rel * np.linalg.norm(F) / np.linalg.norm(n)
    return F + n


def make_map(box, rng):
    """A synthetic density: Gaussian blobs on a connected random walk, low-pass filtered.

    A random-walk backbone rather than independent blobs, because a protein's power spectrum falls
    off with a shape that comes from its chain connectivity, and FSC at 0.143 is decided by the
    high-frequency shells where that falloff lives.
    """
    n_atoms = box * 6
    pos = np.zeros((n_atoms, 3))
    step = rng.standard_normal((n_atoms, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    pos = np.cumsum(step * 3.0, axis=0)
    pos -= pos.mean(0)
    # keep the walk inside the central half of the box, the usual particle-diameter convention
    scale = (box * 0.22) / np.abs(pos).max()
    pos = pos * scale + box / 2.0

    vol = np.zeros((box, box, box), dtype=np.float64)
    idx = np.floor(pos).astype(int)
    ok = np.all((idx >= 0) & (idx < box), axis=1)
    np.add.at(vol, (idx[ok, 0], idx[ok, 1], idx[ok, 2]), 1.0)

    # low-pass to a physically plausible ~3 A envelope
    f = np.fft.fftn(vol)
    kz, ky, kx = np.meshgrid(*[np.fft.fftfreq(box)] * 3, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    f *= np.exp(-k2 * (box * 0.10) ** 2 * 0.5)
    return np.real(np.fft.ifftn(f))


def random_rotations(n, rng):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def slice_coords(box, R):
    """Central-slice coordinates: the 2D Fourier plane of a projection, rotated into the volume."""
    fy = np.fft.fftfreq(box) * box
    fx = np.fft.fftfreq(box) * box
    gy, gx = np.meshgrid(fy, fx, indexing="ij")
    pts = np.stack([np.zeros_like(gx), gy, gx], -1).reshape(-1, 3)   # plane z=0 in slice frame
    return pts @ R.T


def trilinear_gather(F3, coords, box):
    c = coords + box / 2.0
    i0 = np.floor(c).astype(np.int64)
    frac = c - i0
    out = np.zeros(coords.shape[0], dtype=np.complex128)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dz, frac[:, 0], 1 - frac[:, 0])
                     * np.where(dy, frac[:, 1], 1 - frac[:, 1])
                     * np.where(dx, frac[:, 2], 1 - frac[:, 2]))
                idx = i0 + np.array([dz, dy, dx])
                ok = np.all((idx >= 0) & (idx < box), axis=1)
                flat = (idx[:, 0] * box + idx[:, 1]) * box + idx[:, 2]
                out[ok] += w[ok] * F3[flat[ok]]
    return out


def trilinear_scatter(acc, wacc, coords, vals, box):
    """Backprojection: trilinear scatter-add of a Fourier slice into the 3D volume.

    np.bincount rather than np.add.at -- add.at is a Python-level loop over duplicates and is 50x
    slower here, which turns a 20-minute arm into a day.
    """
    c = coords + box / 2.0
    i0 = np.floor(c).astype(np.int64)
    frac = c - i0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dz, frac[:, 0], 1 - frac[:, 0])
                     * np.where(dy, frac[:, 1], 1 - frac[:, 1])
                     * np.where(dx, frac[:, 2], 1 - frac[:, 2]))
                idx = i0 + np.array([dz, dy, dx])
                ok = np.all((idx >= 0) & (idx < box), axis=1)
                flat = ((idx[:, 0] * box + idx[:, 1]) * box + idx[:, 2])[ok]
                ww = w[ok]
                vv = vals[ok]
                n = box**3
                acc += (np.bincount(flat, ww * vv.real, n)
                        + 1j * np.bincount(flat, ww * vv.imag, n))
                wacc += np.bincount(flat, ww, n)


def fsc(a, b, box):
    Fa, Fb = np.fft.fftn(a), np.fft.fftn(b)
    kz, ky, kx = np.meshgrid(*[np.fft.fftfreq(box) * box] * 3, indexing="ij")
    r = np.round(np.sqrt(kx**2 + ky**2 + kz**2)).astype(int).ravel()
    nb = box // 2
    keep = r < nb
    r, A, B = r[keep], Fa.ravel()[keep], Fb.ravel()[keep]
    num = np.bincount(r, np.real(A * np.conj(B)), nb)
    da = np.bincount(r, np.abs(A) ** 2, nb)
    db = np.bincount(r, np.abs(B) ** 2, nb)
    return num / np.sqrt(np.maximum(da * db, 1e-30))


def res_at(curve, thr, box, pixel_a):
    """Resolution in Angstroms where the FSC curve first drops below `thr`, linearly interpolated."""
    for i in range(1, len(curve)):
        if curve[i] < thr:
            f0, f1 = curve[i - 1], curve[i]
            t = (f0 - thr) / max(f0 - f1, 1e-12)
            shell = (i - 1) + t
            if shell <= 0:
                return float("inf")
            return box * pixel_a / shell
    return box * pixel_a / (len(curve) - 1)


def reconstruct(F3flat, rots, imgs_noise, rel, box, half):
    acc = np.zeros(box**3, dtype=np.complex128)
    wacc = np.zeros(box**3, dtype=np.float64)
    for j in half:
        coords = slice_coords(box, rots[j])
        # forward 2D transform of the noisy particle image -- perturbed, this is the FFT we ship
        F2 = perturb(np.fft.fft2(imgs_noise[j]), rel).ravel()
        trilinear_scatter(acc, wacc, coords, F2, box)
    vol = np.where(wacc > 1e-6, acc / np.maximum(wacc, 1e-6), 0.0).reshape(box, box, box)
    # inverse 3D transform -- also perturbed
    return np.real(np.fft.ifftn(perturb(vol, rel)))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    vol = make_map(BOX, rng)
    F3 = np.fft.fftn(vol)
    F3flat = F3.ravel()

    rots = random_rotations(NPROJ, rng)
    print(f"box={BOX} nproj={NPROJ} snr={SNR} pixel={PIXEL_A}A", flush=True)

    # --- generate projections once; every arm sees identical particles and identical noise --------
    imgs = np.empty((NPROJ, BOX, BOX))
    for j in range(NPROJ):
        F2 = trilinear_gather(F3flat, slice_coords(BOX, rots[j]), BOX).reshape(BOX, BOX)
        imgs[j] = np.real(np.fft.ifft2(F2))
    sig = imgs.std()
    imgs += rng.standard_normal(imgs.shape) * (sig / np.sqrt(SNR))
    print(f"projections done {time.time()-t0:.0f}s", flush=True)

    halves = (np.arange(0, NPROJ, 2), np.arange(1, NPROJ, 2))
    out = {"box": BOX, "nproj": NPROJ, "snr": SNR, "pixel_a": PIXEL_A, "arms": {}}
    for name, rel in ARMS.items():
        ta = time.time()
        v1 = reconstruct(F3flat, rots, imgs, rel, BOX, halves[0])
        v2 = reconstruct(F3flat, rots, imgs, rel, BOX, halves[1])
        c = fsc(v1, v2, BOX)
        r143 = res_at(c, 0.143, BOX, PIXEL_A)
        out["arms"][name] = {
            "rel_l2": rel,
            "res_0.143_A": r143,
            "fsc": [float(v) for v in c],
            "seconds": time.time() - ta,
        }
        print(f"{name:14s} rel={rel:.2e}  res@0.143 = {r143:.3f} A   ({time.time()-ta:.0f}s)",
              flush=True)

    base = out["arms"]["arm0_exact"]["res_0.143_A"]
    for name, a in out["arms"].items():
        a["delta_A_vs_control"] = a["res_0.143_A"] - base
    out["verdict"] = {
        "arm1_passes_0.1A": abs(out["arms"]["arm1_3.5e-4"]["delta_A_vs_control"]) < 0.1,
        "arm2_passes_0.1A": abs(out["arms"]["arm2_1.5e-3"]["delta_A_vs_control"]) < 0.1,
        "arm3_passes_0.1A": abs(out["arms"]["arm3_4.1e-2"]["delta_A_vs_control"]) < 0.1,
    }
    p = Path(__file__).resolve().parent / f"e1_fsc_box{BOX}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out["verdict"], indent=1), flush=True)
    print(f"total {time.time()-t0:.0f}s -> {p}", flush=True)


if __name__ == "__main__":
    main()
