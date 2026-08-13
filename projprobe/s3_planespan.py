#!/usr/bin/env python3
"""S3 -- stage 1's plane-span distribution over real HEALPix directions. Host only.

State doc section 4.3 assumes that with the three axis-permuted volume copies, a 32x32 (X, Y) output
tile's plane spans a mean of 33 z-planes and at most 65. If the mean is materially higher, stage 1
stops being negligible and the (X, Y) tile has to be sub-blocked, which costs tile efficiency.
Kill gate: mean > 45.

HEALPix RING-scheme pixel centres, implemented here rather than pulled in: healpy is not installed
on this host and the pix2ang RING formula is short. Verified against two invariants -- 12*Nside^2
pixels and equal-area (the mean of cos(theta) over all pixels must be 0 to ~1e-12).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def healpix_ring_ang(nside: int):
    """(theta, phi) of every RING-scheme pixel centre. Returns arrays of length 12*nside^2."""
    npix = 12 * nside * nside
    ipix = np.arange(npix, dtype=np.int64)
    ncap = 2 * nside * (nside - 1)          # pixels in the north polar cap
    theta = np.empty(npix)
    phi = np.empty(npix)

    # North polar cap
    m = ipix < ncap
    p = ipix[m]
    ph = (p + 1).astype(np.float64)
    i = (0.5 * (1.0 + np.sqrt(1.0 + 2.0 * ph))).astype(np.int64)   # ring index, 1-based
    i = np.maximum(i, 1)
    # correct the ring index where the sqrt rounds the wrong way
    over = 2 * i * (i - 1) >= p + 1
    i[over] -= 1
    i = np.maximum(i, 1)
    j = (p + 1) - 2 * i * (i - 1)                                   # position in ring, 1-based
    theta[m] = np.arccos(1.0 - (i * i) / (3.0 * nside * nside))
    phi[m] = (j - 0.5) * (np.pi / (2.0 * i))

    # Equatorial belt
    m = (ipix >= ncap) & (ipix < npix - ncap)
    p = ipix[m] - ncap
    i = (p // (4 * nside)).astype(np.int64) + nside                 # ring index
    j = (p % (4 * nside)).astype(np.int64) + 1
    s = np.where(((i + nside) % 2) == 1, 1.0, 0.5)                  # ring offset
    theta[m] = np.arccos((2 * nside - i) * 2.0 / (3.0 * nside))
    phi[m] = (j - s) * (np.pi / (2.0 * nside))

    # South polar cap, by symmetry with the north
    m = ipix >= npix - ncap
    p = npix - ipix[m]                                              # 1-based from the south pole
    i = (0.5 * (1.0 + np.sqrt(2.0 * p - 1.0))).astype(np.int64)
    i = np.maximum(i, 1)
    over = 2 * i * (i + 1) < p
    i[over] += 1
    j = 4 * i + 1 - (p - 2 * i * (i - 1))
    theta[m] = np.arccos(-1.0 + (i * i) / (3.0 * nside * nside))
    phi[m] = (j - 0.5) * (np.pi / (2.0 * i))
    return theta, phi


def spans(theta, phi, tile=32):
    """|a| + |b| and the plane span for each direction, with the max-|component|-as-z permutation.

    The slice plane is the plane through the origin normal to the viewing direction n. Written as a
    graph over the two non-normal axes, z = a*X + b*Y with (a, b) = (-n_x/n_z, -n_y/n_z) AFTER the
    permutation that puts n's largest component in z. That permutation is exactly what the three
    axis-permuted volume copies of section 4.3 buy, and it forces |a|, |b| <= 1.
    """
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)
    a_ = np.abs(n)
    order = np.argsort(a_, axis=1)                  # largest component last -> becomes z
    ns = np.take_along_axis(n, order, axis=1)
    a = -ns[:, 0] / ns[:, 2]
    b = -ns[:, 1] / ns[:, 2]
    return np.abs(a) + np.abs(b), 1.0 + tile * (np.abs(a) + np.abs(b))


def main():
    res = {}
    for order in (3, 4, 5):
        nside = 2 ** order
        th, ph = healpix_ring_ang(nside)
        npix = len(th)
        # invariants: pixel count, and equal-area => mean(cos theta) == 0
        chk = {"npix": int(npix), "npix_expected": 12 * nside * nside,
               "mean_cos_theta": float(np.mean(np.cos(th)))}
        s, span = spans(th, ph)
        res[f"order{order}"] = {
            "check": chk, "ndirections": int(npix),
            "abs_a_plus_b": {"mean": float(s.mean()), "max": float(s.max()),
                             "p50": float(np.percentile(s, 50)),
                             "p95": float(np.percentile(s, 95)),
                             "p99": float(np.percentile(s, 99))},
            "span_planes_tile32": {"mean": float(span.mean()), "max": float(span.max()),
                                   "p50": float(np.percentile(span, 50)),
                                   "p95": float(np.percentile(span, 95)),
                                   "p99": float(np.percentile(span, 99))},
        }
        print(f"order {order}: {npix} directions  (expected {12*nside*nside}, "
              f"mean cos(theta) {chk['mean_cos_theta']:+.2e})")
        print(f"   |a|+|b|  mean {s.mean():.4f}  p50 {np.percentile(s,50):.4f}  "
              f"p95 {np.percentile(s,95):.4f}  max {s.max():.4f}")
        print(f"   span     mean {span.mean():.2f}  p50 {np.percentile(span,50):.2f}  "
              f"p95 {np.percentile(span,95):.2f}  max {span.max():.2f}")

    # Continuum reference: the same statistic over a uniform sphere, so the HEALPix number is not
    # trusted on its own.
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    s, span = spans(th, ph)
    res["uniform_sphere_control"] = {"abs_a_plus_b_mean": float(s.mean()),
                                     "span_mean": float(span.mean()),
                                     "span_max": float(span.max())}
    print(f"uniform-sphere control: |a|+|b| mean {s.mean():.4f}, span mean {span.mean():.2f}")

    m4 = res["order4"]["span_planes_tile32"]["mean"]
    print(f"\n--- S3 gate ---\nmean span at order 4 = {m4:.2f}  (gate: < 45)  "
          f"-> {'PASS' if m4 < 45 else 'FAIL'}")
    res["gate"] = {"mean_span_order4": m4, "pass": bool(m4 < 45)}
    json.dump(res, open(HERE / "s3_planespan.json", "w"), indent=1)


main()
