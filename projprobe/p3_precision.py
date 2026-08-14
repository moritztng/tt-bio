#!/usr/bin/env python3
"""P3: what arithmetic precision costs the projection, per spatial-frequency shell.

Two questions the projection spike left open and never separated (its section 8, Q1 and Q3):
the interpolant question it answered in Angstroms, the ROUND-OFF question it only ever
answered as an aggregate per-slice relative L2 on the device.

This measures the round-off error's dependence on spatial frequency, which is the property
FSC at 0.143 depends on. It needs no reconstruction and no card, so it is seconds rather than
hours, and it is the screen that predicts what `p3_precision_fsc.py` will measure in Angstroms.

Precision arms, each a model of a real implementation:

  fp64        the reference. Interpolation weights and volume in double.
  fp32        RELION's GPU texel precision without its texture weight quantisation.
  tex8        RELION's SHIPPING CUDA path: fp32 texels, and the interpolation fractions
              alpha/beta/gamma quantised to 8 fractional bits, which is what the hardware
              texture unit does (CUDA C Programming Guide, "Linear Filtering": stored in
              9-bit fixed point with 8 bits of fractional value). src/acc/acc_projector_impl.h
              creates the texture with cudaFilterModeLinear, and acc_projectorkernel_impl.h
              calls tex3D<XFLOAT> unless PROJECTOR_NO_TEXTURES is set.
  bf16        our kernel: bf16 volume, bf16 coefficient tiles, fp32 accumulation in DST,
              bf16 pack of every intermediate that leaves DST.
  bf16_static bf16 volume only. This error is the SAME on every read of a voxel, so it does
              not average down over orientations; it is the half of bf16 that a refinement
              cannot wash out.
  bf16_dyn    bf16 weights and bf16 packs only, exact volume. Uncorrelated across orientations.

The static/dynamic split is the mechanism, and it is the reason the FFT spike's E1 result
cannot simply be extended: E1 injected an i.i.d. perturbation per transform, which averages
down over 8,000 projections. A bf16 volume does not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2p2_fsc as H          # noqa: E402  (weight functions, refactored to be importable)

HERE = Path(__file__).resolve().parent
_A = sys.argv[1:] if __name__ == "__main__" else []     # argv only when run directly
N = int(_A[0]) if len(_A) > 0 else 256
NORIENT = int(_A[1]) if len(_A) > 1 else 64
SEED = int(_A[2]) if len(_A) > 2 else 11
VARIANT = _A[3] if len(_A) > 3 else "tri"               # tri|twopass|sep2d
PAD = 2
NSHELL = 8

ARMS = ["fp32", "tex8", "bf16", "bf16_static", "bf16_dyn"]


def bf16(x):
    """fp32 -> bf16 -> fp32, round to nearest even. bf16 keeps 8 mantissa bits, so the half-ulp
    is 2^-8 = 3.9e-3 relative and the rms over a smooth distribution is about 1.7e-3."""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    r = ((u >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def bf16c(z):
    """Real and imaginary rounded independently, which is how the volume is stored: section
    15.1 of the projection spike interleaves them per row, each its own bf16 element."""
    return (bf16(np.real(z).astype(np.float32)).astype(np.float64)
            + 1j * bf16(np.imag(z).astype(np.float32)).astype(np.float64))


def tri_weights_q(p, P, frac_bits):
    """RELION LIN_INTERP with the interpolation fraction quantised to `frac_bits` bits.

    frac_bits=None reproduces H.tri_weights exactly; frac_bits=8 is the CUDA texture unit.
    """
    c = P // 2
    q = p + c
    q0 = np.floor(q).astype(np.int64)
    fr = q - q0
    if frac_bits is not None:
        s = float(1 << frac_bits)
        fr = np.round(fr * s) / s
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


def weights(variant, p, uu, vv, u3, v3, P):
    if variant == "tri":
        return H.tri_weights(p, P)
    if variant == "sep2d":
        return H.sep2d_weights(uu, vv, u3, v3, P)
    return H.sep_weights(uu, vv, u3, v3, P)


def project(arm, variant, Vf, Vf32, Vfb, p, uu, vv, u3, v3, P):
    """One slice under one precision model. The contraction is always fp32-accumulated for the
    device arms, because matmul_tiles accumulates into DST in fp32 whatever the input dtype
    (projection spike S0: fp32 and bf16 matmul rates within 2%, so nothing is bought by
    accumulating narrower)."""
    if arm == "tex8":
        idx, wt = tri_weights_q(p, P, 8) if variant == "tri" else weights(variant, p, uu, vv, u3, v3, P)
        return (Vf32[idx] * wt.astype(np.float32)).sum(0).astype(np.complex128)
    idx, wt = weights(variant, p, uu, vv, u3, v3, P)
    if arm == "fp64":
        return (Vf[idx] * wt).sum(0)
    if arm == "fp32":
        return (Vf32[idx] * wt.astype(np.float32)).sum(0).astype(np.complex128)
    if arm == "bf16_static":
        return (Vfb[idx] * wt).sum(0)
    wb = bf16(wt.astype(np.float32)).astype(np.float32)
    src = Vfb if arm in ("bf16",) else Vf32
    val = (src[idx].astype(np.complex64) * wb).sum(0)
    return bf16c(val)          # the pack out of DST


def main():
    P = PAD * N
    rmax = N // 2
    rng = np.random.default_rng(SEED)
    H.N, H.PAD, H.VARIANT, H.CUBIC = N, PAD, VARIANT, False
    t0 = time.time()
    V = H.build_volume(P, rng)
    Vf = V.reshape(-1).astype(np.complex128)
    Vf32 = Vf.astype(np.complex64)
    Vfb = bf16c(Vf32).astype(np.complex128)
    vol_rel = np.linalg.norm(Vfb - Vf) / np.linalg.norm(Vf)
    print(f"box {N} pad {PAD} P {P} variant {VARIANT} {NORIENT} orient  built {time.time()-t0:.1f}s",
          flush=True)
    print(f"bf16 volume storage: rel L2 {vol_rel:.4e}   (this term is STATIC across orientations)",
          flush=True)

    num = {a: np.zeros(NSHELL) for a in ARMS}
    den = np.zeros(NSHELL)
    tot_n = {a: 0.0 for a in ARMS}
    tot_d = 0.0
    for o in range(NORIENT):
        A = H.rand_rot(rng)
        p, uu, vv, u3, v3 = H.slice_coords(rmax, A, PAD)
        q = np.sqrt(uu ** 2 + vv ** 2)
        sh = np.minimum((q / (rmax + 1e-9) * NSHELL).astype(int), NSHELL - 1)
        ref = project("fp64", VARIANT, Vf, Vf32, Vfb, p, uu, vv, u3, v3, P)
        d2 = np.abs(ref) ** 2
        den += np.bincount(sh, d2, NSHELL)
        tot_d += d2.sum()
        for a in ARMS:
            e2 = np.abs(project(a, VARIANT, Vf, Vf32, Vfb, p, uu, vv, u3, v3, P) - ref) ** 2
            num[a] += np.bincount(sh, e2, NSHELL)
            tot_n[a] += e2.sum()
        if (o + 1) % 16 == 0:
            print(f"  {o+1}/{NORIENT} ({time.time()-t0:.0f}s)", flush=True)

    res = {"box": N, "pad": PAD, "norient": NORIENT, "seed": SEED, "variant": VARIANT,
           "nshell": NSHELL, "bf16_volume_rel_l2": vol_rel, "arms": {}}
    print(f"\nper-shell relative L2 vs fp64, low -> high frequency ({VARIANT})")
    print(f"{'arm':12s} {'aggregate':>10s}   shells")
    for a in ARMS:
        per = np.sqrt(num[a] / np.maximum(den, 1e-300))
        agg = float(np.sqrt(tot_n[a] / tot_d))
        res["arms"][a] = {"rel_l2": agg, "per_shell": per.tolist(),
                          "flatness_max_over_min": float(per.max() / max(per.min(), 1e-300))}
        print(f"{a:12s} {agg:10.3e}   " + " ".join(f"{x:8.2e}" for x in per), flush=True)
    out = HERE / f"p3_precision_box{N}_{VARIANT}_s{SEED}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n-> {out.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
