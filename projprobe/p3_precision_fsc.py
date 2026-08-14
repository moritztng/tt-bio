#!/usr/bin/env python3
"""P3 part 2: what arithmetic precision costs in ANGSTROMS at gold-standard FSC 0.143.

Same construction as `s2p2_fsc.py`, which settled the INTERPOLANT question, with the arm now
carrying a precision model as well as an interpolant. Everything that made that harness
discriminate is kept and none of it is garnish:

  * noise, because FSC 0.143 marks where signal falls below noise and a noiseless run returns
    the Nyquist shell for every arm;
  * the SAME noise realisation for both arms, so the comparison is paired;
  * symmetric pipelines, project and backproject with the same interpolant and the same
    precision, because RELION does and its interpolation errors partially cancel.

The control arm is always RELION's own reference: axis-aligned trilinear, fp64. Every number
this prints is therefore Angstroms lost against RELION, which is the unit a cryo-EM user
decides on.

  python3 p3_precision_fsc.py <box> <norient> <snr> <seed> <variant> <precision>

variant    tri | twopass | sep2d
precision  fp64 | fp32 | tex8 | bf16 | bf16_dev | bf16_dev_pess | bf16acc

  fp64           control arithmetic. tri/fp64 against tri/fp64 is the A/A pair.
  fp32           RELION's GPU texel precision.
  tex8           RELION's SHIPPING CUDA path: fp32 texels and the interpolation fractions in
                 8 fractional bits, which is what the texture unit does. Defined for `tri`.
  bf16           the kernel, emulated structurally: bf16 volume, bf16 coefficients, fp32
                 accumulation, bf16 pack out of DST. p3_precision.py measures this at
                 2.18e-3 relative, and the DEVICE measures the real kernel at 5.90e-3
                 (DERIVED in quadrature from stage 1's 3.41e-3 and stage 2's 4.82e-3), because
                 the sparse-weight form here has one pack point where the kernel has three.
                 So `bf16` UNDERSTATES the kernel and is not the arm to quote.
  bf16_dev       bf16 plus an i.i.d. top-up sized to bring the total to the device-measured
                 5.90e-3. This is the arm that answers the question.
  bf16_dev_pess  the same total, with the top-up folded into the VOLUME instead, so all of it
                 is static: identical on every read of a voxel and unable to average down over
                 orientations. The pessimistic bound on the same measured error.
  bf16acc        fp32 projection with the backprojection ACCUMULATOR rounded to bf16 at every
                 flush. The design already rejected this (spike section 29.2 measured bf16
                 accumulation error growing 2.9x over eight contributions while fp32 stayed
                 flat); this prices the rejection in Angstroms. Flush-granularity rounding is
                 far fewer rounding events than the real thing, so it is a LOWER bound on the
                 damage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2p2_fsc as H              # noqa: E402
from p3_precision import bf16, bf16c, tri_weights_q      # noqa: E402

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NORIENT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SNR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 11
VARIANT = sys.argv[5] if len(sys.argv) > 5 else "twopass"
PREC = sys.argv[6] if len(sys.argv) > 6 else "bf16_dev"
PAD = 2
APIX = 1.0
FLUSH = 25

# The device-measured round-off of the real kernel against an fp64 model of the same
# operation, projection spike sections 24.1 (stage 1, band 28: 3.41e-3) and 28.1 (stage 2,
# fused, general shear: 4.82e-3). Composed in quadrature -- DERIVED, not measured end to end,
# because the two stages were never run chained on the card.
DEVICE_REL_L2 = float(np.hypot(3.41e-3, 4.82e-3))
# What p3_precision.py measures for the structural `bf16` model at box 256, both interpolants.
STRUCT_REL_L2 = 2.18e-3
TOPUP = float(np.sqrt(max(DEVICE_REL_L2 ** 2 - STRUCT_REL_L2 ** 2, 0.0)))

# The 0.1 A bar, fixed before any number is read, and inherited from the FFT spike's E1 where
# it decided the same question for the transform. The projection spike separately held the
# INTERPOLANT to a stricter self-imposed 0.05 A; that gate is not relaxed by this one.
BAR_A = 0.1


def fsc_planewise(accv, wsumv, Vv, P, rmax):
    """H.fsc's result, computed one z-plane at a time and never materialising the division.

    Same shells, same float32 radius, same rounding, verified equal to H.fsc at box 96. The
    plane-at-a-time form is what makes box 256 run on a 30 GB host at all: H.fsc's boolean
    masks copy a P^3 complex128 array twice, and the gridding divide allocates a third, which
    is 6.4 GB on top of an 8.5 GB working set and gets the process OOM-killed after the last
    orientation -- i.e. after all the work and before any result. Frees each arm's
    accumulators as it consumes them.
    """
    nb = rmax + 2                       # last bin collects everything outside rmax
    num, da, db = (np.zeros(nb) for _ in range(3))
    ax = (np.arange(P) - P // 2).astype(np.float32)
    a2 = ax ** 2
    for z in range(P):
        s0, s1 = z * P * P, (z + 1) * P * P
        r2 = (a2[z] + a2[:, None] + a2[None, :]).reshape(-1)
        sh = np.round(np.sqrt(r2) / PAD).astype(np.int64)
        np.minimum(sh, rmax + 1, out=sh)
        a = accv[s0:s1] / np.maximum(wsumv[s0:s1], 1e-9)
        b = Vv[s0:s1].astype(np.complex128)
        num += np.bincount(sh, a.real * b.real + a.imag * b.imag, nb)
        da += np.bincount(sh, a.real ** 2 + a.imag ** 2, nb)
        db += np.bincount(sh, b.real ** 2 + b.imag ** 2, nb)
    return (num / np.sqrt(np.maximum(da * db, 1e-30)))[:rmax + 1]


def weights(p, uu, vv, u3, v3, P):
    if VARIANT == "tri":
        return tri_weights_q(p, P, 8) if PREC == "tex8" else H.tri_weights(p, P)
    if VARIANT == "sep2d":
        return H.sep2d_weights(uu, vv, u3, v3, P)
    return H.sep_weights(uu, vv, u3, v3, P)


def main():
    P = PAD * N
    rmax = N // 2
    rng = np.random.default_rng(SEED)
    # A SEPARATE stream for everything the precision model draws, so the volume, the
    # orientations and the noise field are bit-identical across precision arms at the same
    # seed. Sharing one stream would shift it by the arm's own draws and quietly unpair the
    # comparison between arms.
    rng_p = np.random.default_rng(SEED + 1_000_003)
    H.N, H.PAD, H.APIX, H.VARIANT, H.CUBIC = N, PAD, APIX, VARIANT, False
    t0 = time.time()
    V = H.build_volume(P, rng)
    # complex64, a view, exactly as s2p2_fsc.py has it: the control's own volume storage is
    # fp32 and only its arithmetic is fp64, so `fp32` here isolates arithmetic from storage.
    Vf = V.reshape(-1)
    print(f"box {N} [{VARIANT}/{PREC}], padded {P}, {NORIENT} orient, SNR {SNR}, seed {SEED}, "
          f"built in {time.time()-t0:.1f}s", flush=True)

    # The arm's volume, perturbed once. Static error lives here and cannot average down.
    if PREC in ("bf16", "bf16_dev"):
        Va = bf16c(Vf).astype(np.complex64)
    elif PREC == "bf16_dev_pess" or PREC.startswith("pertS"):
        Va = bf16c(Vf).astype(np.complex64) if PREC == "bf16_dev_pess" else Vf.copy()
        rel = TOPUP if PREC == "bf16_dev_pess" else float(PREC[5:])
        s = rel * float(np.linalg.norm(Vf)) / np.sqrt(2.0 * Vf.size)
        for lo in range(0, Va.size, 1 << 24):      # chunked: a P^3 float64 draw is 1.1 GB
            hi = min(lo + (1 << 24), Va.size)
            Va[lo:hi] += (rng_p.normal(0, s, hi - lo)
                          + 1j * rng_p.normal(0, s, hi - lo)).astype(np.complex64)
    else:
        Va = Vf                              # fp32/tex8/bf16acc store fp32, as the control does
    print(f"  arm volume rel L2 {np.linalg.norm(Va-Vf)/np.linalg.norm(Vf):.4e}"
          f"   device round-off target {DEVICE_REL_L2:.4e}  top-up {TOPUP:.4e}", flush=True)

    acc = {k: np.zeros(P ** 3, dtype=np.complex128) for k in ("ref", "arm")}
    wsum = {k: np.zeros(P ** 3) for k in ("ref", "arm")}
    buf = {k: {"i": [], "v": [], "w": []} for k in ("ref", "arm")}

    def flush(kind):
        b = buf[kind]
        if not b["i"]:
            return
        fi, fv, fw = (np.concatenate(b[x]) for x in ("i", "v", "w"))
        acc[kind] += np.bincount(fi, fv.real, P ** 3) + 1j * np.bincount(fi, fv.imag, P ** 3)
        wsum[kind] += np.bincount(fi, fw, P ** 3)
        b["i"].clear(); b["v"].clear(); b["w"].clear()
        if kind == "arm" and PREC == "bf16acc":
            acc[kind] = bf16c(acc[kind].astype(np.complex64)).astype(np.complex128)
            wsum[kind] = bf16(wsum[kind].astype(np.float32)).astype(np.float64)

    for o in range(NORIENT):
        A = H.rand_rot(rng)
        p, uu, vv, u3, v3 = H.slice_coords(rmax, A, PAD)
        npts = p.shape[0]
        noise = None
        for kind in ("ref", "arm"):
            if kind == "ref":
                idx, wt = H.tri_weights(p, P)
                val = (Vf[idx] * wt).sum(0)
            else:
                idx, wt = weights(p, uu, vv, u3, v3, P)
                if PREC in ("bf16", "bf16_dev", "bf16_dev_pess"):
                    wt = bf16(wt.astype(np.float32)).astype(np.float64)
                    val = bf16c((Va[idx].astype(np.complex64) * wt.astype(np.float32)).sum(0))
                elif PREC in ("fp32", "tex8", "bf16acc"):
                    val = (Va[idx].astype(np.complex64)
                           * wt.astype(np.float32)).sum(0).astype(np.complex128)
                else:
                    val = (Va[idx] * wt).sum(0)
                if PREC == "bf16_dev" or (PREC.startswith("pert") and not PREC.startswith("pertS")):
                    rel = TOPUP if PREC == "bf16_dev" else float(PREC[4:])
                    sd = rel * np.sqrt(np.mean(np.abs(val) ** 2)) / np.sqrt(2.0)
                    val = val + (rng_p.normal(0, sd, npts) + 1j * rng_p.normal(0, sd, npts))
            if noise is None:
                sig = np.sqrt(np.mean(np.abs(val) ** 2))
                sd = sig / np.sqrt(max(SNR, 1e-9)) / np.sqrt(2.0)
                noise = rng.normal(0, sd, npts) + 1j * rng.normal(0, sd, npts)
            val = val + noise
            buf[kind]["i"].append(idx.reshape(-1))
            buf[kind]["v"].append((wt * val[None, :]).reshape(-1))
            buf[kind]["w"].append(np.abs(wt).reshape(-1))
        if (o + 1) % FLUSH == 0:
            for kind in ("ref", "arm"):
                flush(kind)
        if (o + 1) % 100 == 0:
            print(f"  {o+1}/{NORIENT}  ({time.time()-t0:.0f}s)", flush=True)

    for kind in ("ref", "arm"):
        flush(kind)
    res = {"box": N, "pad": P, "norient": NORIENT, "apix": APIX, "snr": SNR, "seed": SEED,
           "variant": VARIANT, "precision": PREC, "bar_A": BAR_A,
           "device_rel_l2": DEVICE_REL_L2, "arms": {}}
    print()
    for kind in ("ref", "arm"):
        f = fsc_planewise(acc.pop(kind), wsum.pop(kind), V.reshape(-1), P, rmax)
        r, kk = H.resolution(f, N)
        res["arms"][kind] = {"resolution_A": r, "shell": kk, "fsc": f.tolist()}
        print(f"{kind:4s}: FSC 0.143 at shell {kk:6.2f} -> {r:7.3f} A", flush=True)
    d = res["arms"]["arm"]["resolution_A"] - res["arms"]["ref"]["resolution_A"]
    res["delta_A"] = d
    res["pass"] = bool(abs(d) <= BAR_A)
    print(f"\n{VARIANT}/{PREC} minus RELION trilinear fp64: {d:+.4f} A   (bar {BAR_A} A) -> "
          f"{'PASS' if abs(d) <= BAR_A else 'FAIL'}", flush=True)
    out = HERE / f"p3fsc_box{N}_snr{SNR}_s{SEED}_{VARIANT}_{PREC}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"-> {out.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
