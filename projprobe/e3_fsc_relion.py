#!/usr/bin/env python3
"""E3.6 part 1 -- our gold-standard FSC machinery, run on RELION's OWN half-maps.

The brief asks for resolution at FSC 0.143 against RELION's own output on the same data, and not a
relative L2. Before that comparison can mean anything, our FSC has to reproduce RELION's number on
RELION's own volumes -- otherwise a disagreement later cannot be attributed. This script does exactly
that and nothing else:

  read  Refine3D/job019/run_half{1,2}_class001_unfil.mrc   (the two independent half-set maps)
  FSC   shell-binned, unmasked, in unpadded frequency units
  cross 0.143, linearly interpolated, converted to Angstrom via N * apix / shell

RELION reports for this job, in run.out:
    Auto-refine: + Final resolution (without masking) is: 3.79378

The shell binning and the 0.143 interpolation are lifted unchanged from projprobe/s2p2_fsc.py, which
the precision pass verified. Only PAD (1 here, the maps are not padded) and the pixel size (read from
the MRC header rather than assumed) differ.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOB = Path("/home/ttuser/.coworker/wt/relion-estep-integration/scratch/Tutorial5.0/Refine3D/job019")

MODES = {0: ("i1", 1), 1: ("<i2", 2), 2: ("<f4", 4), 6: ("<u2", 2), 12: ("<f2", 2)}


def read_mrc(p):
    """Minimal MRC reader: the 1024-byte header plus NSYMBT bytes of extended header."""
    raw = p.read_bytes()
    nx, ny, nz, mode = struct.unpack("<4i", raw[0:16])
    mx, my, mz = struct.unpack("<3i", raw[28:40])
    xlen, ylen, zlen = struct.unpack("<3f", raw[40:52])
    nsymbt = struct.unpack("<i", raw[92:96])[0]
    dt, _ = MODES[mode]
    off = 1024 + nsymbt
    a = np.frombuffer(raw[off:off + nx * ny * nz * np.dtype(dt).itemsize], dtype=dt)
    # MRC is stored z-slowest; the array is [nz, ny, nx]
    vol = a.reshape(nz, ny, nx).astype(np.float64)
    apix = xlen / mx if mx else 1.0
    return vol, apix


def fsc(a, b, P, rmax):
    """Shell-binned FSC. Identical to projprobe/s2p2_fsc.py's, with PAD = 1."""
    c = P // 2
    ax = np.arange(P) - c
    R2 = (ax[:, None, None].astype(np.float32) ** 2 + ax[None, :, None].astype(np.float32) ** 2
          + ax[None, None, :].astype(np.float32) ** 2)
    sh = np.round(np.sqrt(R2)).astype(np.int32)
    m = sh <= rmax
    sh, A, B = sh[m], a[m], b[m]
    num = np.bincount(sh, np.real(A * np.conj(B)), rmax + 1)
    da = np.bincount(sh, np.abs(A) ** 2, rmax + 1)
    db = np.bincount(sh, np.abs(B) ** 2, rmax + 1)
    return num / np.sqrt(np.maximum(da * db, 1e-30))


def resolution(f, N, apix):
    """First shell where FSC drops below 0.143, linearly interpolated, in Angstrom."""
    for k in range(1, len(f)):
        if f[k] < 0.143:
            t = (f[k - 1] - 0.143) / max(f[k - 1] - f[k], 1e-12)
            kk = (k - 1) + t
            return (N * apix) / max(kk, 1e-9), kk
    return (N * apix) / max(len(f) - 1, 1), float(len(f) - 1)


def main():
    h1, apix = read_mrc(JOB / "run_half1_class001_unfil.mrc")
    h2, _ = read_mrc(JOB / "run_half2_class001_unfil.mrc")
    N = h1.shape[0]
    assert h1.shape == h2.shape, (h1.shape, h2.shape)
    print(f"RELION half-maps  {h1.shape}  apix {apix:.6f} A", flush=True)

    F1 = np.fft.fftshift(np.fft.fftn(h1))
    F2 = np.fft.fftshift(np.fft.fftn(h2))
    rmax = N // 2
    f = fsc(F1, F2, N, rmax)
    r, kk = resolution(f, N, apix)
    relion_reported = 3.79378
    print(f"  gold-standard FSC 0.143 at shell {kk:.3f} -> {r:.4f} A", flush=True)
    print(f"  RELION reports (run.out, without masking):     {relion_reported:.4f} A", flush=True)
    print(f"  difference {r - relion_reported:+.4f} A", flush=True)
    # the curve, so the crossing is checkable rather than asserted
    print("  FSC by shell (every 8th):", flush=True)
    for k in range(0, rmax + 1, 8):
        print(f"    shell {k:3d}  {N*apix/max(k,1e-9):8.2f} A  FSC {f[k]:+.4f}", flush=True)
    out = HERE / "e3_fsc_relion.json"
    out.write_text(json.dumps({"box": N, "apix": apix, "shell_0143": kk, "resol_A": r,
                               "relion_reported_A": relion_reported,
                               "delta_A": r - relion_reported,
                               "fsc": f.tolist()}, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
