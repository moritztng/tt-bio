#!/usr/bin/env python3
"""P3 -- did RELION still produce RELION's answer when its coarse compare went through the bridge.

Three checks, in RELION's own currency, on the two arms of p3_arms.sh:

  1. gold-standard FSC 0.143 within each arm (half1 vs half2), and arm to arm. This is the number the
     brief asks for. The machinery is projprobe/e3_fsc_relion.py's, verified against
     relion_postprocess to -0.0031 A at the crossing.
  2. cross-FSC between the two arms' corresponding half-maps, which is the sharper test: if the coarse
     scores agree, arm A's half1 and arm T's half1 are the same volume and the cross-FSC is 1.0 to the
     Nyquist edge.
  3. _rlnGoldStandardFsc from each arm's own run_it013_model.star, arm to arm. RELION's in-refinement
     value, a different quantity from 1 and 2, compared only arm to arm.

Plus sha256 per output map, and the per-particle orientation assignments from run_it013_data.star,
because a coarse-score bug shows up as reassigned particles before it shows up in a resolution number.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ttuser/.coworker/wt/relion-acc-backend/projprobe")
from e3_fsc_relion import fsc, read_mrc, resolution  # noqa: E402

P3 = Path("/home/ttuser/relion-scratch/p3")
ARMS = {"A_bridge_declines": "a_run", "T_bridge_active": "t_run"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def star_col(path, key):
    """Pull one loop column out of a STAR file, by label."""
    labels, rows, in_loop, col = [], [], False, None
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("_rln"):
            labels.append(s.split()[0])
            in_loop = True
            continue
        if in_loop and s and not s.startswith(("_", "#", "data_", "loop_")):
            if col is None:
                if key not in labels:
                    labels, in_loop = [], False
                    continue
                col = labels.index(key)
            f = s.split()
            if len(f) > col:
                rows.append(f[col])
        elif in_loop and not s:
            if col is not None:
                break
            labels, in_loop = [], False
    return rows


def gold_std(path):
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip().startswith("_rlnGoldStandardFsc"):
            pass
    return None


def main():
    res = {}
    vols = {}
    print("=== per-arm gold-standard FSC (half1 vs half2, unmasked) ===", flush=True)
    for name, stem in ARMS.items():
        h1p = P3 / f"{stem}_it013_half1_class001.mrc"
        h2p = P3 / f"{stem}_it013_half2_class001.mrc"
        if not h1p.exists():
            print(f"  {name}: MISSING {h1p.name}", flush=True)
            continue
        h1, apix = read_mrc(h1p)
        h2, _ = read_mrc(h2p)
        vols[name] = (h1, h2, apix)
        N = h1.shape[0]
        F1 = np.fft.fftshift(np.fft.fftn(h1))
        F2 = np.fft.fftshift(np.fft.fftn(h2))
        f = fsc(F1, F2, N, N // 2)
        r, kk = resolution(f, N, apix)
        res[name] = {"box": N, "apix": apix, "resol_A": r, "shell": kk,
                     "sha_half1": sha(h1p), "sha_half2": sha(h2p)}
        print(f"  {name:22s} box {N} apix {apix:.6f}  0.143 at shell {kk:.3f} -> {r:.4f} A", flush=True)
        print(f"      sha256/16 half1 {res[name]['sha_half1']}  half2 {res[name]['sha_half2']}",
              flush=True)

    if len(vols) == 2:
        (a1, a2, apix) = vols["A_bridge_declines"]
        (t1, t2, _) = vols["T_bridge_active"]
        N = a1.shape[0]
        print("\n=== arm-to-arm resolution difference at FSC 0.143 ===", flush=True)
        d = res["T_bridge_active"]["resol_A"] - res["A_bridge_declines"]["resol_A"]
        print(f"  T - A = {d:+.4f} A   (bar from relion-precision-fsc.md: within 0.1 A)", flush=True)
        res["delta_resol_A"] = d
        res["within_0.1A"] = bool(abs(d) <= 0.1)

        print("\n=== cross-FSC, arm A half-k vs arm T half-k (1.0 means the same volume) ===",
              flush=True)
        for k, (va, vt) in enumerate(((a1, t1), (a2, t2)), start=1):
            Fa = np.fft.fftshift(np.fft.fftn(va))
            Ft = np.fft.fftshift(np.fft.fftn(vt))
            f = fsc(Fa, Ft, N, N // 2)
            r, kk = resolution(f, N, apix)
            lo = float(np.min(f[1:N // 2 + 1]))
            rl2 = float(np.linalg.norm(va - vt) / max(np.linalg.norm(va), 1e-30))
            res[f"cross_half{k}"] = {"min_fsc": lo, "resol_A": r, "rel_l2": rl2}
            print(f"  half{k}: min FSC over shells 1..{N//2} = {lo:+.6f}   "
                  f"0.143 crossing {r:.4f} A   rel L2 {rl2:.3e}", flush=True)

    print("\n=== RELION's own in-refinement _rlnGoldStandardFsc, arm to arm ===", flush=True)
    for name, stem in ARMS.items():
        p = P3 / f"{stem}_it013_model.star"
        if not p.exists():
            print(f"  {name}: MISSING {p.name}", flush=True)
            continue
        vals = star_col(p, "_rlnGoldStandardFsc")
        if vals:
            v = np.array([float(x) for x in vals])
            k = int(np.argmax(v < 0.143)) if (v < 0.143).any() else len(v) - 1
            res.setdefault(name, {})["gsfsc_shell_0143"] = k
            print(f"  {name:22s} {len(v)} shells, first below 0.143 at shell {k}", flush=True)

    print("\n=== per-particle assignments from run_it013_data.star ===", flush=True)
    cols = ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi",
            "_rlnOriginXAngst", "_rlnOriginYAngst")
    pa = P3 / "a_run_it013_data.star"
    pt = P3 / "t_run_it013_data.star"
    if pa.exists() and pt.exists():
        for c in cols:
            va = np.array([float(x) for x in star_col(pa, c)])
            vt = np.array([float(x) for x in star_col(pt, c)])
            if va.size and va.size == vt.size:
                same = int((va == vt).sum())
                mx = float(np.max(np.abs(va - vt)))
                res.setdefault("assign", {})[c] = {"n": int(va.size), "identical": same, "max_abs": mx}
                print(f"  {c:20s} n={va.size}  identical {same}/{va.size}  max |delta| {mx:.6g}",
                      flush=True)
            else:
                print(f"  {c:20s} size mismatch {va.size} vs {vt.size}", flush=True)

    out = P3 / "p3_compare.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
