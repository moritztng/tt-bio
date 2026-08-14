#!/usr/bin/env python3
"""E3 -- is a whole refinement through the bridge still RELION's answer?

`p3_compare.py` asked this of ONE iteration and got 4452/4452 bit-identical assignments. This asks it
of the whole trajectory, which is a different question: a bit-exact single iteration says nothing
about a loop whose output is its own next input.

Four checks, in RELION's own currency, over both arms of e2e_campaign.sh:

  1. gold-standard FSC 0.143 within each arm (half1 vs half2). The number the brief asks for. The
     machinery is projprobe/e3_fsc_relion.py's, verified against relion_postprocess to -0.0031 A.
  2. cross-FSC between the arms' corresponding half-maps. Sharper than 1: if the coarse scores agree,
     ref half1 and tt half1 are the same volume and the cross-FSC is 1.0 to the Nyquist edge.
  3. per-particle assignments from each arm's final run_data.star, all five columns.
  4. per-iteration reassignment rate, ref against tt, iteration by iteration. This is the one that
     answers compounding: a rate that grows over the trajectory is feedback, a flat one is not.

Plus sha256 per output map, REPORTED BUT NOT USED AS AN IDENTITY CHECK: relion-acc-backend §4.8 ran
the reference arm twice and the half-map sha256 differed while all 4,452 assignments and the FSC
crossing were identical. RELION's ALTCPU reconstruction is not bit-reproducible run to run, so a sha
difference here is noise, not a bridge effect.

Paths are the stable clone, not a worktree: p3_compare.py imports from
/home/ttuser/.coworker/wt/relion-acc-backend/projprobe, which the concluded task took with it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

S = Path("/home/ttuser/relion-scratch")
sys.path.insert(0, str(S / "tt-bio" / "projprobe"))
from e3_fsc_relion import fsc, read_mrc, resolution  # noqa: E402

E2E = S / "e2e"
ARMS = {"ref_relion_own": "ref_run", "tt_bridge_active": "tt_run"}
COLS = ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi",
        "_rlnOriginXAngst", "_rlnOriginYAngst")


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


def half_maps(stem):
    """Converged auto-refine writes run_half1_class001_unfil.mrc with no _itNNN. Fall back to the
    last per-iteration map if the arm stopped before convergence, and say which was used."""
    a = E2E / f"{stem}_half1_class001_unfil.mrc"
    b = E2E / f"{stem}_half2_class001_unfil.mrc"
    if a.exists() and b.exists():
        return a, b, "converged"
    its = sorted(E2E.glob(f"{stem}_it0*_half1_class001.mrc"))
    if not its:
        return None, None, "missing"
    last = its[-1].name.split("_half1")[0]
    return (E2E / f"{last}_half1_class001.mrc"), (E2E / f"{last}_half2_class001.mrc"), last


def data_star(stem):
    p = E2E / f"{stem}_data.star"
    if p.exists():
        return p
    its = sorted(E2E.glob(f"{stem}_it0*_data.star"))
    return its[-1] if its else None


def main():
    res = {"arms": {}}
    vols = {}
    print("=== per-arm gold-standard FSC 0.143 (half1 vs half2, unmasked) ===", flush=True)
    for name, stem in ARMS.items():
        h1p, h2p, which = half_maps(stem)
        if h1p is None or not h1p.exists():
            print(f"  {name}: MISSING half maps for {stem}", flush=True)
            continue
        h1, apix = read_mrc(h1p)
        h2, _ = read_mrc(h2p)
        vols[name] = (h1, h2, apix)
        N = h1.shape[0]
        f = fsc(np.fft.fftshift(np.fft.fftn(h1)), np.fft.fftshift(np.fft.fftn(h2)), N, N // 2)
        r, kk = resolution(f, N, apix)
        res["arms"][name] = {"source": which, "box": N, "apix": apix, "resol_A": r, "shell": kk,
                             "sha_half1": sha(h1p), "sha_half2": sha(h2p)}
        print(f"  {name:18s} [{which}] box {N} apix {apix:.6f}  0.143 -> {r:.4f} A", flush=True)
        print(f"      sha256/16 half1 {sha(h1p)}  half2 {sha(h2p)}   "
              f"(reported, NOT an identity check: ALTCPU reconstruction is not bit-reproducible)",
              flush=True)

    if len(vols) == 2:
        (a1, a2, apix) = vols["ref_relion_own"]
        (t1, t2, _) = vols["tt_bridge_active"]
        N = a1.shape[0]
        d = res["arms"]["tt_bridge_active"]["resol_A"] - res["arms"]["ref_relion_own"]["resol_A"]
        res["delta_resol_A"] = d
        res["within_0.1A"] = bool(abs(d) <= 0.1)
        print(f"\n=== resolution delta at FSC 0.143 ===\n  tt - ref = {d:+.4f} A"
              f"   (bar: within 0.1 A)", flush=True)

        print("\n=== cross-FSC, ref half-k vs tt half-k (1.0 means the same volume) ===", flush=True)
        for k, (va, vt) in enumerate(((a1, t1), (a2, t2)), start=1):
            f = fsc(np.fft.fftshift(np.fft.fftn(va)), np.fft.fftshift(np.fft.fftn(vt)), N, N // 2)
            r, _ = resolution(f, N, apix)
            lo = float(np.min(f[1:N // 2 + 1]))
            rl2 = float(np.linalg.norm(va - vt) / max(np.linalg.norm(va), 1e-30))
            res[f"cross_half{k}"] = {"min_fsc": lo, "resol_A": r, "rel_l2": rl2}
            print(f"  half{k}: min FSC over shells 1..{N//2} = {lo:+.6f}   "
                  f"0.143 crossing {r:.4f} A   rel L2 {rl2:.3e}", flush=True)

    print("\n=== final per-particle assignments ===", flush=True)
    pa, pt = data_star("ref_run"), data_star("tt_run")
    if pa and pt:
        print(f"  ref {pa.name}   tt {pt.name}", flush=True)
        for c in COLS:
            va = np.array([float(x) for x in star_col(pa, c)])
            vt = np.array([float(x) for x in star_col(pt, c)])
            if va.size and va.size == vt.size:
                same = int((va == vt).sum())
                mx = float(np.max(np.abs(va - vt)))
                res.setdefault("assign_final", {})[c] = {
                    "n": int(va.size), "identical": same, "max_abs": mx}
                print(f"  {c:20s} n={va.size}  identical {same}/{va.size}  "
                      f"max |delta| {mx:.6g}", flush=True)
            else:
                print(f"  {c:20s} size mismatch {va.size} vs {vt.size}", flush=True)

    # The compounding check. A single iteration cannot show feedback; a trajectory can.
    print("\n=== reassignment rate per iteration (does any disagreement compound?) ===", flush=True)
    res["per_iteration"] = {}
    for p in sorted(E2E.glob("ref_run_it0*_data.star")):
        it = p.name.split("_it")[1][:3]
        q = E2E / f"tt_run_it{it}_data.star"
        if not q.exists():
            continue
        row = {}
        for c in COLS:
            va = np.array([float(x) for x in star_col(p, c)])
            vt = np.array([float(x) for x in star_col(q, c)])
            if va.size and va.size == vt.size:
                row[c] = {"n": int(va.size), "reassigned": int((va != vt).sum()),
                          "max_abs": float(np.max(np.abs(va - vt)))}
        if row:
            res["per_iteration"][it] = row
            worst = max(row.values(), key=lambda r: r["reassigned"])
            print(f"  it{it}: worst column reassigned {worst['reassigned']}/{worst['n']} "
                  f"= {100.0*worst['reassigned']/worst['n']:.3f}%", flush=True)
    if len(res["per_iteration"]) >= 2:
        ks = sorted(res["per_iteration"])
        rate = [max(v["reassigned"] for v in res["per_iteration"][k].values()) for k in ks]
        res["compounding"] = {"iterations": ks, "worst_reassigned": rate,
                              "grows": bool(rate[-1] > rate[0])}
        print(f"  trajectory {ks} -> {rate}   "
              f"{'GROWS (compounding)' if rate[-1] > rate[0] else 'flat or shrinking'}", flush=True)

    out = E2E / "e2e_compare.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
