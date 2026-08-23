#!/usr/bin/env python3
"""Re-score OpenFold3's fused-SDPA arms with lDDT, because the metric that rejected them cannot see.

`state/openfold3-fused-sdpa-gpu-reference-check.md` rejected the OF3 fused arm (`P`) on all-atom
Kabsch RMSD to a rented H200 reference: an 8.879 A margin at 298 aa against a reference whose own
five-seed spread that same doc reports as 17.834 A. That is n = 1 device fold per arm compared
inside a spread twice the size of the margin, which is the exact defect execution pass 1 refuted
for RF3 (`state/fused-sdpa-adopt.md` §0).

Same folds, already paid for, three metrics that can see through sampler noise:

  1. CA-lDDT against the 1HCL crystal (Mariani 2013, 15 A reference-defined radius, 0.5/1/2/4 A
     thresholds), for both arms AND for each of the five H200 reference seeds. The reference's own
     lDDT range is the yardstick: if `P` sits inside it, the rejection has no power.
  2. the P - on lDDT margin with a residue bootstrap CI.
  3. the pairwise CA RMSD matrix over {on, P, ref_seed0..4} and its 0.5 A single-linkage
     clustering, to say whether OF3's 8.879 A margin is also just a basin assignment.

cdk2x2_512 is CDK2 followed by its own residues 1-214, so every pair set is restricted to
WITHIN-SEGMENT pairs via of3_score_ref.GT_SEGMENTS. Cross-segment distances are an artifact of the
chimera and would manufacture a difference
(memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`).

Acceptance gate: the CA-RMSD-to-1HCL numbers the prior doc reports (298 on 9.437 / P 15.821;
512 copy1 6.662 / 8.144; copy2 5.759 / 6.148) must reproduce to 3 decimals before any lDDT number
here is believed. If they do not, the parser or the segment mapping is wrong.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))
from cif_rmsd import kabsch_rmsd                       # noqa: E402
from of3_score_ref import ca_map, GT_SEGMENTS          # noqa: E402
from basin_lddt import lddt_per_residue                # noqa: E402

RNG = np.random.default_rng(0)
# from state/openfold3-fused-sdpa-gpu-reference-check.md §3, computed by perf/of3_ref/score.py
ACCEPT = {(298, "whole"): {"on": 9.437, "P": 15.821},
          (512, "copy1(res 1-298)"): {"on": 6.662, "P": 8.144},
          (512, "copy2(res 299-512)"): {"on": 5.759, "P": 6.148}}


def seg_rmsd_gt(fold_ca, gt, pairs):
    """of3_score_ref.gt_rmsd's per-structure key set, so the acceptance gate is comparable."""
    use = [(a, b) for a, b in pairs if a in fold_ca and b in gt]
    bad = [(a, b) for a, b in use if fold_ca[a][0] != gt[b][0]]
    assert not bad, f"residue identity mismatch vs 1HCL: {bad[:5]}"
    A = np.array([fold_ca[a][1] for a, _ in use])
    B = np.array([gt[b][1] for _, b in use])
    return float(kabsch_rmsd(A, B)), len(use)


def main():
    ap = argparse.ArgumentParser()
    d = Path("/tmp/of3ref/perf/of3_ref")
    ap.add_argument("--cifs", type=Path, default=d / "cifs")
    ap.add_argument("--refdir", type=Path, default=d / "ref")
    ap.add_argument("--gt", type=Path, default=d / "cifs" / "1hcl.cif")
    ap.add_argument("--out", type=Path, default=HERE / "of3_lddt.json")
    a = ap.parse_args()

    gt = ca_map(a.gt)
    report = {"gt": {"file": a.gt.name, "resolved_ca": len(gt)}, "sizes": {}}

    for size in (298, 512):
        arms = {arm: sorted(a.cifs.glob(f"{size}_{arm}_*.cif")) for arm in ("on", "P")}
        if not arms["on"]:
            continue
        refs = sorted(a.refdir.glob(f"ref_{size}_seed*.cif"),
                      key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
        # one CIF per arm carries the verdict; the second is the device A/A repeat (0.000000 A)
        struct = {"on": arms["on"][0], "P": arms["P"][0]}
        for r in refs:
            struct["ref_seed" + re.search(r"seed(\d+)", r.name).group(1)] = r
        cas = {k: ca_map(p) for k, p in struct.items()}
        block = {"cifs": {k: p.name for k, p in struct.items()}}
        print(f"\n===== {size} aa =====   structures: {list(struct)}")

        for label, pairs in GT_SEGMENTS[size].items():
            # --- acceptance gate, per-structure key set, same logic as of3_score_ref.gt_rmsd
            acc = {}
            for k, m in cas.items():
                acc[k], n_k = seg_rmsd_gt(m, gt, pairs)
            want = ACCEPT.get((size, label), {})
            for arm, v in want.items():
                got = acc[arm]
                assert abs(got - v) < 1e-3, \
                    f"ACCEPTANCE FAILED {size} {label} {arm}: got {got:.6f}, doc says {v}"
            if want:
                print(f"  acceptance gate [{label}]: " +
                      "  ".join(f"{k} {acc[k]:.3f}=={v}" for k, v in want.items()) + "  OK")

            # --- common key set across every structure, for lDDT and the pairwise matrix
            keys = [(f, g) for f, g in pairs
                    if g in gt and all(f in m for m in cas.values())]
            bad = [(f, g) for f, g in keys if cas["on"][f][0] != gt[g][0]]
            assert not bad, f"identity mismatch {bad[:5]}"
            ref_xyz = np.array([gt[g][1] for _, g in keys])
            xyz = {k: np.array([m[f][1] for f, _ in keys]) for k, m in cas.items()}
            names = list(xyz)
            print(f"  [{label}] {len(keys)} common CA positions")

            # --- 1. lDDT vs the crystal
            pr, glob = {}, {}
            for k in names:
                pr[k], glob[k] = lddt_per_residue(xyz[k], ref_xyz)
            refvals = [glob[k] for k in names if k.startswith("ref_seed")]

            # --- 3. pairwise CA RMSD + 0.5 A single linkage
            n = len(names)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    M[i, j] = M[j, i] = kabsch_rmsd(xyz[names[i]], xyz[names[j]])
            cl = [-1] * n
            c = 0
            for i in range(n):
                if cl[i] >= 0:
                    continue
                st, cl[i] = [i], c
                while st:
                    u = st.pop()
                    for j in range(n):
                        if cl[j] < 0 and M[u, j] < 0.5:
                            cl[j] = c
                            st.append(j)
                c += 1

            print(f"    {'structure':>12} {'lDDT':>8} {'RMSDgt':>8} {'basin':>6}")
            for i, k in enumerate(names):
                print(f"    {k:>12} {glob[k]:8.5f} {acc[k]:8.3f} {cl[i]:6d}")
            print(f"    reference lDDT range: [{min(refvals):.5f}, {max(refvals):.5f}]"
                  f"   P {glob['P']:.5f} "
                  f"{'INSIDE' if min(refvals) <= glob['P'] <= max(refvals) else 'OUTSIDE'}"
                  f"   on {glob['on']:.5f} "
                  f"{'INSIDE' if min(refvals) <= glob['on'] <= max(refvals) else 'OUTSIDE'}")
            print(f"    basins: " + "; ".join(
                f"{cc}:{[names[i] for i in range(n) if cl[i]==cc]}" for cc in range(c)))
            print("    pairwise CA RMSD (A)")
            print("        " + " ".join(f"{k[:9]:>9}" for k in names))
            for i, k in enumerate(names):
                print(f"    {k:>9} " + " ".join(f"{M[i,j]:9.3f}" for j in range(n)))

            # --- 2. margin, fused minus shipped, with a residue bootstrap
            dpr = pr["P"] - pr["on"]
            boot = np.array([dpr[RNG.integers(0, len(dpr), len(dpr))].mean()
                             for _ in range(20000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"    lDDT margin (P - on) = {glob['P']-glob['on']:+.5f}"
                  f"   per-residue point {dpr.mean():+.5f}"
                  f"   bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]")

            block[label] = {
                "n_common_ca": len(keys),
                "acceptance_rmsd_gt": {k: round(v, 6) for k, v in acc.items()},
                "lddt": {k: round(glob[k], 6) for k in names},
                "ref_lddt_range": [round(min(refvals), 6), round(max(refvals), 6)],
                "P_inside_ref_range": bool(min(refvals) <= glob["P"] <= max(refvals)),
                "on_inside_ref_range": bool(min(refvals) <= glob["on"] <= max(refvals)),
                "lddt_margin_P_minus_on": round(float(glob["P"] - glob["on"]), 6),
                "lddt_margin_boot_ci95": [round(float(lo), 6), round(float(hi), 6)],
                "pairwise_rmsd": {"names": names, "M": M.round(4).tolist(), "cluster": cl},
            }
        report["sizes"][str(size)] = block

    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
