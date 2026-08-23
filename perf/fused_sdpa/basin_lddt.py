#!/usr/bin/env python3
"""Is the arms' difference a basin flip or a real accuracy loss?

Global CA RMSD on cdk2x2_298 has a per-seed spread of ~1.0 A against a margin of 0.06 A, so it
cannot separate the two triangle-attention arms. Two host-only tests on the twelve CIFs already
committed under perf/fused_sdpa/seeds/, no device:

  1. the 12x12 pairwise CA RMSD matrix. If the folds cluster into a small number of tight groups
     that do NOT respect the arm boundary, whole-chain RMSD is measuring which basin the sampler
     drew, and no number of seeds turns that into an accuracy measure.
  2. CA-lDDT against 1HCL (Mariani 2013: inclusion radius 15 A on the reference, thresholds
     0.5/1/2/4 A, mean over thresholds). lDDT is superposition-free and local, so a rigid
     inter-lobe hinge rotation barely moves it. If lDDT agrees across arms while RMSD does not,
     that is the basin reading confirmed, and lDDT is the metric with the power.

Paired bootstrap over residues gives an honest CI on the arm difference in lDDT.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))
from cif_rmsd import kabsch_rmsd            # noqa: E402
from of3_score_ref import ca_map            # noqa: E402

ARMS = ["def", "hifi"]
R0 = 15.0
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
RNG = np.random.default_rng(0)


def folds(arm):
    return sorted((HERE / "seeds" / arm).glob("f*_seed*"))


def coords(cif, keys):
    m = ca_map(cif)
    return np.array([m[k][1] for k in keys])


def lddt_per_residue(pred_xyz, ref_xyz):
    """Standard lDDT restricted to CA. Returns (per_residue, global)."""
    dp = np.linalg.norm(pred_xyz[:, None] - pred_xyz[None], axis=-1)
    dr = np.linalg.norm(ref_xyz[:, None] - ref_xyz[None], axis=-1)
    n = len(ref_xyz)
    inc = (dr < R0) & ~np.eye(n, dtype=bool)          # reference-defined inclusion set
    err = np.abs(dp - dr)
    hit = np.stack([(err < t) & inc for t in THRESHOLDS]).sum(0) / len(THRESHOLDS)
    per_res = np.divide(hit.sum(1), inc.sum(1), out=np.zeros(n), where=inc.sum(1) > 0)
    return per_res, float(hit[inc].sum() / inc.sum())


def main():
    gt = ca_map(HERE / "cifs" / "1hcl.cif")
    all_cifs = {a: folds(a) for a in ARMS}
    first = ca_map(all_cifs["def"][0] / "cdk2x2_298.cif")
    keys = sorted(set(first) & set(gt))
    bad = [k for k in keys if first[k][0] != gt[k][0]]
    assert not bad, f"identity mismatch {bad[:5]}"
    ref = np.array([gt[k][1] for k in keys])
    print(f"scored CA: {len(keys)}  (fold residues {len(first)}, 1HCL {len(gt)})")

    labels, xyz = [], []
    for a in ARMS:
        for d in all_cifs[a]:
            labels.append(f"{a}/{d.name}")
            xyz.append(coords(d / "cdk2x2_298.cif", keys))

    # 1. pairwise RMSD matrix
    n = len(labels)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = kabsch_rmsd(xyz[i], xyz[j])
    print("\n=== pairwise CA Kabsch RMSD (A) ===")
    print("            " + " ".join(f"{l.split('/')[1][:6]:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:>11} " + " ".join(f"{M[i,j]:7.3f}" for j in range(n)))

    # single-linkage clustering at 0.5 A to name the basins
    thr = 0.5
    cluster = [-1] * n
    c = 0
    for i in range(n):
        if cluster[i] >= 0:
            continue
        stack, cluster[i] = [i], c
        while stack:
            k = stack.pop()
            for j in range(n):
                if cluster[j] < 0 and M[k, j] < thr:
                    cluster[j] = c
                    stack.append(j)
        c += 1
    print(f"\nsingle-linkage clusters at {thr} A:")
    for cc in range(c):
        mem = [labels[i] for i in range(n) if cluster[i] == cc]
        print(f"  basin {cc}: {mem}")

    # 2. lDDT
    per_res, glob = {}, {}
    for l, X in zip(labels, xyz):
        pr, g = lddt_per_residue(X, ref)
        per_res[l], glob[l] = pr, g
    print("\n=== CA-lDDT vs 1HCL, and global CA RMSD vs 1HCL ===")
    print(f"{'fold':>11} {'lDDT':>8} {'RMSD':>8} {'basin':>6}")
    for i, l in enumerate(labels):
        print(f"{l:>11} {glob[l]:8.5f} {kabsch_rmsd(xyz[i], ref):8.4f} {cluster[i]:6d}")

    out = {"n_ca": len(keys), "labels": labels, "cluster": cluster,
           "pairwise_rmsd": M.round(4).tolist(),
           "lddt": {l: round(glob[l], 6) for l in labels},
           "rmsd_gt": {l: round(float(kabsch_rmsd(xyz[i], ref)), 4) for i, l in enumerate(labels)}}

    # paired arm comparison on the 5 distinct seeds (f0..f4), plus the f5 A/A repeat separately
    for tag, idxs in (("seeds0-4", range(5)), ("with_f5_repeat", range(6))):
        d, dl = [], []
        for k in idxs:
            a, b = f"def/{all_cifs['def'][k].name}", f"hifi/{all_cifs['hifi'][k].name}"
            d.append(glob[b] - glob[a])
            dl.append(per_res[b] - per_res[a])
        d = np.array(d)
        dv = [glob["def/" + all_cifs["def"][k].name] for k in idxs]
        hv = [glob["hifi/" + all_cifs["hifi"][k].name] for k in idxs]
        print(f"\n--- lDDT margin (hifi - def), {tag}: per-seed {np.round(d,5).tolist()}")
        print(f"    mean {d.mean():+.5f}   arm spreads def {max(dv)-min(dv):.5f} hifi {max(hv)-min(hv):.5f}")
        # residue bootstrap on the seed-averaged per-residue margin
        mean_pr = np.mean(dl, axis=0)
        boot = np.array([mean_pr[RNG.integers(0, len(mean_pr), len(mean_pr))].mean()
                         for _ in range(20000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"    residue-bootstrap 95% CI on mean per-residue lDDT margin: "
              f"[{lo:+.5f}, {hi:+.5f}]  (point {mean_pr.mean():+.5f})")
        out[f"lddt_margin_{tag}"] = {"per_seed": d.round(6).tolist(),
                                     "mean": round(float(d.mean()), 6),
                                     "boot_ci95": [round(float(lo), 6), round(float(hi), 6)]}

    (HERE / "basin_lddt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {HERE/'basin_lddt.json'}")


if __name__ == "__main__":
    main()
