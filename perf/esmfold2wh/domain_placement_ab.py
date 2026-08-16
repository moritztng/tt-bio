"""Score inter-domain placement for the ESMFold2 MSA depth-cap A/B.

The question p2 left open: does bounding the MSA to 5120 rows move DOMAINS relative to
each other? cdk2_640 could not answer it (chimera + floppy linker, and a half split cuts
through a domain), so this scores a natural multi-domain single chain against its
experimental structure.

Domain boundaries are derived FROM the experimental structure, not asserted: a spectral
bipartition of the CA contact graph, which is the standard way to cut a chain into
compact units. The boundary it finds is printed so it can be checked against the
literature.

Metrics, all over the residues the experiment actually observes:
  * per-domain Kabsch RMSD  -- is each domain internally right
  * domain-2 RMSD after superposing on domain 1 only, and vice versa -- the direct
    inter-domain placement error
  * inter-domain rotation discrepancy in degrees
  * TM-score (sequence-dependent, normalised by the experimental length)
  * lDDT, superposition-free
Reported for each arm against the experiment, and for the two arms against each other.
"""
import json
import sys

import gemmi
import numpy as np


def kabsch(mob, ref):
    """Rotation+translation taking `mob` onto `ref`. Returns (R, t, rmsd)."""
    mc, rc = mob.mean(0), ref.mean(0)
    H = (mob - mc).T @ (ref - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = rc - R @ mc
    rmsd = float(np.sqrt((((mob @ R.T + t) - ref) ** 2).sum(1).mean()))
    return R, t, rmsd


def apply(R, t, x):
    return x @ R.T + t


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def tm_score(mob, ref):
    """Sequence-dependent TM-score of `mob` onto `ref`, normalised by len(ref)."""
    L = len(ref)
    d0 = 1.24 * (L - 15) ** (1 / 3) - 1.8 if L > 21 else 0.5
    d0 = max(d0, 0.5)
    best = 0.0
    for seed in [L] + [w for w in (L // 2, L // 4, 32, 16, 8, 4) if 4 <= w < L]:
        for start in range(0, L - seed + 1, max(1, seed // 2)):
            idx = np.arange(start, start + seed)
            for _ in range(30):
                if len(idx) < 3:
                    break
                R, t, _ = kabsch(mob[idx], ref[idx])
                d = np.linalg.norm(apply(R, t, mob) - ref, axis=1)
                best = max(best, float((1.0 / (1.0 + (d / d0) ** 2)).mean()))
                cut = d0
                new = np.where(d < cut)[0]
                while len(new) < 3 and cut < 20:
                    cut += 0.5
                    new = np.where(d < cut)[0]
                if len(new) == len(idx) and np.array_equal(new, idx):
                    break
                idx = new
    return best


def lddt(mob, ref, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Superposition-free CA lDDT of `mob` against `ref`."""
    dr = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    dm = np.linalg.norm(mob[:, None] - mob[None], axis=-1)
    mask = (dr < cutoff) & ~np.eye(len(ref), dtype=bool)
    dd = np.abs(dm - dr)[mask]
    per = [float((dd < th).mean()) for th in thresholds]
    return 100.0 * float(np.mean(per)), float(dd.mean()), float(np.median(dd))


def spectral_domains(xyz, contact=8.0):
    """Split a chain into two compact domains by the Fiedler vector of its CA contact
    graph. Returns a boolean mask (True = domain 1)."""
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    W = np.exp(-(d ** 2) / (2 * contact ** 2))
    np.fill_diagonal(W, 0.0)
    deg = W.sum(1)
    Ln = np.eye(len(W)) - (W / np.sqrt(np.outer(deg, deg)))
    w, v = np.linalg.eigh(Ln)
    f = v[:, np.argsort(w)[1]]
    m = f > 0
    return m if m.sum() >= (~m).sum() else ~m


def segments(ids, mask):
    """Contiguous residue-id runs where `mask` holds, as (start, end) pairs."""
    out, run = [], None
    for i, keep in zip(ids, mask):
        if keep and run is None:
            run = [int(i), int(i)]
        elif keep:
            run[1] = int(i)
        elif run is not None:
            out.append(tuple(run))
            run = None
    if run is not None:
        out.append(tuple(run))
    return out


def read_ca(path, use_label_seq):
    """{residue index -> CA xyz} plus {residue index -> B-factor}, chain A polymer."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    xyz, bf = {}, {}
    for ch in st[0]:
        poly = ch.get_polymer()
        if len(poly) < 50:
            continue
        for res in poly:
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            key = res.label_seq if use_label_seq else res.seqid.num
            if key is None:
                continue
            xyz[int(key)] = np.array([ca.pos.x, ca.pos.y, ca.pos.z])
            bf[int(key)] = ca.b_iso
        break
    return xyz, bf


def pair_metrics(mob, ref, d1):
    """Every placement number for one (mobile, reference) pair."""
    d2 = ~d1
    _, _, glob = kabsch(mob, ref)
    R1, t1, rms1 = kabsch(mob[d1], ref[d1])
    R2, t2, rms2 = kabsch(mob[d2], ref[d2])
    ld, mean_dd, med_dd = lddt(mob, ref)
    return {
        "global_rmsd": glob,
        "domain1_rmsd": rms1,
        "domain2_rmsd": rms2,
        "domain2_rmsd_on_domain1_frame": float(
            np.sqrt(((apply(R1, t1, mob[d2]) - ref[d2]) ** 2).sum(1).mean())),
        "domain1_rmsd_on_domain2_frame": float(
            np.sqrt(((apply(R2, t2, mob[d1]) - ref[d1]) ** 2).sum(1).mean())),
        "interdomain_rotation_deg": rot_angle(R2 @ R1.T),
        "tm_score": tm_score(mob, ref),
        "lddt": ld,
        "mean_abs_dd": mean_dd,
        "median_abs_dd": med_dd,
    }


def main(exp_path, arm_paths, out_json):
    exp_xyz, _ = read_ca(exp_path, use_label_seq=True)
    arms = {tag: read_ca(p, use_label_seq=False) for tag, p in arm_paths.items()}
    ids = sorted(set(exp_xyz) & set.intersection(*[set(a[0]) for a in arms.values()]))
    ids = np.array(ids)
    E = np.array([exp_xyz[i] for i in ids])
    M = {tag: np.array([a[0][i] for i in ids]) for tag, a in arms.items()}

    d1 = spectral_domains(E)
    res = {
        "n_scored_residues": int(len(ids)),
        "domain1_segments": segments(ids, d1),
        "domain2_segments": segments(ids, ~d1),
        "domain1_size": int(d1.sum()),
        "domain2_size": int((~d1).sum()),
        "vs_experiment": {t: pair_metrics(M[t], E, d1) for t in M},
        "plddt": {t: {"mean": float(np.mean([a[1][i] for i in ids])),
                      "min": float(np.min([a[1][i] for i in ids]))}
                  for t, a in arms.items()},
    }
    tags = list(M)
    if len(tags) == 2:
        res["arm_vs_arm"] = pair_metrics(M[tags[1]], M[tags[0]], d1)
        res["arm_vs_arm_order"] = f"{tags[1]} onto {tags[0]}"
    print(json.dumps(res, indent=2))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    exp = sys.argv[1]
    out = sys.argv[-1]
    paths = dict(a.split("=", 1) for a in sys.argv[2:-1])
    main(exp, paths, out)
