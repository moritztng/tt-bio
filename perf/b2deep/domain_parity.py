#!/usr/bin/env python3
"""Per-domain parity scoring for the cdk2x2_N perf fixtures.

Why this exists. cdk2x2_N is CDK2 (PDB 1HCL, 298 aa) concatenated with a truncated second copy of
itself as ONE chain. It was built to scale token count for timing and it does that well. It cannot
score parity: the two pseudo-domains have no evolved interface, so their relative orientation is the
softest degree of freedom in the structure, and ANY bit-level perturbation flips it. Four
independent non-bit-exact arms (S1 fused-SDPA, L1, L2, L1+L2) all scored 7.7-9.0 A global all-atom
RMSD with both domains internally intact and hinge rotations of 45-72 deg. A number invariant to
which change produced it is not measuring the change.

So: superpose each domain separately. A lever that only re-rounds shows sub-A domain RMSD and a
large hinge angle. A lever that breaks something shows domain RMSD that is not sub-A. For a verdict
that does not need this decomposition at all, fold cdk2x2_298 -- pure CDK2, one real domain, no
hinge -- where the plain global RMSD is valid again.

    domain_parity.py <ref.cif> <arm.cif> [<arm.cif> ...]
"""
import sys
import numpy as np

BACKBONE = {"N", "CA", "C", "O"}
JUNCTION = 299          # residues 1..298 are CDK2 copy 1, 299.. are the truncated copy 2


def read_cif(path):
    """Minimal mmCIF _atom_site reader -> [(label_seq_id, label_atom_id, xyz)]."""
    rows, cols, inloop = [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".")[1])
            inloop = True
            continue
        if inloop and (s.startswith("#") or s == ""):
            if rows:
                break
            continue
        if inloop and cols and not s.startswith("_"):
            f = s.split()
            if len(f) >= len(cols):
                rows.append(f)
    i = {c: n for n, c in enumerate(cols)}
    return [(int(f[i["label_seq_id"]]), f[i["label_atom_id"]],
             np.array([float(f[i["Cartn_x"]]), float(f[i["Cartn_y"]]), float(f[i["Cartn_z"]])]))
            for f in rows]


def kabsch(P, Q):
    """RMSD after optimal superposition, and the rotation that achieved it."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, W = np.linalg.svd(Pc.T @ Qc)
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ W))]) @ W
    return float(np.sqrt((((Pc @ R) - Qc) ** 2).sum(1).mean())), R


def select(atoms, lo, hi, mode):
    return np.array([a[2] for a in atoms if lo <= a[0] <= hi
                     and (mode == "all" or (mode == "bb" and a[1] in BACKBONE)
                          or (mode == "ca" and a[1] == "CA"))])


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    ref = read_cif(argv[1])
    last = max(a[0] for a in ref)
    monomer = last < JUNCTION
    if monomer:
        print(f"{last} residues, below the {JUNCTION} junction: one real domain, global RMSD is valid")
    print("%-28s %-4s %9s %9s %9s %9s" % ("arm", "mode", "global", "dom1", "dom2", "hinge deg"))
    for path in argv[2:]:
        arm = read_cif(path)
        if len(arm) != len(ref):
            print(f"{path}: {len(arm)} atoms against the reference's {len(ref)}, skipped")
            continue
        for mode in ("all", "ca"):
            g, _ = kabsch(select(arm, 1, last, mode), select(ref, 1, last, mode))
            if monomer:
                print("%-28s %-4s %9.3f %9s %9s %9s"
                      % (path.split("/")[-2], mode, g, "-", "-", "-"))
                continue
            r1, R1 = kabsch(select(arm, 1, JUNCTION - 1, mode), select(ref, 1, JUNCTION - 1, mode))
            r2, R2 = kabsch(select(arm, JUNCTION, last, mode), select(ref, JUNCTION, last, mode))
            rel = R1.T @ R2
            ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
            print("%-28s %-4s %9.3f %9.3f %9.3f %9.1f"
                  % (path.split("/")[-2], mode, g, r1, r2, ang))


if __name__ == "__main__":
    main(sys.argv)
