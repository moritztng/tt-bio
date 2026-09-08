import numpy as np, sys

def atoms(p):
    """(label, xyz) per ATOM row of an mmCIF written by tt_bio."""
    rows, hdr, inloop = [], [], False
    for ln in open(p):
        s = ln.strip()
        if s.startswith('_atom_site.'):
            hdr.append(s.split('.')[1]); inloop = True; continue
        if inloop and (s.startswith('ATOM') or s.startswith('HETATM')):
            f = s.split()
            d = dict(zip(hdr, f))
            rows.append(((d['label_atom_id'], d['label_comp_id'], d['label_seq_id'], d['label_asym_id']),
                         (float(d['Cartn_x']), float(d['Cartn_y']), float(d['Cartn_z']))))
        elif inloop and rows and not (s.startswith('ATOM') or s.startswith('HETATM')):
            break
    return [r[0] for r in rows], np.array([r[1] for r in rows])

la, A = atoms(sys.argv[1])
lb, B = atoms(sys.argv[2])
assert la == lb, "atom ordering differs between the two files"
print("atoms compared: %d" % len(A))
d = np.linalg.norm(A - B, axis=1)
print("as-written (no superposition): rmsd %.4f A, max %.4f A" % (np.sqrt((d**2).mean()), d.max()))
# Kabsch: rotate B onto A, then measure. R maps B-space into A-space.
Ac, Bc = A - A.mean(0), B - B.mean(0)
U, S, Vt = np.linalg.svd(Bc.T @ Ac)
D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
R = U @ D @ Vt
Br = Bc @ R
d2 = np.linalg.norm(Ac - Br, axis=1)
print("superposed:                    rmsd %.4f A, max %.4f A" % (np.sqrt((d2**2).mean()), d2.max()))
# negative control: the superposition must NOT flatten an actually-different structure.
rng = np.random.default_rng(0)
Bs = Bc[rng.permutation(len(Bc))]
U, S, Vt = np.linalg.svd(Bs.T @ Ac); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
dn = np.linalg.norm(Ac - Bs @ (U @ D @ Vt), axis=1)
print("neg control (shuffled atoms):  rmsd %.4f A" % np.sqrt((dn**2).mean()))
