"""Mol-1's reference conformer for every CCD code: the coordinates get_ccd_ref_info reads
(mol.GetConformer(mol.ref_conf_id)), in the mol's atom order, with the atom names."""
import pickle, sys
import numpy as np

pkl, out = sys.argv[1], sys.argv[2]
mols = pickle.load(open(pkl, "rb"))
codes, counts, names, xyz = [], [], [], []
for code, mol in mols.items():
    n = mol.GetNumAtoms()
    if n == 0:
        continue
    pos = mol.GetConformer(mol.ref_conf_id).GetPositions()
    inv = {i: a for a, i in mol.atom_map.items()}
    codes.append(code); counts.append(n)
    names += [inv[i] for i in range(n)]; xyz.append(pos.astype(np.float32))
np.savez_compressed(out, codes=np.array(codes), counts=np.array(counts, np.int32), names=np.array(names),
                    xyz=np.concatenate(xyz))
print(len(codes), sum(counts))
