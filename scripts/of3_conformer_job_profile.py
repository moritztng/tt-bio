#!/usr/bin/env python3
"""Per-part cost of one OF3 reference-conformer job (sequential, no pool). Usage: PYTHONPATH=<tree> python scripts/of3_conformer_job_profile.py perf/size512/fixtures/cdk2x2_512.yaml"""
import time, random, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from biotite.interface.rdkit import to_mol
import yaml
from tt_bio._vendor.openfold3.core.data.primitives.structure import query as Q, conformer as C
from tt_bio._vendor.openfold3.core.data.resources.residues import MoleculeType, PROTEIN_RESTYPE_1TO3, MOLECULE_TYPE_TO_LEAVING_ATOMS
seq = yaml.safe_load(open(sys.argv[1]))["sequences"][0]["protein"]["sequence"]
leav = MOLECULE_TYPE_TO_LEAVING_ATOMS[MoleculeType.PROTEIN]
jobs = [(Q.atom_array_from_ccd_code(PROTEIN_RESTYPE_1TO3[c], chain_id="A", res_id=i+1, molecule_type=MoleculeType.PROTEIN), leav) for i,c in enumerate(seq)]
print(len(jobs))
T = {}
def t(k, f, *a, **kw):
    s=time.perf_counter(); r=f(*a, **kw); T[k]=T.get(k,0)+time.perf_counter()-s; return r
for rep in range(2):
    T.clear(); random.seed(0)
    s0=time.perf_counter()
    for arr, lv in jobs:
        C.pool_seed(random.randint(0,10**9))
        mask = t("mask", lambda: ~np.isin(arr.atom_name, lv))
        mol = t("to_mol", to_mol, arr, kekulize=True)
        t("sanitize", Chem.SanitizeMol, mol)
        mol.RemoveConformer(0)
        t("smiles", Chem.MolToSmiles, mol)
        molH = t("addhs", Chem.AddHs, mol)
        st = AllChem.ETKDGv3(); st.clearConfs=False; st.randomSeed = C._etkdg_seed()
        t("embed_direct", AllChem.EmbedMolecule, molH, st)
        molH2 = Chem.AddHs(mol)
        st.randomSeed = 1
        t("embed_functimeout", C.func_timeout, timeout=120, func=AllChem.EmbedMolecule, args=(molH2, st))
        m3 = t("removehs", C.safe_remove_all_hs, molH)
        t("annot", lambda: Q.set_atomwise_annotation(Q.set_atomwise_annotation(m3, "atom_name", arr.atom_name), "used_atom_mask", [True]*m3.GetNumAtoms()))
    tot=time.perf_counter()-s0
    random.seed(0); C.pool_reset()
    s=time.perf_counter()
    for arr, lv in jobs:
        C.pool_seed(random.randint(0,10**9)); Q.processed_reference_molecule_from_atom_array(arr, atoms_to_mask=lv)
    full=time.perf_counter()-s
    print(f"rep{rep} instrumented total {tot*1e3:.0f} ms; real seq full {full*1e3:.0f} ms ({full/len(jobs)*1e3:.2f} ms/job)")
    for k,v in sorted(T.items(), key=lambda x:-x[1]): print(f"  {k:18s} {v/len(jobs)*1e3:.3f} ms/job")
