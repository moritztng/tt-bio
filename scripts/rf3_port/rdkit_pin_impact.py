#!/usr/bin/env python3
"""Score what pinning rdkit changes for the tt-bio models that already used it.

RF3 pins ``rdkit==2026.03.4`` because its chirality features come out of RDKit's
chiral-centre perception, which moves between releases. tt-bio was previously
unpinned, so the pin can change Boltz-2, BoltzGen, Protenix-v2 and OpenFold3 too.
This scores that directly, card-free, by running the exact conformer-generation
calls each of those paths makes and dumping the geometry.

Run it once under each rdkit version and diff the two JSON outputs.

Which paths can even be affected:

- ``tt_bio/data/parse.py:compute_3d_conformer`` (Boltz-2) and
  ``tt_bio/boltzgen/data/parse/schema.py`` leave ``ETKDGv3().randomSeed`` at its
  default of -1, i.e. random per call. Those conformers are already not
  reproducible run to run, so a version change cannot make them less so; what it
  could move is the distribution. Reported as UNSEEDED.
- ``tt_bio/protenix_data.py:_smiles_to_mol`` fixes ``randomSeed = 0xF00D``.
- OpenFold3 draws its seed from the global ``random``, so it is reproducible under
  a seeded caller.

Both of the seeded paths are therefore rdkit-version-sensitive by construction,
and are what this compares.
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

import rdkit

# Ligands the repo's own examples actually use, plus the two RF3 fixture ligands.
LIGANDS = {
    "imidazole": "[nH]1cc[nH+]c1",
    "atp_like": "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O",
    "chiral_sugar": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "steroid": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen_chiral": "CC(C)Cc1ccc(cc1)[C@H](C)C(O)=O",
}


def _coords(mol) -> list:
    conf = mol.GetConformer()
    return np.round(np.array(conf.GetPositions()), 6).tolist()


def protenix_path(smiles: str) -> dict:
    """Exactly tt_bio/protenix_data.py::_smiles_to_mol."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return {"coords": _coords(mol), "n_atoms": mol.GetNumAtoms()}


def openfold3_path(smiles: str, seed: int = 42) -> dict:
    """OpenFold3's strategy: ETKDGv3 with the seed drawn from the global random."""
    random.seed(seed)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    strategy = AllChem.ETKDGv3()
    strategy.clearConfs = False
    strategy.randomSeed = random.randint(1, 1_000_000)
    with rdBase.BlockLogs():
        AllChem.EmbedMolecule(mol, strategy)
    mol = Chem.RemoveHs(mol)
    return {"coords": _coords(mol), "n_atoms": mol.GetNumAtoms()}


def chirality(smiles: str) -> dict:
    """What RF3's chiral features are built from."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    return {"centres": [[int(i), str(t)] for i, t in centres]}


def main() -> int:
    out = {"rdkit": rdkit.__version__, "ligands": {}}
    for name, smiles in LIGANDS.items():
        entry = {}
        for label, fn in (
            ("protenix_seeded", protenix_path),
            ("openfold3_seeded", openfold3_path),
            ("chirality", chirality),
        ):
            try:
                entry[label] = fn(smiles)
            except Exception as exc:  # a failure is itself a difference
                entry[label] = {"error": f"{type(exc).__name__}: {exc}"}
        out["ligands"][name] = entry
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(f"rdkit {rdkit.__version__} -> {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
