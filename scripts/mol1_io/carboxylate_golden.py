"""Golden for tests/test_oxt_repair.py: Mol-1's own fix_cterminal_carboxyl_oxygens on seeded random coordinates.

    /home/moritz/mol1-upstream/io-venv/bin/python -I scripts/mol1_io/carboxylate_golden.py \
        tests/data/mol1_io/carboxylate_golden.npz

Layout: protein chain A (GLY, ALA terminal with OXT), protein chain B (one ALA without OXT, so no rebuild), a
three-atom ligand chain L. Four samples, so O is the nearer oxygen in some and OXT in others.
"""
import sys

import numpy as np
import torch
from biotite.structure import AtomArray
from mol.runner.inference import fix_cterminal_carboxyl_oxygens

atoms = ([("A", 1, n) for n in ("N", "CA", "C", "O")] + [("A", 2, n) for n in ("N", "CA", "C", "O", "CB", "OXT")]
         + [("B", 1, n) for n in ("N", "CA", "C", "O", "CB")] + [("L", 1, n) for n in ("C1", "C2", "O1")])
arr = AtomArray(len(atoms))
arr.set_annotation("chain_id", np.array([a[0] for a in atoms]))
arr.res_id[:] = [a[1] for a in atoms]
arr.set_annotation("atom_name", np.array([a[2] for a in atoms]))
coords = torch.randn(4, len(atoms), 3, generator=torch.Generator().manual_seed(0)) * 2
out = fix_cterminal_carboxyl_oxygens(coords, arr)
np.savez(sys.argv[1], coords=coords.numpy(), out=out.numpy(), chain=arr.chain_id, res_id=arr.res_id,
         atom_name=arr.atom_name)
