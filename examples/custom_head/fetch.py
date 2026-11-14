"""Download each PDB entry, write a single-sequence Boltz-2 input and the true Cb contacts.

The input sequence is the residues the deposited structure resolves in its first polymer chain,
so every predicted token has a true position to score against.
"""

import sys
import urllib.request
from pathlib import Path

import gemmi
import numpy as np


def cb(res):
    atom = res.find_atom("CB", "*") or res.find_atom("CA", "*")
    return np.array(atom.pos.tolist()) if atom else None


def fetch(pdb_id, out):
    text = urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_id}.cif").read().decode()
    st = gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    st.setup_entities()
    chain = st[0][0]
    poly = [r for r in chain.get_polymer() if cb(r) is not None]
    seq = "".join(gemmi.find_tabulated_residue("MET" if r.name == "MSE" else r.name).one_letter_code.upper()
                  for r in poly)
    xyz = np.stack([cb(r) for r in poly])
    contacts = (np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) < 8.0).astype(np.float32)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "inputs" / f"{pdb_id.lower()}.yaml").write_text(
        f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n      msa: empty\n")
    np.save(out / "truth" / f"{pdb_id.lower()}.npy", contacts)
    i, j = np.triu_indices(len(seq), k=6)
    print(f"{pdb_id}: {len(seq)} residues, {int(contacts[i, j].sum())} contacts at |i-j| >= 6")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for pdb_id in sys.argv[2:]:
        fetch(pdb_id, out)
