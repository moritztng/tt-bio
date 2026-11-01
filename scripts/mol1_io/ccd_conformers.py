"""Cut Mol-1's CCD table (protenix_data.write_conformer_table) down to the given codes, for test goldens.

    python scripts/mol1_io/ccd_conformers.py <components.cif.rdkit_mol.pkl> <out.npz> CODE [CODE ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.protenix_data import write_conformer_table  # noqa: E402

if __name__ == "__main__":
    print(write_conformer_table(sys.argv[1], sys.argv[2], set(sys.argv[3:])), "codes")
