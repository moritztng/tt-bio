"""Add the entity / entity_poly / entity_poly_seq blocks biotite does not write.

protenix derives an entity SEQUENCE only from the `entity_poly` category (parser.entity_poly_type);
with no such block every chain is classified as a ligand and cif_to_input_json emits no "sequence",
which is the KeyError prepare-msa hit.
"""
import sys
import numpy as np
from biotite.structure.io.pdbx import CIFFile, CIFCategory, get_structure
from biotite.structure import get_residue_starts
from biotite.sequence import ProteinSequence

path = sys.argv[1]
f = CIFFile.read(path)
blk = f.block
arr = get_structure(f, model=1)
starts = get_residue_starts(arr)
res3 = arr.res_name[starts].tolist()
one = "".join(ProteinSequence.convert_letter_3to1(r) for r in res3)

blk["entity"] = CIFCategory({"id": "1", "type": "polymer",
                             "pdbx_description": "cropped chain"})
blk["entity_poly"] = CIFCategory({
    "entity_id": "1", "type": "polypeptide(L)", "nstd_linkage": "no", "nstd_monomer": "no",
    "pdbx_seq_one_letter_code": one, "pdbx_seq_one_letter_code_can": one,
    "pdbx_strand_id": "A"})
n = len(res3)
blk["entity_poly_seq"] = CIFCategory({
    "entity_id": ["1"] * n, "num": [str(i + 1) for i in range(n)], "mon_id": res3})
a = blk["atom_site"]
a["label_entity_id"] = np.full(arr.array_length(), "1")
f.write(path)
print("  %s: %d residues, entity_poly seq len %d" % (path.split("/")[-1], n, len(one)))
