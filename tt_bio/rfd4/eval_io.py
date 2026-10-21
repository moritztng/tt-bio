"""Read an RFD4 annotated CIF into what the evaluate metrics need, without atomworks.

RFD4 writes its conditioning back into the design CIF: per-atom columns on `atom_site`
(`condition_sequence_1_residue`, `atomize`, `condition_CoordinateGroup_1_atom`, ...) and per-atom
loops keyed by atom index for vector-valued conditions (`_condition_coordinate_1_atom`). Upstream
reads them through atomworks' `parse(..., STANDARD_ANNOTATIONS_ONLY)`; this reads the same columns
with biotite and returns them as plain arrays.
"""

from dataclasses import dataclass, field

import numpy as np
from biotite.structure.io.pdbx import CIFFile, get_structure

# atomworks `constants.py:484`: the code an unconditioned C_SEQ residue carries.
MASKED = "<M>"

# rfproteina `conditions.py:482-498`, in upstream's order.
SEQUENCE_PROBABILITY_TOKENS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "C", "G", "U", "DA", "DC", "DG", "DT",
)  # fmt: skip

# rfproteina `conditions.py:303-313`: reserved CoordinateGroup values.
CRDGRP_UNKNOWN = -1
CRDGRP_ABSOLUTE = 0

_OPTIONAL_COLUMNS = (
    "condition_sequence_1_residue",
    "atomize",
    "condition_CoordinateGroup_1_atom",
    "condition_secondary-structure_1_residue",
    "condition_hotspot_1_atom",
    "annotation_src-coordinate-conditioned-region_1_atom",
    "condition_rasa-binned_1_atom",
    *(f"condition_sequence-probability-{t}_1_residue" for t in SEQUENCE_PROBABILITY_TOKENS),
)


@dataclass
class Design:
    atoms: object  # biotite AtomArray, bonds included
    is_polymer: np.ndarray
    is_protein: np.ndarray
    c_seq: np.ndarray  # per atom, MASKED where unconditioned
    atomize: np.ndarray
    is_crd: np.ndarray  # C_CRD mask: any finite conditioned coordinate
    crd_group: np.ndarray
    seq_prob: np.ndarray = field(default=None)  # (n_atoms, 28), NaN where unconditioned
    c_sec: np.ndarray = None  # SSEnum ints per atom; None when the CIF has no C_SEC column at all
    hotspot: np.ndarray = None  # C_HOT per atom, False where unconditioned
    crd_src: np.ndarray = None  # SA_CRDSRC: the atom's coordinates came from conditioning
    coords_with_insertion: np.ndarray = None  # SA_CWI (n_atoms, 3), NaN where not conditioned
    rasa: np.ndarray = None  # C_RAS RasaEnum ints, 0 where unconditioned
    centroid: np.ndarray = None  # C_CENTROID (n_atoms, 3), NaN where unconditioned


def _atom_site_columns(block):
    return set(block["atom_site"].keys())


def _per_atom_loop(block, name, n_atoms, width):
    """A `_<name>.idx0 / .val0..` loop scattered to a dense (n_atoms, width) array, NaN elsewhere."""
    out = np.full((n_atoms, width), np.nan)
    if name not in block:
        return out
    cat = block[name]
    idx = cat["idx0"].as_array(int)
    for k in range(width):
        out[idx, k] = cat[f"val{k}"].as_array(float)
    return out


def _chain_types_from_residues(atoms):
    from biotite.structure import filter_amino_acids, filter_nucleotides

    aa, na = filter_amino_acids(atoms), filter_nucleotides(atoms)
    is_polymer = np.zeros(atoms.array_length(), dtype=bool)
    is_protein = np.zeros(atoms.array_length(), dtype=bool)
    for chain in np.unique(atoms.chain_id):
        sel = atoms.chain_id == chain
        is_polymer[sel] = bool((aa[sel] | na[sel]).all())
        is_protein[sel] = bool(aa[sel].all())
    return is_polymer, is_protein


def _as_bool(values):
    return np.isin(np.char.lower(values.astype(str)), ("true", "1", "yes", "y"))


def read_design(path) -> Design:
    cif = CIFFile.read(str(path))
    block = cif.block
    present = _atom_site_columns(block)
    extra = [c for c in _OPTIONAL_COLUMNS if c in present]
    atoms = get_structure(cif, model=1, include_bonds=True, extra_fields=extra)
    n = atoms.array_length()

    poly_entities = set(block["entity_poly"]["entity_id"].as_array(str)) if "entity_poly" in block else set()
    protein_entities = set()
    if "entity_poly" in block:
        ep = block["entity_poly"]
        for eid, typ in zip(ep["entity_id"].as_array(str), ep["type"].as_array(str)):
            if typ.lower().startswith("polypeptide"):
                protein_entities.add(eid)
    if poly_entities:
        entity = block["atom_site"]["label_entity_id"].as_array(str)
        is_polymer = np.isin(entity, list(poly_entities))
        is_protein = np.isin(entity, list(protein_entities))
    else:
        # No `_entity_poly` (a CIF written by a plain structure writer): atomworks then infers the
        # chain type from its residues. A chain is a polymer when every residue in it is an amino
        # acid or a nucleotide, and protein when every residue is an amino acid.
        is_polymer, is_protein = _chain_types_from_residues(atoms)

    def ann(name, default):
        return atoms.get_annotation(name) if name in extra else np.full(n, default)

    c_seq = ann("condition_sequence_1_residue", MASKED).astype(str)
    atomize = _as_bool(ann("atomize", "False"))
    crd_group = ann("condition_CoordinateGroup_1_atom", str(CRDGRP_UNKNOWN)).astype(float).astype(int)
    crd = _per_atom_loop(block, "condition_coordinate_1_atom", n, 3)
    seq_prob = np.stack(
        [
            ann(f"condition_sequence-probability-{t}_1_residue", "nan").astype(str)
            for t in SEQUENCE_PROBABILITY_TOKENS
        ],
        axis=-1,
    )
    seq_prob = np.where(np.isin(seq_prob, (".", "?")), "nan", seq_prob).astype(float)
    return Design(
        atoms=atoms,
        is_polymer=is_polymer,
        is_protein=is_protein,
        c_seq=c_seq,
        atomize=atomize,
        is_crd=np.isfinite(crd).any(axis=1),
        crd_group=crd_group,
        seq_prob=seq_prob,
        hotspot=_as_bool(ann("condition_hotspot_1_atom", "False")),
        crd_src=ann("annotation_src-coordinate-conditioned-region_1_atom", ".").astype(str) != ".",
        coords_with_insertion=_per_atom_loop(block, "annotation_coords-with-insertion_1_atom", n, 3),
        rasa=ann("condition_rasa-binned_1_atom", "0").astype(str),
        centroid=_per_atom_loop(block, "condition_ChainCentroid_1_atom", n, 3),
        c_sec=(
            atoms.get_annotation("condition_secondary-structure_1_residue").astype(float).astype(int)
            if "condition_secondary-structure_1_residue" in extra
            else None
        ),
    )
