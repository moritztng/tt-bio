"""TFG input features: the `constraint` block resolved against tt-bio's atom layout.

Port of OpenDDE v1.2.0 (Apache-2.0) opendde/data/inference/json_to_feature.py
(`_handle_constraint_field`, `parse_contact_restraints`, `_resolve_contact_atom_name`,
`build_user_distance_restraint_features`, `build_rigid_movable_features`,
`build_epitope_features`, `get_a_bond_atom`). Error messages are upstream's verbatim.

Addressing is upstream's: `entity` is the 1-based index of the input's `sequences` entry,
`copy` the 1-based copy within that entry (YAML `id: [H, L]` is one entity, H copy 1, L copy 2),
`position` the 1-based residue in the sequence. tt-bio's own `entity_id` feature groups chains by
identical sequence instead, so it is not used here; the grouping comes from `entities` (one list
of chain ids per `sequences` entry, see `entity_groups`). Atom indices index tt-bio's atom axis
(the concatenated `ref_*` features), so they are tt-bio's order, not upstream's.
"""
from __future__ import annotations

import logging
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..opendde_data import _atom_names
from ..protenix_data import MOL_TYPE_IDS

logger = logging.getLogger(__name__)

_SUPPORTED = {"contact", "movable_chains", "epitope"}
_EPITOPE_KEYS = {"residues", "paratope", "min_fraction"}
CDR_WINDOWS = [[24, 40], [50, 66], [88, 115]]
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_AA1TO3 = {v: k for k, v in _AA3TO1.items()}


def _strict_int(value: Any, label: str) -> int:
    """An integer, or a string of digits. Booleans and fractional numbers are refused."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got the boolean {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{label} must be an integer (or a string of digits), got {value!r}.")


def _strict_number(value: Any, label: str) -> float:
    """A finite number. Booleans, strings, null, NaN and infinities are refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    return float(value)


def _parse_contacts(contacts: list) -> list[dict]:
    """upstream parse_contact_restraints."""
    parsed = []
    for ci, d in enumerate(contacts):
        if not isinstance(d, dict):
            raise ValueError(f"constraint.contact[{ci}] must be an object.")
        sides = []
        for idx, side in enumerate(["left", "right"]):
            entity_raw = d.get(f"{side}_entity", d.get(f"entity{idx + 1}"))
            if entity_raw is None:
                raise ValueError(
                    f"constraint.contact[{ci}] missing entity{idx + 1}/{side}_entity.")
            entity_id = _strict_int(entity_raw, f"constraint.contact[{ci}].entity{idx + 1}")
            copy_id = d.get(f"{side}_copy", d.get(f"copy{idx + 1}"))
            if copy_id is not None:
                copy_id = _strict_int(copy_id, f"constraint.contact[{ci}].copy{idx + 1}")
            position_raw = d.get(f"{side}_position", d.get(f"position{idx + 1}"))
            if position_raw is None:
                raise ValueError(
                    f"constraint.contact[{ci}] missing position{idx + 1}/{side}_position.")
            position = _strict_int(position_raw, f"constraint.contact[{ci}].position{idx + 1}")
            sides.append({"entity_id": entity_id, "copy_id": copy_id, "position": position,
                          "atom_name": d.get(f"{side}_atom", d.get(f"atom{idx + 1}"))})
        lo = _strict_number(d.get("min_distance", 3.5), f"constraint.contact[{ci}].min_distance")
        hi = _strict_number(d.get("max_distance", 8.0), f"constraint.contact[{ci}].max_distance")
        if lo < 0.0:
            raise ValueError(f"constraint.contact[{ci}]: min_distance ({lo}) must be >= 0.")
        if hi < lo:
            raise ValueError(
                f"constraint.contact[{ci}]: max_distance ({hi}) < min_distance ({lo}).")
        parsed.append({"left": sides[0], "right": sides[1], "min_distance": lo,
                       "max_distance": hi})
    return parsed


def _parse_movable(chains: Any) -> list[str] | None:
    """The structure-free half of upstream build_rigid_movable_features."""
    if chains is None:
        return None
    if not isinstance(chains, list) or not chains:
        raise ValueError("constraint.movable_chains must be a non-empty list of chain ids.")
    wanted = [str(c) for c in chains]
    if len(set(wanted)) != len(wanted):
        raise ValueError("constraint.movable_chains contains a repeated chain id.")
    return wanted


def _parse_epitope(epitope: Any, movable: list[str] | None) -> dict | None:
    """The structure-free half of upstream build_epitope_features, in upstream's check order."""
    if epitope is None:
        return None
    if not isinstance(epitope, dict):
        raise ValueError("constraint.epitope must be an object.")
    unknown = sorted(set(epitope.keys()) - _EPITOPE_KEYS)
    if unknown:
        raise ValueError(f"Unknown constraint.epitope field(s): {unknown}.")
    if movable is None:
        raise ValueError(
            "constraint.epitope requires constraint.movable_chains (the antibody chains that move).")
    residues = epitope.get("residues")
    if not isinstance(residues, list) or not residues:
        raise ValueError("constraint.epitope.residues must be a non-empty list.")
    min_fraction = _strict_number(epitope.get("min_fraction", 0.5),
                                  "constraint.epitope.min_fraction")
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("constraint.epitope.min_fraction must be in (0, 1].")
    items = []
    for idx, item in enumerate(residues):
        if not isinstance(item, dict):
            raise ValueError(f"constraint.epitope.residues[{idx}] must be an object.")
        extra = sorted(set(item.keys()) - {"entity", "copy", "position", "residue"})
        if extra:
            raise ValueError(f"constraint.epitope.residues[{idx}]: unknown field(s) {extra}.")
        for key in ("entity", "copy", "position"):
            if item.get(key) is None:
                raise ValueError(f"constraint.epitope.residues[{idx}] missing {key}.")
        items.append({
            "entity": _strict_int(item["entity"], f"constraint.epitope.residues[{idx}].entity"),
            "copy": _strict_int(item["copy"], f"constraint.epitope.residues[{idx}].copy"),
            "position": _strict_int(item["position"],
                                    f"constraint.epitope.residues[{idx}].position"),
            "residue": item.get("residue"),
        })
    spec = epitope.get("paratope", "all")
    if spec in ("all", "cdr"):
        paratope = spec
    elif isinstance(spec, dict) and set(spec.keys()) == {"windows"} and isinstance(spec["windows"], dict):
        paratope = {}
        for chain, spans in spec["windows"].items():
            if str(chain) not in movable:
                raise ValueError(
                    f"constraint.epitope.paratope.windows names {chain!r}, which is not a movable chain.")
            if (not isinstance(spans, list) or not spans
                    or any(not isinstance(w, list) or len(w) != 2 for w in spans)):
                raise ValueError(
                    f"constraint.epitope.paratope.windows[{chain!r}] must be a list of [low, high] pairs.")
            pairs = [[_strict_int(w[0], f"windows[{chain!r}] low"),
                      _strict_int(w[1], f"windows[{chain!r}] high")] for w in spans]
            if any(lo > hi for lo, hi in pairs):
                raise ValueError(
                    f"constraint.epitope.paratope.windows[{chain!r}] has a window with low > high.")
            paratope[str(chain)] = pairs
    else:
        raise ValueError(
            'constraint.epitope.paratope must be "cdr", "all" or {"windows": {chain: [[low, high], ...]}}.')
    return {"residues": items, "paratope": paratope, "min_fraction": min_fraction,
            "declared": len(residues)}


def validate_constraint(constraint: Any, use_tfg_guidance: bool) -> dict | None:
    """Check a `constraint` block against upstream's schema; returns it normalized or None.

    Normalized form: {"contact": [parsed restraint], "movable_chains": [ids] | None,
    "epitope": {...} | None}. Every malformed field raises ValueError with upstream's message.
    A contact list without TFG only logs upstream's hint; an epitope without TFG raises.
    """
    if constraint is None:
        return None
    if not isinstance(constraint, dict):
        raise ValueError("The 'constraint' field must be an object with the keys contact, "
                         "movable_chains and epitope.")
    if "epitope" in constraint:
        if constraint["epitope"] is None:
            raise ValueError("constraint.epitope is null; give an object or remove the key.")
        if "contact" in constraint:
            raise ValueError(
                "constraint.epitope and constraint.contact cannot be combined in one request.")
        if not use_tfg_guidance:
            raise ValueError("constraint.epitope is applied only through TFG guidance; "
                             "pass --use_tfg_guidance true or remove constraint.epitope.")
    unknown = sorted(set(constraint.keys()) - _SUPPORTED)
    if unknown:
        raise ValueError("Unsupported constraint field(s): %s. Supported: contact, "
                         "movable_chains, epitope." % ", ".join(unknown))
    contacts = constraint.get("contact")
    if contacts is not None and not isinstance(contacts, list):
        raise ValueError(f"constraint.contact must be a list; got {type(contacts).__name__}.")
    if contacts and not use_tfg_guidance:
        logger.warning("constraint.contact is present but TFG guidance is disabled. "
                       "Enable with --use_tfg_guidance true to apply epitope distance restraints.")
    movable = _parse_movable(constraint.get("movable_chains"))
    return {"contact": _parse_contacts(contacts or []), "movable_chains": movable,
            "epitope": _parse_epitope(constraint.get("epitope"), movable)}


def entity_groups(path) -> list[list[str]]:
    """Chain ids per upstream entity, in input order, read the way main._read_bio_chains reads.

    YAML: one list per `sequences` entry (protein/rna/dna/ligand), its `id` list in order.
    FASTA: one list per record; a comma-separated header id is one entity with several copies.
    """
    path = Path(path)
    groups: list[list[str]] = []
    if path.suffix.lower() in (".yml", ".yaml"):
        import yaml
        doc = yaml.safe_load(path.read_text()) or {}
        for entry in doc.get("sequences", []):
            if not isinstance(entry, dict):
                continue
            for key, default in (("protein", "A"), ("rna", "A"), ("dna", "A"), ("ligand", "L")):
                sub = entry.get(key)
                if not isinstance(sub, dict):
                    continue
                if key == "ligand" and not (sub.get("ccd") or sub.get("smiles")):
                    continue
                ids = sub.get("id", default)
                ids = [str(x) for x in ids] if isinstance(ids, (list, tuple)) else str(ids).split(",")
                groups.append([c.strip() for c in ids])
    else:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith(">"):
                groups.append([c.strip() for c in line[1:].split("|")[0].split(",")])
    return groups


class AtomTable:
    """Per-atom chain/entity/copy/residue/name, read back off tt-bio's feats.

    chains: main._read_bio_chains output, parallel to build_complex_features' input (asym_id
    is the chain's index in it). entities: chain ids per upstream entity (`entity_groups`);
    None makes every chain its own entity with copy 1.
    """

    def __init__(self, feats: dict, chains: list, entities: list[list[str]] | None = None):
        ids = [str(c[0]) for c in chains]
        if entities is None:
            entities = [[c] for c in ids]
        where = {}
        for e, group in enumerate(entities, start=1):
            for k, cid in enumerate(group, start=1):
                where[str(cid)] = (e, k)
        missing = [c for c in ids if c not in where]
        if missing:
            raise ValueError(f"chains {missing} belong to no entity in {entities}.")
        tok = feats["atom_to_token_idx"].long()
        asym = feats["asym_id"].long()[tok].numpy()
        self.chain_index = asym
        self.chain_id = np.array([ids[i] for i in asym], dtype=object)
        self.entity = np.array([where[ids[i]][0] for i in asym], dtype=np.int64)
        self.copy = np.array([where[ids[i]][1] for i in asym], dtype=np.int64)
        self.res_id = feats["residue_index"].long()[tok].numpy()
        self.atom_name = np.array(_atom_names(feats), dtype=object)
        self.heavy = feats["ref_element"].argmax(-1).numpy() != 0      # index 0 is hydrogen
        mt = feats["mol_type"].long()[tok].numpy()
        self.entity_mol_type = {}
        for e, group in enumerate(entities, start=1):
            sel = np.isin(self.chain_id, [str(c) for c in group])
            if sel.any():
                self.entity_mol_type[e] = int(mt[sel][0])
        self.chains = chains
        self.ids = ids

    def __len__(self):
        return len(self.atom_name)

    def residue_name(self, atoms: np.ndarray) -> str:
        """Upstream's res_name of the residue the atoms belong to (MSE reads as MET)."""
        ci = int(self.chain_index[atoms[0]])
        _cid, seq, _msa, mt, mods = self.chains[ci]
        pos = int(self.res_id[atoms[0]])
        if mt == "ligand":
            return seq[4:].split(",")[pos - 1] if seq.startswith("CCD_") else "UNL"
        for m in mods or []:
            if int(m["position"]) == pos:
                code = str(m["ccd"]).upper()
                return "MET" if code == "MSE" else code
        c = "".join(seq.split())[pos - 1].upper()
        if mt == "protein":
            return _AA1TO3.get(c, "UNK")
        if mt == "rna":
            return c if c in "ACGU" else "N"
        return "D" + c if c in "ACGT" else "DN"


def _contact_features(contacts: list, t: AtomTable) -> dict:
    pairs, lowers, uppers = [], [], []
    for ci, r in enumerate(contacts):
        side_idx = []
        for side_name in ("left", "right"):
            s = r[side_name]
            atom_name = s["atom_name"]
            if atom_name is None or str(atom_name).strip() == "":
                if t.entity_mol_type.get(s["entity_id"]) != MOL_TYPE_IDS["protein"]:
                    poly = {MOL_TYPE_IDS["rna"]: "polyribonucleotide",
                            MOL_TYPE_IDS["dna"]: "polydeoxyribonucleotide"}.get(
                        t.entity_mol_type.get(s["entity_id"]),
                        "non-polymer" if s["entity_id"] in t.entity_mol_type else "")
                    raise ValueError(
                        f"constraint.contact[{ci}] {side_name}_atom/atom is required "
                        f"for non-protein entity {s['entity_id']} (type={poly!r}).")
                atom_name = "CA"
            atom_name = str(atom_name)
            mask = ((t.entity == s["entity_id"]) & (t.res_id == s["position"])
                    & (t.atom_name == atom_name))
            if s["copy_id"] is not None:
                mask &= t.copy == s["copy_id"]
            idx = np.where(mask)[0]
            if idx.size == 0:
                raise ValueError(
                    f"No atom found for constraint.contact[{ci}] {side_name}: "
                    f"entity={s['entity_id']} position={s['position']} atom={atom_name!r} "
                    f"copy={s['copy_id']}.")
            side_idx.append(idx)
        if len(side_idx[0]) != len(side_idx[1]):
            raise ValueError(
                f"constraint.contact[{ci}]: asymmetric copy counts "
                f"({len(side_idx[0])} vs {len(side_idx[1])}); "
                "specify copy1/copy2 explicitly or keep entity counts equal.")
        for a, b in zip(side_idx[0], side_idx[1]):
            pairs.append([int(a), int(b)])
            lowers.append(float(r["min_distance"]))
            uppers.append(float(r["max_distance"]))
    index = (torch.as_tensor(pairs, dtype=torch.int64).T if pairs
             else torch.empty((2, 0), dtype=torch.int64))
    return {"user_distance_restraint_index": index,
            "user_distance_restraint_lower_bound": torch.as_tensor(lowers, dtype=torch.float32),
            "user_distance_restraint_upper_bound": torch.as_tensor(uppers, dtype=torch.float32)}


def _movable_features(wanted: list[str] | None, t: AtomTable) -> dict:
    if wanted is None:
        return {"user_rigid_movable_atom": torch.empty((0,), dtype=torch.bool)}
    present = set(np.unique(t.chain_id).tolist())
    missing = [c for c in wanted if c not in present]
    if missing:
        raise ValueError(f"constraint.movable_chains names chains that are absent: {missing}. "
                         f"Chains present: {sorted(present)}.")
    mask = np.isin(t.chain_id, wanted)
    if mask.all():
        raise ValueError("constraint.movable_chains names every chain; nothing would stay fixed.")
    return {"user_rigid_movable_atom": torch.as_tensor(mask, dtype=torch.bool)}


def _epitope_features(ep: dict | None, movable_mask: torch.Tensor, t: AtomTable) -> dict:
    if ep is None:
        return {}
    movable = movable_mask.numpy().astype(bool)
    seen: dict[tuple[int, int, int], list[int]] = {}
    for idx, it in enumerate(ep["residues"]):
        entity, copy_id, position = it["entity"], it["copy"], it["position"]
        atoms = np.where((t.entity == entity) & (t.copy == copy_id) & (t.res_id == position))[0]
        if atoms.size == 0:
            raise ValueError(f"constraint.epitope.residues[{idx}]: no atom found for entity={entity} "
                             f"copy={copy_id} position={position}.")
        name = t.residue_name(atoms)
        if name not in _AA3TO1:
            raise ValueError(
                f"constraint.epitope.residues[{idx}] is {name!r}, not a standard amino acid.")
        letter = _AA3TO1[name]
        expected = it["residue"]
        if expected is not None and letter != str(expected).upper():
            raise ValueError(
                f"constraint.epitope.residues[{idx}]: residue {expected!r} given, "
                f"the sequence has {letter!r} at entity={entity} copy={copy_id} position={position}.")
        if movable[atoms].any():
            raise ValueError(f"constraint.epitope.residues[{idx}] lies in a movable chain; "
                             "the epitope must stay fixed.")
        seen[(entity, copy_id, position)] = atoms.tolist()
    seen = dict(sorted(seen.items()))
    n = len(seen)
    k = max(1, math.ceil(Fraction(repr(ep["min_fraction"])) * n))
    moving_chains = sorted(set(t.chain_id[movable].tolist()))
    spec = ep["paratope"]
    if spec == "all":
        para = movable.copy()
    else:
        windows = {c: CDR_WINDOWS for c in moving_chains} if spec == "cdr" else spec
        para = np.zeros(len(t), dtype=bool)
        for chain, spans in windows.items():
            for lo, hi in spans:
                para |= movable & (t.chain_id == chain) & (t.res_id >= lo) & (t.res_id <= hi)
    if not para.any():
        raise ValueError("constraint.epitope: the paratope selects no atom of the movable chains.")
    width = max(len(v) for v in seen.values())
    index = np.full((n, width), -1, dtype=np.int64)
    for row, atoms in enumerate(seen.values()):
        index[row, :len(atoms)] = atoms
    para_res = {c: len({int(r) for r in t.res_id[para & (t.chain_id == c)]}) for c in moving_chains}
    logger.info("EPITOPE_INPUT declared=%d unique_resolved=%d K=%d min_fraction=%s "
                "paratope_atoms=%d paratope_residues_per_chain=%s movable_chains=%s",
                ep["declared"], n, k, ep["min_fraction"], int(para.sum()), para_res, moving_chains)
    return {"user_epitope_atom_index": torch.as_tensor(index, dtype=torch.int64),
            "user_epitope_paratope_atom": torch.as_tensor(para, dtype=torch.bool),
            "user_epitope_k": torch.as_tensor([k], dtype=torch.int64)}


def constraint_features(constraint: dict | None, feats: dict, chains: list,
                        entities: list[list[str]] | None = None) -> dict:
    """Resolve a validated constraint (validate_constraint output) to upstream's TFG tensors.

    Emits user_distance_restraint_index [2, M] int64, user_distance_restraint_lower_bound /
    _upper_bound [M] float32, user_rigid_movable_atom [N_atom] bool (empty when
    movable_chains is absent) and, with an epitope, user_epitope_atom_index [n, W] int64
    (-1 padded), user_epitope_paratope_atom [N_atom] bool, user_epitope_k [1] int64.
    """
    constraint = constraint or {"contact": [], "movable_chains": None, "epitope": None}
    t = AtomTable(feats, chains, entities)
    out = _contact_features(constraint["contact"], t)
    out.update(_movable_features(constraint["movable_chains"], t))
    out.update(_epitope_features(constraint["epitope"], out["user_rigid_movable_atom"], t))
    return out


# ---------------------------------------------------------------------------------------------
# Physics-guidance geometry features: opendde/data/core/geometry_featurizer.py (GeometryFeaturizer,
# get_ccd_geometry_features and the extract_*_from_mol helpers, copied with their semantics).

RDKIT_GEOMETRY_FEATURES = [
    "pairwise_distance_index", "pairwise_distance_upper_bound", "pairwise_distance_lower_bound",
    "pairwise_distance_is_bond", "pairwise_distance_is_angle",
    "experimental_torsion_index", "experimental_torsion_force_constant",
    "experimental_torsion_sign", "linear_triple_bond_index", "chiral_index",
    "chiral_orientation", "stereo_bond_index", "stereo_bond_orientation",
    "planar_improper_index", "planar_improper_is_carbonyl",
]
GEOMETRY_FEATURES = ["interchain_bond_index", "symmetric_chain_index"] + RDKIT_GEOMETRY_FEATURES
METAL_ATOMIC_NUMBERS = frozenset(list(range(3, 5)) + list(range(11, 14)) + list(range(19, 32))
                                 + list(range(37, 52)) + list(range(55, 85)) + list(range(87, 119)))
_INDEX_WIDTHS = {
    "interchain_bond_index": 2, "symmetric_chain_index": 2, "pairwise_distance_index": 2,
    "linear_triple_bond_index": 3, "experimental_torsion_index": 4, "chiral_index": 4,
    "stereo_bond_index": 4, "planar_improper_index": 4,
}
# upstream STD_RESIDUES: the 20 amino acids + UNK, RNA A/G/C/U/N, DNA DA/DG/DC/DT/DN
STD_RESIDUES = frozenset(list(_AA3TO1) + ["UNK", "A", "G", "C", "U", "N",
                                          "DA", "DG", "DC", "DT", "DN"])


def _angle_triples(bond_index: np.ndarray) -> np.ndarray:
    """upstream build_angle_triples_from_bonds."""
    from itertools import combinations
    if bond_index.size == 0:
        return np.empty((0, 3), dtype=int)
    neighbors = [[] for _ in range(bond_index.max() + 1)]
    for u, v in bond_index:
        if u != v:
            neighbors[u].append(v)
            neighbors[v].append(u)
    angles = [[i, j, k] for j, nb in enumerate(neighbors) if len(nb) >= 2
              for i, k in combinations(sorted(nb), 2)]
    return np.asarray(angles, dtype=int) if angles else np.empty((0, 3), dtype=int)


def _pairwise_distance(mol) -> dict:
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
    bm = GetMoleculeBoundsMatrix(mol)
    n = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    bond_index = np.asarray(bonds, dtype=int) if bonds else np.empty((0, 2), dtype=int)
    bond_pairs = {tuple(sorted((int(i), int(j)))) for i, j in bond_index.tolist()}
    tri = _angle_triples(bond_index)
    angle_pairs = {tuple(sorted((int(i), int(k)))) for i, _, k in tri.tolist()}
    out = {k: [] for k in RDKIT_GEOMETRY_FEATURES[:5]}
    for i in range(n - 1):
        for j in range(i + 1, n):
            out["pairwise_distance_index"].append([i, j])
            out["pairwise_distance_upper_bound"].append(float(bm[i, j]))
            out["pairwise_distance_lower_bound"].append(float(bm[j, i]))
            out["pairwise_distance_is_bond"].append(int((i, j) in bond_pairs))
            out["pairwise_distance_is_angle"].append(int((i, j) in angle_pairs))
    return out


def _experimental_torsion(mol) -> dict:
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import GetExperimentalTorsions
    index, force_constant, sign, marked = [], [], [], set()
    for t in GetExperimentalTorsions(mol, useSmallRingTorsions=True):
        ai = list(t["atomIndices"])
        index.append(ai)
        force_constant.append(list(t["V"]))
        sign.append(list(t["signs"]))
        marked.add(tuple(sorted((ai[1], ai[2]))))
    ring_info = mol.GetRingInfo()
    for ring in (ring_info.AtomRings() if ring_info is not None else ()):
        n = len(ring)
        if 3 < n < 7:
            for a in range(n):
                idx4 = [ring[a], ring[(a + 1) % n], ring[(a + 2) % n], ring[(a + 3) % n]]
                jk = tuple(sorted((idx4[1], idx4[2])))
                if jk in marked:
                    continue
                if all(mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP2
                       for i in idx4):
                    index.append(idx4)
                    force_constant.append([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
                    sign.append([1, -1, 1, 1, 1, 1])
                    marked.add(jk)
    return {"experimental_torsion_index": index,
            "experimental_torsion_force_constant": force_constant,
            "experimental_torsion_sign": sign}


def _chiral(mol) -> dict:
    from itertools import combinations
    from rdkit import Chem
    from rdkit.Chem.rdMolTransforms import GetDihedralRad
    assert mol.GetNumConformers() > 0, "mol does not have ref pos"
    conf = mol.GetConformer(0)
    index, orient = [], []
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() not in (Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                       Chem.ChiralType.CHI_TETRAHEDRAL_CW):
            continue
        nb = sorted(n.GetIdx() for n in atom.GetNeighbors())
        if len(nb) > 4 or len(nb) < 3:
            continue
        for idx in combinations(nb, 3):
            d = [*idx, atom.GetIdx()]
            index.append(d)
            orient.append(1.0 if GetDihedralRad(conf, *d) >= 0.0 else -1.0)
    return {"chiral_index": index, "chiral_orientation": orient}


def _linear_triple_bond(mol) -> dict:
    from rdkit import Chem
    index = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        b, e = bond.GetBeginAtom(), bond.GetEndAtom()
        bi, ei = b.GetIdx(), e.GetIdx()
        if bond.GetIsAromatic() or b.GetIsAromatic() or e.GetIsAromatic():
            continue
        if not (b.GetHybridization() == Chem.HybridizationType.SP
                and e.GetHybridization() == Chem.HybridizationType.SP):
            continue
        for n in sorted(x.GetIdx() for x in b.GetNeighbors() if x.GetIdx() != ei):
            index.append([n, bi, ei])
        for n in sorted(x.GetIdx() for x in e.GetNeighbors() if x.GetIdx() != bi):
            index.append([bi, ei, n])
    return {"linear_triple_bond_index": index}


def _stereo_bond(mol) -> dict:
    from rdkit.Chem.rdchem import BondStereo
    from rdkit.Chem.rdMolTransforms import GetDihedralRad
    assert mol.GetNumConformers() > 0, "mol does not have ref pos"
    conf = mol.GetConformer(0)

    def orientation(d):
        return 0.0 if np.abs(GetDihedralRad(conf, *d)) < np.pi / 2 else 1.0

    index, orient = [], []
    for bond in mol.GetBonds():
        if bond.GetStereo() not in (BondStereo.STEREOE, BondStereo.STEREOZ):
            continue
        bi, ei = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bn = sorted(x.GetIdx() for x in bond.GetBeginAtom().GetNeighbors() if x.GetIdx() != ei)
        en = sorted(x.GetIdx() for x in bond.GetEndAtom().GetNeighbors() if x.GetIdx() != bi)
        if not bn or not en:
            continue
        d = [bn[0], bi, ei, en[0]]
        index.append(d)
        orient.append(orientation(d))
        if len(bn) == 2 and len(en) == 2:
            d = [bn[1], bi, ei, en[1]]
            index.append(d)
            orient.append(orientation(d))
    return {"stereo_bond_index": index, "stereo_bond_orientation": orient}


def _planar_improper(mol) -> dict:
    from rdkit import Chem
    index, carbonyl = [], []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ("C", "N", "O"):
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP2:
            continue
        nbs = atom.GetNeighbors()
        if len(nbs) != 3:
            continue
        c = atom.GetIdx()
        n1, n2, n3 = sorted(nb.GetIdx() for nb in nbs)
        index += [[n1, n2, c, n3], [n3, n1, c, n2], [n2, n3, c, n1]]
        has = any(atom.GetSymbol() == "C" and nb.GetSymbol() == "O"
                  and nb.GetHybridization() == Chem.HybridizationType.SP2 for nb in nbs)
        carbonyl += [float(has)] * 3
    return {"planar_improper_index": index, "planar_improper_is_carbonyl": carbonyl}


def mol_geometry_features(mol, name: str = "?") -> dict:
    """upstream get_ccd_geometry_features on one rdkit Mol, in the Mol's own atom indices."""
    import copy
    from rdkit import Chem
    mol = copy.deepcopy(mol)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        Chem.GetSymmSSSR(mol)
        return {**_pairwise_distance(mol), **_experimental_torsion(mol), **_chiral(mol),
                **_linear_triple_bond(mol), **_stereo_bond(mol), **_planar_improper(mol)}
    except Exception:
        logger.warning("mol %s: geometry features failed, returning empty features", name)
        return {k: [] for k in RDKIT_GEOMETRY_FEATURES}


def _dict_to_tensor(d: dict) -> dict:
    """upstream GeometryFeaturizer.dict_to_tensor: *_index -> int64 (K, N), floats -> float32."""
    for k, v in d.items():
        if k.endswith("_index"):
            v = torch.as_tensor(v, dtype=torch.int64)
            if v.ndim == 2:
                v = v.T
            if v.numel() == 0 and k in _INDEX_WIDTHS:
                v = v.new_empty((_INDEX_WIDTHS[k], 0))
        else:
            v = torch.as_tensor(v)
            if v.is_floating_point():
                v = v.to(torch.float32)
            elif not v.dtype == torch.bool:
                v = v.to(torch.int64)
        d[k] = v
    return d


# Where CCD component Mols come from, in order. OpenDDE's own cache is the exact upstream
# input; the other two are fallbacks whose Kekulé form can differ (see _fallback_mol).
OPENDDE_CCD_MOL_PKL = "~/.cache/opendde/common/components.cif.rdkit_mol.pkl"
BOLTZ_CCD_PKL = "~/.boltz/ccd.pkl"
_PKL_CACHE: dict = {}


def _load_pkl(path: str) -> dict:
    """{code: Mol} from a pickle, read once per process; {} when the file is absent."""
    import os
    import pickle
    from rdkit import Chem
    path = os.path.expanduser(os.environ.get("TT_BIO_OPENDDE_CCD_MOL_PKL", path)
                              if path == OPENDDE_CCD_MOL_PKL else path)
    if path not in _PKL_CACHE:
        _PKL_CACHE[path] = {}
        if os.path.exists(path):
            Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AtomProps)
            with open(path, "rb") as f:
                _PKL_CACHE[path] = pickle.load(f)  # noqa: S301
    return _PKL_CACHE[path]


def _kekulized(mol):
    """RDKit's own Kekulé bond orders with the aromatic flags kept."""
    from rdkit import Chem
    mol = Chem.Mol(mol)
    for b in mol.GetBonds():
        if b.GetIsAromatic():
            b.SetBondType(Chem.BondType.AROMATIC)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception:
        logger.warning("could not kekulize a CCD Mol; keeping its aromatic bond orders")
    return mol


def _fallback_mol(code: str):
    """A CCD Mol when OpenDDE's cache is absent: Boltz's ccd.pkl as is, else the tt-bio `mols`
    pickle with hydrogens added and RDKit's Kekulé form.

    Upstream's Mols carry explicit hydrogens and Kekulé bond orders with aromatic flags set;
    RDKit's bounds matrix and ring torsions read both. ccd.pkl has the same atoms and bond
    order but sometimes another Kekulé form (adenine), and the `mols` pickles are heavy-atom
    only with AROMATIC bond types, so either can shift bounds and ring torsions of fused
    heteroaromatics (ccd.pkl measured against upstream on 30 common components: ATP, ADP,
    SAH, FAD and NAD differ, the other 25 match exactly).
    """
    mol = _load_pkl(BOLTZ_CCD_PKL).get(code)
    if mol is not None:
        return mol
    from rdkit import Chem
    from ..data.mol import load_molecules
    from ..protenix_data import _default_mol_dir
    try:
        mol = load_molecules(_default_mol_dir(), [code])[code]
    except (FileNotFoundError, ValueError):
        return None
    if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    return _kekulized(mol)


def ccd_mol(code: str):
    """The CCD component Mol upstream featurizes `code` from, or the closest fallback; None
    when no source has it (then the residue gets no geometry features, as upstream when
    get_ccd_ref_info returns empty)."""
    mol = _load_pkl(OPENDDE_CCD_MOL_PKL).get(code)
    if mol is None:
        mol = _fallback_mol(code)
        if mol is None:
            logger.warning("CCD component %s found in no CCD source; no geometry features "
                           "for it", code)
    return mol


def _mol_atom_names(mol) -> dict:
    """{rdkit atom idx: atom name}: OpenDDE's Mol.atom_map, else the `name` atom prop."""
    amap = getattr(mol, "atom_map", None)
    if amap:
        return {int(i): str(n) for n, i in amap.items()}
    return {a.GetIdx(): a.GetProp("name") for a in mol.GetAtoms() if a.HasProp("name")}


def _residue_mol(t: AtomTable, atoms: np.ndarray, mols: dict):
    """(res_name, rdkit Mol, {mol atom idx: global atom idx}) for one residue, or None to skip.

    A polymer residue or CCD ligand reads `mols[code]` when given, else `ccd_mol(code)`. A
    SMILES ligand reads the Mol protenix_data embeds for it, atoms named the way
    ligand_atom_features names them.
    """
    from ..protenix_data import _smiles_to_mol
    ci = int(t.chain_index[atoms[0]])
    cid, seq, _msa, mt, _mods = t.chains[ci]
    name = t.residue_name(atoms)
    if mt == "ligand" and not seq.startswith("CCD_"):
        mol = _smiles_to_mol(seq)
        names, j = {}, 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() > 1:
                names[a.GetIdx()] = (a.GetProp("name") if a.HasProp("name")
                                     else a.GetSymbol().upper() + str(j + 1))
                j += 1
    else:
        if mt == "ligand" and "," in seq:
            raise NotImplementedError(
                f"chain {cid}: multi-component CCD ligand {seq!r}; tt-bio gives all its atoms "
                "residue_index 1, so upstream's per-component residues are not recoverable.")
        if name not in mols:
            mols[name] = ccd_mol(name)
        mol = mols[name]
        if mol is None:
            return None
        names = _mol_atom_names(mol)
    if mol.GetNumAtoms() == 0:
        return None
    by_name = {nm: idx for idx, nm in names.items()}
    local = {}
    for g in atoms:
        idx = by_name.get(t.atom_name[g])
        if idx is not None:
            local[idx] = int(g)
    return name, mol, local


def geometry_features(feats: dict, chains: list, entities: list[list[str]] | None = None,
                      bonds: list | None = None, mols: dict | None = None,
                      exclude_std_residue: bool = True) -> dict:
    """upstream GeometryFeaturizer(atom_array, exclude_std_residue=True).get_features().

    Per residue (ligand chain = one residue, a `modifications:` residue = its CCD code) whose
    name is not a standard residue on a polymer chain, the CCD component's RDKit restraints
    are mapped by atom name onto tt-bio's atom axis; restraints touching an atom tt-bio does
    not carry (hydrogens, leaving atoms) are dropped, as upstream drops atoms absent from its
    AtomArray. Residues containing a metal are skipped. interchain_bond_index lists the user
    covalent `bonds` ((chain, res, atom) pairs, as build_complex_features takes them) that
    join two chains; symmetric_chain_index the asym_id pairs of one upstream entity.
    """
    t = AtomTable(feats, chains, entities)
    mols = dict(mols or {})
    out = {k: [] for k in RDKIT_GEOMETRY_FEATURES}
    key = t.chain_index.astype(np.int64) * (1 << 32) + t.res_id
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stops = np.r_[starts[1:], len(t)]
    cache: dict = {}
    for start, stop in zip(starts, stops):
        atoms = np.arange(start, stop)
        ci = int(t.chain_index[start])
        mt = t.chains[ci][3]
        name = t.residue_name(atoms)
        if exclude_std_residue and name in STD_RESIDUES and mt != "ligand":
            continue
        got = _residue_mol(t, atoms, mols)
        if got is None:
            continue
        name, mol, local = got
        if any(a.GetAtomicNum() in METAL_ATOMIC_NUMBERS for a in mol.GetAtoms()):
            continue
        ck = name if t.chains[ci][1].startswith("CCD_") or mt != "ligand" else t.chains[ci][1]
        if ck not in cache:
            cache[ck] = mol_geometry_features(mol, name)
        feat = cache[ck]
        for fname in ("pairwise_distance", "experimental_torsion", "linear_triple_bond",
                      "chiral", "stereo_bond", "planar_improper"):
            ik = f"{fname}_index"
            keep = []
            for fi, ai in enumerate(feat[ik]):
                if all(i in local for i in ai):
                    out[ik].append([local[i] for i in ai])
                    keep.append(fi)
            for vk in RDKIT_GEOMETRY_FEATURES:
                if fname in vk and vk != ik:
                    out[vk].extend(feat[vk][i] for i in keep)
    inter = []
    for (c1, r1, a1), (c2, r2, a2) in bonds or []:
        if str(c1) == str(c2):
            continue
        pair = []
        for c, r, a in ((c1, r1, a1), (c2, r2, a2)):
            idx = np.flatnonzero((t.chain_id == str(c)) & (t.res_id == int(r))
                                 & (t.atom_name == str(a)))
            if idx.size != 1:
                raise ValueError(f"bond endpoint ({c}, {r}, {a}) resolves to {idx.size} atoms.")
            pair.append(int(idx[0]))
        inter.append(sorted(pair))
    chain_entity = {}
    for ci, e in zip(t.chain_index.tolist(), t.entity.tolist()):
        chain_entity.setdefault(ci, e)
    asym = sorted(chain_entity)
    sym = [[a, b] for a in asym for b in asym if b > a and chain_entity[a] == chain_entity[b]]
    out["interchain_bond_index"] = inter
    out["symmetric_chain_index"] = sym
    return _dict_to_tensor(out)
