# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

"""Minimal YAML -> Nesso ``Structure`` + ``Record``

Supported ``sequences``:
  - ``protein``: ``id`` (str or list), ``sequence``, optional ``esm`` (path to
    precomputed ``.safetensors`` ESM embedding for this sequence).
  - ``ligand``: ``id``, and one of ``smiles``, ``ccd``, or ``sdf``. An optional
    ``conformer`` key may point to a pickle of an RDKit ``Mol`` with at least one
    conformer to skip on-the-fly ETKDG embedding.

Top-level optional ``properties`` block can declare
``affinity.binder: <chain_id>`` to request affinity prediction at inference time.

Returns ``(Structure, Record, entity_to_seq, entity_to_esm_path)``:
  - ``entity_to_seq``: protein ``entity_id -> sequence`` (for ESM / MD5 ``msa_id``).
  - ``entity_to_esm_path``: protein ``entity_id -> esm path`` when provided in YAML.
  Ligand chains use ``msa_id=-1`` on the record.
"""

import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

from tt_bio._vendor.nesso.data import const
from tt_bio._vendor.nesso.data.types import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    Coords,
    Ensemble,
    Interface,
    Record,
    Residue,
    Structure,
)


@dataclass
class _ChainData:
    mol_type: int
    entity_id: int
    residues: list[dict[str, Any]]


@dataclass(frozen=True)
class _EntityProtein:
    sequence: str
    esm: str | None = None
    pocket_mask: str | None = None


@dataclass(frozen=True)
class _EntityLigandSmiles:
    smiles: str
    conformer: str | None = None


@dataclass(frozen=True)
class _EntityLigandCCD:
    ccd: str


@dataclass(frozen=True)
class _EntityLigandSDF:
    sdf: str


_Entity = _EntityProtein | _EntityLigandSmiles | _EntityLigandCCD | _EntityLigandSDF


def load_ccd_mol_dict(path: Path) -> dict[str, Chem.Mol]:
    """Load ``dict[CCD_id, RDKit Mol]`` from a single pickle (e.g. ``ccd.pkl``)."""
    with path.open("rb") as f:
        obj = pickle.load(f)  # noqa: S301
    if not isinstance(obj, dict):
        msg = f"Expected dict[str, Mol] in {path}, got {type(obj).__name__}"
        raise TypeError(msg)
    return obj  # type: ignore[return-value]


def _load_mol(
    mol_dir: Path | None,
    code: str,
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
) -> Chem.Mol:
    if ccd_dict is not None and code in ccd_dict:
        return Chem.Mol(ccd_dict[code])
    if mol_dir is None:
        msg = f"No RDKit Mol for {code!r}: not in ccd_pkl and mol_dir not set"
        raise FileNotFoundError(msg)
    path = mol_dir / f"{code}.pkl"
    if not path.exists():
        msg = f"Missing component pickle: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def get_conformer(mol: Chem.Mol) -> Chem.Conformer:
    if mol.GetNumConformers() == 0:
        opts = AllChem.ETKDGv3()
        opts.clearConfs = False
        cid = AllChem.EmbedMolecule(mol, opts)
        if cid < 0:
            opts.useRandomCoords = True
            AllChem.EmbedMolecule(mol, opts)
    return mol.GetConformer(0)


def _standard_residue(
    res_name: str,
    mol_dir: Path | None,
    res_idx: int,
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
) -> dict[str, Any]:
    name = "MET" if res_name == "MSE" else res_name
    if name not in const.ref_atoms or not const.ref_atoms[name]:
        msg = f"Unsupported or unknown residue token {name!r}"
        raise ValueError(msg)
    mol = _load_mol(mol_dir, name, ccd_dict=ccd_dict)
    mol = Chem.RemoveHs(mol, sanitize=False)
    conf = get_conformer(mol)
    by_name = {a.GetProp("name"): a for a in mol.GetAtoms()}
    atoms: list[dict[str, Any]] = []
    for atom_name in const.ref_atoms[name]:
        a = by_name[atom_name]
        p = conf.GetAtomPosition(a.GetIdx())
        atoms.append(
            {
                "name": atom_name[:4],
                "element": int(a.GetAtomicNum()),
                "charge": float(a.GetFormalCharge()),
                "coord": (float(p.x), float(p.y), float(p.z)),
            }
        )
    return {
        "name": name[:5],
        "type": int(const.token_ids[name]),
        "idx": res_idx,
        "is_standard": True,
        "atoms": atoms,
        "bonds": [],
        "atom_center": int(const.res_to_center_atom_id[name]),
        "atom_disto": int(const.res_to_disto_atom_id[name]),
    }


def _protein_residues(
    raw_seq: str,
    mol_dir: Path | None,
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    cmap = const.prot_letter_to_token
    mol_type = const.chain_type_ids["PROTEIN"]
    # Upstream falls back to UNK for anything not in the map, which silently turns a
    # typo into a real prediction on a different sequence. The map already covers all
    # 26 letters plus '-', and X is a legitimate unknown residue, so a miss here is
    # always an input error -- lowercase, a digit, whitespace, a gap character.
    bad = sorted({c for c in raw_seq if c not in cmap})
    if bad:
        first = next(i for i, c in enumerate(raw_seq) if c in bad)
        raise ValueError(
            f"protein sequence has {len(bad)} unrecognized residue code(s) "
            f"{bad!r}, first at position {first}. Use the 20 standard one-letter "
            f"codes, X for an unknown residue, or '-' for a gap."
        )
    residues = [
        _standard_residue(cmap[c], mol_dir, j, ccd_dict=ccd_dict)
        for j, c in enumerate(raw_seq)
    ]
    return mol_type, residues


def _ligand_residue_from_mol(
    mol: Chem.Mol, res_name: str, res_idx: int
) -> dict[str, Any]:
    conf = get_conformer(mol)
    idx_map: dict[int, int] = {}
    atoms: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        idx_map[atom.GetIdx()] = len(atoms)
        nm = atom.GetProp("name") if atom.HasProp("name") else atom.GetSymbol().upper()
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(
            {
                "name": nm[:4],
                "element": int(atom.GetAtomicNum()),
                "charge": float(atom.GetFormalCharge()),
                "coord": (float(p.x), float(p.y), float(p.z)),
            }
        )
    unk_b = const.bond_type_ids[const.unk_bond_type]
    bonds: list[tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in idx_map or j not in idx_map:
            continue
        tname = bond.GetBondType().name
        bt = const.bond_type_ids.get(tname, unk_b)
        bonds.append((idx_map[i], idx_map[j], int(bt)))
    unk_t = int(const.unk_token_ids["PROTEIN"])
    ad = min(1, len(atoms) - 1) if len(atoms) > 1 else 0
    return {
        "name": res_name[:5],
        "type": unk_t,
        "idx": res_idx,
        "is_standard": False,
        "atoms": atoms,
        "bonds": bonds,
        "atom_center": 0,
        "atom_disto": ad,
    }


def _ligand_from_conformer_pkl(
    path: str | Path,
    lig_tag: str,
    mol_dir: Path | None = None,
    rid: str = "",
    record_id: str = "",
) -> tuple[int, list[dict[str, Any]]]:
    """Load an RDKit ``Mol`` (heavy-atom, with conformer) from ``path`` as a ligand residue."""
    with Path(path).open("rb") as f:
        mol = pickle.load(f)  # noqa: S301
    if mol.GetNumConformers() > 0 and mol.GetConformers()[0].GetId() != 0:
        conf = Chem.Conformer(mol.GetConformers()[0])
        conf.SetId(0)
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=False)
    for atom, rnk in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol)):
        sym = atom.GetSymbol().upper()
        atom.SetProp("name", f"{sym}{int(rnk) + 1}"[:4])
    _dump_ligand_mol(mol, mol_dir, rid, record_id=record_id)
    return const.chain_type_ids["NONPOLYMER"], [
        _ligand_residue_from_mol(mol, lig_tag, 0)
    ]


def _assign_ligand_atom_names(mol: Chem.Mol) -> None:
    for atom, rnk in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol)):
        if atom.HasProp("name"):
            continue
        sym = atom.GetSymbol().upper()
        atom.SetProp("name", f"{sym}{int(rnk) + 1}"[:4])


def _dump_ligand_mol(
    mol: Chem.Mol, mol_dir: Path | None, rid: str, *, record_id: str = ""
) -> None:
    if mol_dir is not None and rid:
        stem = f"{record_id}__{rid}" if record_id else rid
        with (mol_dir / f"{stem}.pkl").open("wb") as f:
            pickle.dump(mol, f)


def _ligand_smiles_residue(
    smi: str,
    mol_dir: Path | None = None,
    rid: str = "",
    *,
    record_id: str = "",
) -> tuple[int, list[dict[str, Any]]]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        msg = f"Invalid SMILES: {smi!r}"
        raise ValueError(msg)
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    get_conformer(mol)
    mol_nh = Chem.RemoveHs(mol, sanitize=False)
    _assign_ligand_atom_names(mol_nh)
    _dump_ligand_mol(mol_nh, mol_dir, rid, record_id=record_id)
    return const.chain_type_ids["NONPOLYMER"], [
        _ligand_residue_from_mol(mol_nh, rid or "LIG", 0)
    ]


def _ligand_ccd_residue(
    ccd_code: str,
    mol_dir: Path | None = None,
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
    rid: str = "",
    record_id: str = "",
) -> tuple[int, list[dict[str, Any]]]:
    mol = _load_mol(mol_dir, ccd_code, ccd_dict=ccd_dict)
    mol = Chem.RemoveHs(mol, sanitize=False)
    _assign_ligand_atom_names(mol)
    get_conformer(mol)
    _dump_ligand_mol(mol, mol_dir, rid, record_id=record_id)
    return const.chain_type_ids["NONPOLYMER"], [
        _ligand_residue_from_mol(mol, ccd_code, 0)
    ]


def _ligand_sdf_residue(
    sdf_path: str,
    mol_dir: Path | None = None,
    rid: str = "",
    *,
    record_id: str = "",
) -> tuple[int, list[dict[str, Any]]]:
    path = Path(sdf_path)
    if not path.exists():
        msg = f"SDF file not found: {path}"
        raise FileNotFoundError(msg)
    mol = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    if mol is None:
        msg = f"Invalid SDF: {path}"
        raise ValueError(msg)
    _assign_ligand_atom_names(mol)
    get_conformer(mol)
    _dump_ligand_mol(mol, mol_dir, rid, record_id=record_id)
    return const.chain_type_ids["NONPOLYMER"], [
        _ligand_residue_from_mol(mol, rid or "LIG", 0)
    ]


def _parse_entity(kind: str, block: dict[str, Any]) -> _Entity:
    if kind == "protein":
        return _EntityProtein(
            str(block["sequence"]), block.get("esm"), block.get("pocket_mask")
        )
    ligand_keys = [k for k in ("smiles", "ccd", "sdf") if k in block]
    if len(ligand_keys) != 1:
        raise ValueError("ligand needs exactly one of 'smiles', 'ccd', or 'sdf'")
    key = ligand_keys[0]
    if key == "ccd":
        return _EntityLigandCCD(str(block["ccd"]))
    if key == "sdf":
        return _EntityLigandSDF(str(block["sdf"]))
    return _EntityLigandSmiles(str(block["smiles"]), block.get("conformer"))


def _chain_data_for_entity(
    entity: _Entity,
    mol_dir: Path | None,
    smiles_ligand_idx: list[int],
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
    rid: str = "",
    record_id: str = "",
) -> _ChainData:
    if isinstance(entity, _EntityProtein):
        mol_type, residues = _protein_residues(
            entity.sequence, mol_dir, ccd_dict=ccd_dict
        )
    elif isinstance(entity, _EntityLigandSmiles):
        smiles_ligand_idx[0] += 1
        if entity.conformer:
            mol_type, residues = _ligand_from_conformer_pkl(
                entity.conformer,
                rid or "LIG",
                mol_dir=mol_dir,
                rid=rid,
                record_id=record_id,
            )
        else:
            mol_type, residues = _ligand_smiles_residue(
                entity.smiles, mol_dir=mol_dir, rid=rid, record_id=record_id
            )
    elif isinstance(entity, _EntityLigandCCD):
        mol_type, residues = _ligand_ccd_residue(
            entity.ccd,
            mol_dir=mol_dir,
            ccd_dict=ccd_dict,
            rid=rid,
            record_id=record_id,
        )
    elif isinstance(entity, _EntityLigandSDF):
        mol_type, residues = _ligand_sdf_residue(
            entity.sdf, mol_dir=mol_dir, rid=rid, record_id=record_id
        )
    else:
        raise TypeError(entity)
    return _ChainData(mol_type=mol_type, entity_id=-1, residues=residues)


def _structure_from_chains(
    chains_map: dict[str, _ChainData], chain_order: list[str]
) -> Structure:
    atom_rows: list[tuple] = []
    bond_rows: list[tuple] = []
    res_rows: list[tuple] = []
    coord_rows: list[tuple] = []
    chain_rows: list[tuple] = []
    atom_idx = 0
    res_g = 0
    sym_count: dict[int, int] = defaultdict(int)

    for asym_id, cname in enumerate(chain_order):
        ch = chains_map[cname]
        eid = ch.entity_id
        sid = sym_count[eid]
        sym_count[eid] += 1
        a0 = atom_idx
        r0 = res_g
        atom_num = sum(len(r["atoms"]) for r in ch.residues)
        res_num = len(ch.residues)
        chain_rows.append(
            (
                str(cname)[:5],
                ch.mol_type,
                eid,
                sid,
                asym_id,
                a0,
                atom_num,
                r0,
                res_num,
            )
        )
        for res in ch.residues:
            ra = atom_idx
            n_at = len(res["atoms"])
            ac = atom_idx + int(res["atom_center"])
            ad = atom_idx + int(res["atom_disto"])
            res_rows.append(
                (
                    res["name"],
                    int(res["type"]),
                    int(res["idx"]),
                    atom_idx,
                    n_at,
                    ac,
                    ad,
                    bool(res["is_standard"]),
                    True,
                )
            )
            for at in res["atoms"]:
                atom_rows.append(
                    (
                        str(at["name"])[:4],
                        (0.0, 0.0, 0.0),
                        True,
                        int(at["element"]),
                        float(at["charge"]),
                    )
                )
                coord_rows.append((at["coord"],))
                atom_idx += 1
            for b1, b2, bt in res["bonds"]:
                bond_rows.append(
                    (asym_id, asym_id, res_g, res_g, ra + b1, ra + b2, int(bt))
                )
            res_g += 1

    atoms = np.array(atom_rows, dtype=Atom)
    bonds = np.array(bond_rows, dtype=Bond) if bond_rows else np.array([], dtype=Bond)
    residues = np.array(res_rows, dtype=Residue)
    chains_arr = np.array(chain_rows, dtype=Chain)
    interfaces = np.array([], dtype=Interface)
    mask = np.ones(len(chain_rows), dtype=bool)
    coords = np.array(coord_rows, dtype=Coords)
    ensemble = np.array([(0, len(coord_rows))], dtype=Ensemble)
    return Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains_arr,
        interfaces=interfaces,
        mask=mask,
        coords=coords,
        ensemble=ensemble,
    )


def _record_from_structure(
    struct: Structure,
    entity_to_seq: dict[int, str],
    record_id: str,
    *,
    binder_asym_id: int | None = None,
) -> Record:
    prot = const.chain_type_ids["PROTEIN"]
    chains_rx: list[ChainInfo] = []
    for i in range(len(struct.chains)):
        if not struct.mask[i]:
            continue
        c = struct.chains[i]
        eid = int(c["entity_id"])
        mt = int(c["mol_type"])
        if mt == prot and eid in entity_to_seq:
            mid: str | int = hashlib.md5(entity_to_seq[eid].encode("utf-8")).hexdigest()
        else:
            mid = -1
        chains_rx.append(
            ChainInfo(
                chain_name=str(c["name"]).strip(),
                msa_id=mid,
            )
        )
    return Record(
        id=record_id,
        chains=chains_rx,
        binder_asym_id=binder_asym_id,
    )


def _validate_affinity_binders(
    schema: dict[str, Any], chain_is_ligand: dict[str, bool]
) -> None:
    """Validate if properties.affinity.binder references a ligand chain.
    Raises ValueError when a binder names a chain id that is absent from the schema, or that exists but is not a ligand.
    """
    for prop in schema.get("properties") or []:
        if not isinstance(prop, dict):
            continue
        affinity = prop.get("affinity")
        if not isinstance(affinity, dict) or "binder" not in affinity:
            continue

        binder = affinity["binder"]
        if not isinstance(binder, str):
            raise ValueError(
                "affinity.binder must be a single chain id (str), "
                f"got {type(binder).__name__}"
            )
        if binder not in chain_is_ligand:
            available = ", ".join(sorted(chain_is_ligand)) or "<none>"
            raise ValueError(
                f"affinity.binder {binder!r} does not match any chain id "
                f"(available chains: {available})"
            )
        if not chain_is_ligand[binder]:
            raise ValueError(
                f"affinity.binder {binder!r} refers to a protein chain; "
                "the binder must be a ligand chain"
            )


def _resolve_binder_asym_id(
    schema: dict[str, Any], chain_order: list[str]
) -> int | None:
    """Return asym_id of the declared binder, or None if unspecified."""
    chain_to_asym = {cid: idx for idx, cid in enumerate(chain_order)}
    for prop in schema.get("properties") or []:
        if not isinstance(prop, dict):
            continue
        affinity = prop.get("affinity")
        if not isinstance(affinity, dict) or "binder" not in affinity:
            continue
        binder = affinity["binder"]
        if isinstance(binder, str) and binder in chain_to_asym:
            return chain_to_asym[binder]
    return None


def _chain_is_ligand_from_schema(schema: dict[str, Any]) -> dict[str, bool]:
    """Map chain_id to bool indicating if it is a ligand from schema dict."""
    out: dict[str, bool] = {}
    for item in schema.get("sequences") or []:
        if not isinstance(item, dict) or not item:
            continue
        kind = next(iter(item)).lower()
        if kind not in ("protein", "ligand"):
            continue
        block = item[kind]
        if not isinstance(block, dict):
            continue
        raw_id = block.get("id")
        if isinstance(raw_id, str):
            ids = [raw_id]
        elif isinstance(raw_id, (list, tuple)):
            ids = [c for c in raw_id if isinstance(c, str)]
        else:
            continue
        for cid in ids:
            out[cid] = kind == "ligand"
    return out


def validate_schema(schema: dict[str, Any]) -> None:
    """Schema validation before preprocessing.

    Enforces that ``properties.affinity.binder`` names an existing
    ligand chain (not a protein, and not a missing id).
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a mapping")
    _validate_affinity_binders(schema, _chain_is_ligand_from_schema(schema))


def parse_schema(
    schema: dict[str, Any],
    mol_dir: Path | None = None,
    *,
    ccd_dict: dict[str, Chem.Mol] | None = None,
    record_id: str = "record",
) -> tuple[Structure, Record, dict[int, str], dict[int, str]]:
    """Parse a YAML schema dict into
    ``(Structure, Record, entity_to_seq, entity_to_esm_path)``
    """
    if not isinstance(schema, dict) or schema.get("version", 1) != 1:
        raise ValueError("Schema must be a dict with version: 1")
    if "sequences" not in schema:
        raise ValueError("Schema must contain sequences:")

    groups: dict[_Entity, list[dict[str, Any]]] = defaultdict(list)
    for item in schema["sequences"]:
        kind = next(iter(item.keys())).lower()
        if kind not in ("protein", "ligand"):
            raise ValueError(f"Unsupported entity type {kind!r} (only protein, ligand)")
        block = item[kind]
        groups[_parse_entity(kind, block)].append(block)

    chains_map: dict[str, _ChainData] = {}
    chain_order: list[str] = []
    entity_to_seq: dict[int, str] = {}
    entity_to_esm: dict[int, str] = {}
    smiles_ligand_idx = [1]

    for entity_id, (entity, blocks) in enumerate(groups.items()):
        # For SMILES/SDF ligands use the first chain id as pkl name and residue tag
        # so multiple ligands don't collide on the same "LIG" name.
        if not isinstance(entity, _EntityProtein):
            raw_first = blocks[0]["id"]
            lid = raw_first if isinstance(raw_first, str) else list(raw_first)[0]
        else:
            lid = record_id
        data = _chain_data_for_entity(
            entity,
            mol_dir,
            smiles_ligand_idx,
            ccd_dict=ccd_dict,
            rid=lid,
            record_id=record_id,
        )
        data.entity_id = entity_id
        if isinstance(entity, _EntityProtein):
            entity_to_seq[entity_id] = entity.sequence
            if entity.esm:
                entity_to_esm[entity_id] = entity.esm
        for block in blocks:
            raw_id = block["id"]
            for chain_id in [raw_id] if isinstance(raw_id, str) else list(raw_id):
                if chain_id not in chains_map:
                    chain_order.append(chain_id)
                chains_map[chain_id] = data

    if not chains_map:
        raise ValueError("No chains parsed from schema")

    ligand_type = const.chain_type_ids["NONPOLYMER"]
    chain_is_ligand = {
        cid: cd.mol_type == ligand_type for cid, cd in chains_map.items()
    }
    _validate_affinity_binders(schema, chain_is_ligand)

    struct = _structure_from_chains(chains_map, chain_order)

    binder_asym_id = _resolve_binder_asym_id(schema, chain_order)
    rec = _record_from_structure(
        struct,
        entity_to_seq,
        record_id,
        binder_asym_id=binder_asym_id,
    )
    return struct, rec, entity_to_seq, entity_to_esm


def parse_yaml(
    path: Path,
    mol_dir: Path | None = None,
    *,
    ccd_pkl: Path | None = None,
    ccd_dict: dict[str, Chem.Mol] | None = None,
    record_id: str | None = None,
) -> tuple[Structure, Record, dict[int, str], dict[int, str]]:
    """Parse YAML into ``(Structure, Record, entity_to_seq, entity_to_esm_path)``.

    ``ccd_dict`` (in-memory) takes precedence over ``ccd_pkl`` (disk); ``_load_mol``
    raises if a standard residue needs a CCD source that was not provided.
    """
    if ccd_dict is None and ccd_pkl is not None:
        ccd_dict = load_ccd_mol_dict(ccd_pkl)

    with path.open("r") as f:
        schema = yaml.safe_load(f)

    rid = record_id if record_id is not None else path.stem
    return parse_schema(schema, mol_dir, ccd_dict=ccd_dict, record_id=rid)


def esm_keys(entity_to_seq: dict[int, str]) -> dict[str, str]:
    """``{md5(sequence): sequence}`` for protein entities only."""
    return {
        hashlib.md5(s.encode("utf-8")).hexdigest(): s for s in entity_to_seq.values()
    }
