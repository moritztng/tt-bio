# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.random import RandomState
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot

from tt_bio._vendor.nesso.data import const
# tt-bio patch: upstream imports this from nesso.model.modules.utils, whose body is
# identical to ours down to the order of global-RNG draws, so we reuse rather than
# carry a second copy. NOTE: it applies a RANDOM roto-translation to every
# conformer's ref_pos at inference time -- see scripts/nesso1_port/parity_gate.py.
from tt_bio.boltz2 import center_random_augmentation
from tt_bio._vendor.nesso.data.io import load_safetensor
from tt_bio._vendor.nesso.data.pad import pad_dim
from tt_bio._vendor.nesso.data.types import Record, Tokenized


# Nesso relies on RDKit atom properties (notably `atom.GetProp("name")`) that are
# attached to CCD-derived residue molecules. PyTorch `DataLoader` multiprocessing
# (spawn on macOS/Windows) pickles the Dataset, and RDKit drops atom properties by
# default unless explicitly enabled.
if hasattr(Chem, "SetDefaultPickleProperties") and hasattr(
    Chem, "PropertyPickleOptions"
):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AtomProps)


####################################################################################################
# FEATURES
####################################################################################################


def convert_atom_name(name) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str or np.ndarray
        The atom name (string for Atom).

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def extract_esm_features(
    esm_emb_path: Path,
    use_final_layer_only: bool = False,
):
    # Load embeddings, [num_layers, num_residues + 2, esm_embed_dim]
    s_chain = load_safetensor(esm_emb_path)

    # If using final layer only, then keep the final layer only
    if use_final_layer_only:
        s_chain = s_chain[-1, 1:-1]
    else:
        # Permute dimensions to [num_residues, num_layers, esm_embed_dim]
        s_chain = s_chain[:, 1:-1]
        s_chain = s_chain.permute(1, 0, 2)
    return s_chain


def select_subset_from_mask(mask: np.ndarray, p: float) -> np.ndarray:
    """Subsample True entries in mask using a geometric draw."""
    num_true = int(np.sum(mask))
    v = int(np.random.geometric(p)) + 1
    k = min(v, num_true)
    true_indices = np.where(mask)[0]
    selected_indices = np.random.choice(true_indices, size=k, replace=False)
    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1
    return new_mask


def _token_atom_coords(data: Tokenized, token: dict) -> np.ndarray:
    """Heavy-atom coordinates for a token (aligned with process_atom_features)."""
    offset = int(data.structure.ensemble[0]["atom_coord_idx"])
    start = int(token["atom_idx"])
    end = start + int(token["atom_num"])
    atoms = data.structure.atoms[start:end]
    valid_indices = [start + i for i, a in enumerate(atoms) if a["element"] != 1]
    if not valid_indices:
        return np.zeros((0, 3), dtype=np.float32)
    idx = offset + np.array(valid_indices, dtype=np.int64)
    return data.structure.coords[idx]["coords"].astype(np.float32)


def process_token_features(  # noqa: C901, PLR0915, PLR0912
    data: Tokenized,
    max_tokens: Optional[int] = None,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_dist_bins: int = 64,
    binder_pocket_conditioned_prop: float = 0.0,
    binder_pocket_cutoff: float = 6.0,
    binder_pocket_sampling_geometric_p: float = 0.0,
    only_ligand_binder_pocket: bool = True,
) -> dict[str, Tensor]:
    """Simplified version of getting token features.

    Parameters
    ----------
    data : Tokenized
        The input data to the model.
    max_tokens : int
        The maximum number of tokens.
    min_dist, max_dist, num_dist_bins
        Distogram binning.
    binder_pocket_conditioned_prop
        If > 0, with this probability activate binder/pocket labels (training).
    binder_pocket_cutoff
        Distance cutoff (Å) for pocket tokens around the binder.
    binder_pocket_sampling_geometric_p
        If > 0, geometric subsampling of pocket tokens.
    only_ligand_binder_pocket
        If True, only ligands can be chosen as binder (default is True, no protein fallback).

    Returns
    -------
    dict[str, Tensor]
        The token features.
    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features (copy to avoid stride issues with structured arrays)
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"].copy()).long()
    asym_id = from_numpy(token_data["asym_id"].copy()).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    res_type = one_hot(res_type, num_classes=const.num_tokens)

    ## Conditioning features ##
    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    disto_mask = from_numpy(token_data["disto_mask"].copy()).float()

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens, num_tokens, dtype=torch.float)
    bonds_type = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1
        bond_type = token_bond["type"]
        bonds_type[token_1, token_2] = bond_type
        bonds_type[token_2, token_1] = bond_type

    bonds = bonds.unsqueeze(-1)

    # Pocket / binder conditioning
    n_tok = len(token_data)
    pocket_labels = np.full(
        n_tok, const.pocket_contact_info["UNSPECIFIED"], dtype=np.int64
    )
    if (
        binder_pocket_conditioned_prop > 0.0
        and random.random() < binder_pocket_conditioned_prop
    ):
        binder_asym_ids = np.unique(
            token_data["asym_id"][
                token_data["mol_type"] == const.chain_type_ids["NONPOLYMER"]
            ]
        )
        if len(binder_asym_ids) == 0 and not only_ligand_binder_pocket:
            binder_asym_ids = np.unique(token_data["asym_id"])

        if len(binder_asym_ids) > 0:
            pocket_asym_id = int(random.choice(binder_asym_ids))
            binder_mask = token_data["asym_id"] == pocket_asym_id

            binder_coords_list = []
            for token in token_data:
                if int(token["asym_id"]) == pocket_asym_id:
                    binder_coords_list.append(_token_atom_coords(data, token))
            if binder_coords_list:
                binder_coords = np.concatenate(binder_coords_list, axis=0)
            else:
                binder_coords = np.zeros((0, 3), dtype=np.float32)

            token_dist = np.full(n_tok, 1000.0, dtype=np.float32)
            for i, token in enumerate(token_data):
                if (
                    token["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    and int(token["asym_id"]) != pocket_asym_id
                    and token["resolved_mask"]
                ):
                    token_coords = _token_atom_coords(data, token)
                    if token_coords.shape[0] == 0 or binder_coords.shape[0] == 0:
                        continue
                    token_dist[i] = float(
                        np.min(
                            np.linalg.norm(
                                token_coords[:, None, :] - binder_coords[None, :, :],
                                axis=-1,
                            )
                        )
                    )

            pocket_mask = token_dist < binder_pocket_cutoff

            if np.sum(pocket_mask) > 0:
                pocket_labels = np.full(
                    n_tok, const.pocket_contact_info["UNSELECTED"], dtype=np.int64
                )
                pocket_labels[binder_mask] = const.pocket_contact_info["BINDER"]

                if binder_pocket_sampling_geometric_p > 0.0:
                    pocket_mask = select_subset_from_mask(
                        pocket_mask, binder_pocket_sampling_geometric_p
                    )

                pocket_labels[pocket_mask.astype(bool)] = const.pocket_contact_info[
                    "POCKET"
                ]

    pocket_feature = from_numpy(pocket_labels.copy()).long()
    pocket_feature = one_hot(
        pocket_feature, num_classes=len(const.pocket_contact_info)
    ).float()

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            unspecified_oh = torch.zeros(
                pad_len, len(const.pocket_contact_info), dtype=torch.float32
            )
            unspecified_oh[:, const.pocket_contact_info["UNSPECIFIED"]] = 1.0
            pocket_feature = torch.cat([pocket_feature, unspecified_oh], dim=0)

    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "token_bonds": bonds,
        "type_bonds": bonds_type,
        "token_pad_mask": pad_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
    }

    return token_features


def _get_atom_name_to_ref(molecules: dict[str, Any], res_name: str):
    """Return dict atom_name -> RDKit Atom for the residue. molecules[res_name] may be Mol or dict."""
    mol_or_ref = molecules.get(res_name)
    # otherwise convert to dict of atom_name -> RDKit Atom
    if (
        res_name == "LIG"
    ):  # these are our own processed ligands, so it needs to be processed differently
        canonical_order = AllChem.CanonicalRankAtoms(mol_or_ref)
        return {
            f"{a.GetSymbol().upper()}{canonical_order[a.GetIdx()] + 1}"[:4]: a
            for a in mol_or_ref.GetAtoms()
            if a.GetAtomicNum() != 1
        }

    # otherwise its a standard molecule processed with CCD, so its fine
    sanitize = lambda x: str(x).strip()
    return {
        sanitize(a.GetProp("name")): a
        for a in mol_or_ref.GetAtoms()
        if a.GetAtomicNum() != 1
    }


def process_atom_features(
    data: Tokenized,
    random: np.random.RandomState,
    molecules: dict[str, Any],
    atoms_per_window_queries: int = 32,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Build atom-level features for the trunk. Only outputs used by encoder/diffusion/loss.

    molecules: res_name -> Mol or dict(atom_name -> Atom). Standard AAs use dict from load_standard_aa_mols.
    ref_pos: conformer coords when available, else random (no structure coords).
    """
    tokens = data.tokens
    structure = data.structure
    offset = structure.ensemble[0]["atom_coord_idx"]
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]
    unk_hybridization = const.hybridization_type_ids.get(
        const.unk_hybridization_type, 0
    )

    atom_name_list = []
    atom_element_list = []
    atom_charge_list = []
    atom_conformer_list = []
    atom_chirality_list = []
    atom_hybridization_list = []
    ref_space_uid_list = []
    coord_data_list = []
    atom_to_token_list = []
    resolved_mask_list = []

    chain_res_ids = {}
    res_index_to_conf_id = {}
    atom_idx = 0

    for token_id, token in enumerate(tokens):
        res_name = str(token["res_name"])
        chain_idx = int(token["asym_id"])
        res_id = int(token["res_idx"])
        if (chain_idx, res_id) not in chain_res_ids:
            chain_res_ids[(chain_idx, res_id)] = len(chain_res_ids)

        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        valid_indices = [
            start + i
            for i, a in enumerate(structure.atoms[start:end])
            if a["element"] != 1
        ]
        token_atoms = structure.atoms[valid_indices]
        n_atoms = len(valid_indices)

        ref_space_uid_list.extend([chain_res_ids[(chain_idx, res_id)]] * n_atoms)
        atom_to_token_list.extend([token_id] * n_atoms)

        sanitize = lambda x: str(x).strip()
        atom_name_list.append(
            np.array([convert_atom_name(sanitize(a["name"])) for a in token_atoms])
        )

        atom_name_to_ref = _get_atom_name_to_ref(molecules, res_name)
        token_atoms_ref = [atom_name_to_ref[sanitize(a["name"])] for a in token_atoms]
        atom_element_list.extend([a.GetAtomicNum() for a in token_atoms_ref])
        atom_charge_list.extend([a.GetFormalCharge() for a in token_atoms_ref])
        atom_chirality_list.extend(
            [
                const.chirality_type_ids.get(a.GetChiralTag().name, unk_chirality)
                for a in token_atoms_ref
            ]
        )
        atom_hybridization_list.extend(
            [
                const.hybridization_type_ids.get(
                    a.GetHybridization().name, unk_hybridization
                )
                for a in token_atoms_ref
            ]
        )
        mol = molecules[res_name]
        if hasattr(mol, "GetConformers") and list(mol.GetConformers()):
            conformers = list(mol.GetConformers())
            if (chain_idx, res_id) not in res_index_to_conf_id:
                res_index_to_conf_id[(chain_idx, res_id)] = random.randint(
                    0, len(conformers)
                )
            conf_id = res_index_to_conf_id[(chain_idx, res_id)]
            conf = conformers[min(conf_id, len(conformers) - 1)]
            conformer_pos = np.array(
                [
                    (
                        conf.GetAtomPosition(a.GetIdx()).x,
                        conf.GetAtomPosition(a.GetIdx()).y,
                        conf.GetAtomPosition(a.GetIdx()).z,
                    )
                    for a in token_atoms_ref
                ],
                dtype=np.float32,
            )
        else:
            # conformer not available so using random conformer from normal distribution
            conformer_pos = random.randn(n_atoms, 3).astype(np.float32)
        atom_conformer_list.append(conformer_pos)

        token_coords = structure.coords[offset + np.array(valid_indices)]["coords"]
        coord_data_list.append(token_coords[np.newaxis, ...])
        resolved_mask_list.append(token_atoms["is_present"].copy())
        atom_idx += n_atoms

    num_atoms = atom_idx
    L = len(tokens)
    atom_name_arr = np.concatenate(atom_name_list, axis=0)
    atom_element = np.array(atom_element_list, dtype=np.int64)
    atom_charge = np.array(atom_charge_list, dtype=np.float32)
    atom_conformer = np.concatenate(atom_conformer_list, axis=0)
    atom_chirality = np.array(atom_chirality_list, dtype=np.int64)
    atom_hybridization = np.array(atom_hybridization_list, dtype=np.int64)
    ref_space_uid = np.array(ref_space_uid_list, dtype=np.int64)
    coord_data = np.concatenate(coord_data_list, axis=1)
    resolved_mask = from_numpy(np.concatenate(resolved_mask_list)).float()

    ref_atom_name_chars = from_numpy(atom_name_arr).long()
    ref_atom_name_chars = one_hot(ref_atom_name_chars, num_classes=64).reshape(
        ref_atom_name_chars.shape[0], -1
    )
    ref_element = one_hot(
        from_numpy(atom_element).long(), num_classes=const.num_elements
    )
    ref_charge = from_numpy(atom_charge.copy()).float()
    ref_chirality = from_numpy(atom_chirality.copy()).long()
    ref_hybridization = from_numpy(atom_hybridization.copy()).long()
    ref_pos = from_numpy(atom_conformer.copy()).float()
    ref_space_uid = from_numpy(ref_space_uid.copy()).long()
    coords = from_numpy(coord_data.copy()).float()

    # Center the ground truth coordinates
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]

    # Apply random roto-translation to the input conformers
    for i in range(torch.max(ref_space_uid) + 1):
        included = ref_space_uid == i
        if torch.sum(included) > 0 and torch.any(resolved_mask[included]):
            ref_pos[included] = center_random_augmentation(
                ref_pos[included][None], resolved_mask[included][None], centering=True
            )[0]

    num_token_classes = max_tokens if max_tokens is not None else L
    atom_to_token = one_hot(
        torch.tensor(atom_to_token_list, dtype=torch.long),
        num_classes=num_token_classes,
    )

    pad_len = (
        (num_atoms - 1) // atoms_per_window_queries + 1
    ) * atoms_per_window_queries - num_atoms
    pad_mask = torch.ones(num_atoms, dtype=torch.float)
    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_chirality = pad_dim(ref_chirality, 0, pad_len)
        ref_hybridization = pad_dim(ref_hybridization, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len)

    atom_features = {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_chirality": ref_chirality,
        "ref_hybridization": ref_hybridization,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "res_index_to_conf_id": res_index_to_conf_id,
    }
    return atom_features


def process_esm_features(
    data: Tokenized,
    record: Record,
    esm_emb_dir: Path,
    use_esm_all_layers: bool = False,
    max_tokens: Optional[int] = None,
    esm_emb_dim: int = 1280,
    esm_num_layers: int = 33,
) -> dict[str, Tensor]:
    """Build ESM single-representation features aligned to tokens."""
    if use_esm_all_layers:
        esm_full = torch.zeros(len(data.tokens), esm_num_layers, esm_emb_dim)
    else:
        esm_full = torch.zeros(len(data.tokens), esm_emb_dim)

    record_chains = record.chains
    for chain in data.structure.chains:
        if chain["mol_type"] != const.chain_type_ids["PROTEIN"]:
            continue
        record_chain = next(c for c in record_chains if c.chain_name == chain["name"])
        chain_token_mask = (data.tokens["asym_id"] == chain["asym_id"]) & (
            data.tokens["mol_type"] == 0
        )
        if np.any(chain_token_mask):
            msa_id = record_chain.msa_id
            esm_emb_path = esm_emb_dir / f"{msa_id}.safetensors"
            if not esm_emb_path.exists():
                raise ValueError(f"ESM embedding file not found: {esm_emb_path}")
            chain_esm = extract_esm_features(
                esm_emb_path, use_final_layer_only=(not use_esm_all_layers)
            )
            local_res_idxs = data.tokens["res_idx"][chain_token_mask]
            if chain["res_num"] != chain_esm.shape[0]:
                raise ValueError("ESM embedding length does not match chain length")
            torch_mask = torch.from_numpy(chain_token_mask)
            esm_full[torch_mask] = chain_esm[torch.from_numpy(local_res_idxs).long()]

    s_esm = esm_full
    if max_tokens is not None and s_esm.shape[0] < max_tokens:
        pad_shape = list(s_esm.shape)
        pad_shape[0] = max_tokens - s_esm.shape[0]
        s_esm = torch.cat([s_esm, torch.zeros(*pad_shape)], dim=0)
    return {"s_esm": s_esm}


class NessoFeaturizer:
    """Orchestrates token, atom, and ESM feature extraction."""

    def __init__(
        self,
        esm_emb_dir: Path,
        esm_emb_dim: int = 1280,
        esm_num_layers: int = 33,
    ):
        self.esm_emb_dir = Path(esm_emb_dir)
        self.esm_emb_dim = esm_emb_dim
        self.esm_num_layers = esm_num_layers

    def process(
        self,
        tokenized: Tokenized,
        record: Record,
        molecules: dict[str, Any],
        random: RandomState,
        num_dist_bins: int = 64,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        use_esm_all_layers: bool = False,
        atoms_per_window_queries: int = 32,
        binder_pocket_conditioned_prop: float = 0.0,
        binder_pocket_cutoff: float = 6.0,
        binder_pocket_sampling_geometric_p: float = 0.0,
        only_ligand_binder_pocket: bool = False,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Build model feature dict for one ``tokenized`` example."""
        esm_features = process_esm_features(
            tokenized,
            record=record,
            esm_emb_dir=self.esm_emb_dir,
            max_tokens=max_tokens,
            use_esm_all_layers=use_esm_all_layers,
            esm_emb_dim=self.esm_emb_dim,
            esm_num_layers=self.esm_num_layers,
        )
        token_features = process_token_features(
            tokenized,
            max_tokens=max_tokens,
            num_dist_bins=num_dist_bins,
            min_dist=min_dist,
            max_dist=max_dist,
            binder_pocket_conditioned_prop=binder_pocket_conditioned_prop,
            binder_pocket_cutoff=binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=binder_pocket_sampling_geometric_p,
            only_ligand_binder_pocket=only_ligand_binder_pocket,
        )
        atom_features = process_atom_features(
            tokenized,
            random=random,
            molecules=molecules,
            atoms_per_window_queries=atoms_per_window_queries,
            max_tokens=max_tokens,
        )
        atom_features.pop("res_index_to_conf_id")
        return {**esm_features, **token_features, **atom_features}
