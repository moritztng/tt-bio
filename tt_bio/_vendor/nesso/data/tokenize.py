# Adapted from https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024 Jeremy Wohlwend
from dataclasses import dataclass

import numpy as np

from tt_bio._vendor.nesso.data import const
from tt_bio._vendor.nesso.data.types import Structure, TokenBond, Token


@dataclass
class TokenData:
    """TokenData datatype."""

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    res_name: str
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: np.ndarray
    disto_coords: np.ndarray
    resolved_mask: bool
    disto_mask: bool
    modified: bool


def token_astuple(token: TokenData) -> tuple:
    """Convert a TokenData object to a tuple."""
    return (
        token.token_idx,
        token.atom_idx,
        token.atom_num,
        token.res_idx,
        token.res_type,
        token.res_name,
        token.sym_id,
        token.asym_id,
        token.entity_id,
        token.mol_type,
        token.center_idx,
        token.disto_idx,
        token.center_coords,
        token.disto_coords,
        token.resolved_mask,
        token.disto_mask,
        token.modified,
    )


def get_unk_token(chain: np.ndarray) -> int:
    """Get the unk token for a residue.

    Parameters
    ----------
    chain : np.ndarray
        The chain.

    Returns
    -------
    int
        The unk token.

    """
    if chain["mol_type"] == const.chain_type_ids["DNA"]:
        unk_token = const.unk_token["DNA"]
    elif chain["mol_type"] == const.chain_type_ids["RNA"]:
        unk_token = const.unk_token["RNA"]
    else:
        unk_token = const.unk_token["PROTEIN"]

    res_id = const.token_ids[unk_token]
    return res_id


def tokenize_structure(  # noqa: C901, PLR0915
    struct: Structure,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize a structure.

    Parameters
    ----------
    struct : Structure
        The structure to tokenize.

    Returns
    -------
    np.ndarray
        The tokenized data.
    np.ndarray
        The tokenized bonds.

    """
    # Create token data
    token_data = []

    # Keep track of atom_idx to token_idx
    token_idx = 0
    atom_to_token = {}

    # Filter to valid chains only
    chains = struct.chains[struct.mask]

    # Ensemble atom id start in coords table.
    # For cropper and other operations, hardcoded to 0th conformer.
    offset = struct.ensemble[0]["atom_coord_idx"]

    for chain in chains:
        # Get residue indices
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        for res in struct.residues[res_start:res_end]:
            # Get atom indices
            atom_start = res["atom_idx"]
            atom_end = res["atom_idx"] + res["atom_num"]

            # Standard residues are tokens
            if res["is_standard"]:
                # Get center and disto atoms
                # if atom_start > res["atom_center"], it
                # means the center and disto atoms are relative
                # to the start of the residue, otherwise they are relative
                if atom_start > res["atom_center"]:
                    atom_center_idx = res["atom_center"] + atom_start
                    atom_disto_idx = res["atom_disto"] + atom_start
                else:
                    atom_center_idx = res["atom_center"]
                    atom_disto_idx = res["atom_disto"]

                # check if center and disto atoms are correct
                center = struct.atoms[atom_center_idx]
                disto = struct.atoms[atom_disto_idx]

                # Token is present if centers are
                is_present = res["is_present"] & center["is_present"]
                is_disto_present = res["is_present"] & disto["is_present"]

                # Apply chain transformation
                c_coords = struct.coords[offset + atom_center_idx]["coords"]
                d_coords = struct.coords[offset + atom_disto_idx]["coords"]

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    atom_idx=res["atom_idx"],
                    atom_num=res["atom_num"],
                    res_idx=res["res_idx"],
                    res_type=res["res_type"],
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    center_idx=atom_center_idx,
                    disto_idx=atom_disto_idx,
                    center_coords=c_coords,
                    disto_coords=d_coords,
                    resolved_mask=is_present,
                    disto_mask=is_disto_present,
                    modified=False,
                )
                token_data.append(token_astuple(token))

                # Update atom_idx to token_idx
                atom_to_token.update(
                    dict.fromkeys(range(atom_start, atom_end), token_idx)
                )

                token_idx += 1

            # Non-standard are tokenized per atom
            elif chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]:
                # We use the unk protein token as res_type
                unk_token = const.unk_token["PROTEIN"]
                unk_id = const.token_ids[unk_token]

                # Get atom coordinates
                atom_data = struct.atoms[atom_start:atom_end]
                atom_coords = struct.coords[offset + atom_start : offset + atom_end][
                    "coords"
                ]

                # Tokenize each atom
                for i, atom in enumerate(atom_data):
                    if atom["element"] == 1:
                        continue  # skip hydrogens

                    # Token is present if atom is
                    is_present = res["is_present"] & atom["is_present"]
                    index = atom_start + i

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=index,
                        atom_num=1,
                        res_idx=res["res_idx"],
                        res_type=unk_id,
                        res_name=res["name"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=index,
                        disto_idx=index,
                        center_coords=atom_coords[i],
                        disto_coords=atom_coords[i],
                        resolved_mask=is_present,
                        disto_mask=is_present,
                        modified=chain["mol_type"]
                        != const.chain_type_ids["NONPOLYMER"],
                    )
                    token_data.append(token_astuple(token))

                    # Update atom_idx to token_idx
                    atom_to_token[index] = token_idx
                    token_idx += 1

            # Modified residues are tokenized at residue level
            else:
                res_type = get_unk_token(chain)

                # Get center and disto atoms
                if atom_start > res["atom_center"]:
                    atom_center_idx = res["atom_center"] + atom_start
                    atom_disto_idx = res["atom_disto"] + atom_start
                else:
                    atom_center_idx = res["atom_center"]
                    atom_disto_idx = res["atom_disto"]

                center = struct.atoms[atom_center_idx]
                disto = struct.atoms[atom_disto_idx]

                # Token is present if centers are
                is_present = res["is_present"] & center["is_present"]
                is_disto_present = res["is_present"] & disto["is_present"]

                # Apply chain transformation
                c_coords = struct.coords[offset + atom_center_idx]["coords"]
                d_coords = struct.coords[offset + atom_disto_idx]["coords"]

                # Create token
                token = TokenData(
                    token_idx=token_idx,
                    atom_idx=res["atom_idx"],
                    atom_num=res["atom_num"],
                    res_idx=res["res_idx"],
                    res_type=res_type,
                    res_name=res["name"],
                    sym_id=chain["sym_id"],
                    asym_id=chain["asym_id"],
                    entity_id=chain["entity_id"],
                    mol_type=chain["mol_type"],
                    center_idx=atom_center_idx,
                    disto_idx=atom_disto_idx,
                    center_coords=c_coords,
                    disto_coords=d_coords,
                    resolved_mask=is_present,
                    disto_mask=is_disto_present,
                    modified=True,
                )
                token_data.append(token_astuple(token))

                # Update atom_idx to token_idx
                atom_to_token.update(
                    dict.fromkeys(range(atom_start, atom_end), token_idx)
                )

                token_idx += 1

    # Create token bonds
    token_bonds = []

    # Add atom-atom bonds from ligands
    for bond in struct.bonds:
        if bond["atom_1"] not in atom_to_token or bond["atom_2"] not in atom_to_token:
            continue
        token_bond = (
            atom_to_token[bond["atom_1"]],
            atom_to_token[bond["atom_2"]],
            bond["type"] + 1,
        )
        token_bonds.append(token_bond)

    token_data = np.array(token_data, dtype=Token)
    token_bonds = np.array(token_bonds, dtype=TokenBond)

    return token_data, token_bonds
