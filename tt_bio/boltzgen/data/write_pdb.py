import re

from tt_bio.data import const
from tt_bio.data.pdb import chain_id_at, remarks
from tt_bio.boltzgen.data.data import Structure


def to_pdb(structure: Structure) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output PDB file

    """
    pdb_lines = []

    atom_index = 1
    atom_reindex_ter = []
    shortened_res_names: dict[str, str] = {}
    renamed_chains: dict[str, str] = {}

    # Add all atom sites.
    for tag_index, chain in enumerate(structure.chains):
        # We rename the chains in alphabetical order. The old generator ran A..Z then AA, AB,
        # ... , and a two-character tag written as `{chain_tag:>1}` shifted every column after
        # it: past 26 chains the file was silently malformed. chain_id_at stays one character
        # for all 62 a PDB can hold and refuses the 63rd with a readable message.
        chain_idx = chain["asym_id"]
        chain_tag = chain_id_at(tag_index)
        if str(chain["name"]) != chain_tag:
            renamed_chains[str(chain["name"])] = chain_tag

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        residues = structure.residues[res_start:res_end]
        for residue in residues:
            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            atoms = structure.atoms[atom_start:atom_end]
            atom_coords = atoms["coords"]
            res_name = str(residue["name"])
            if len(res_name) > 3:  # noqa: PLR2004 -- the resName column is 3 wide
                shortened_res_names[res_name] = res_name[:3]
            for i, atom in enumerate(atoms):
                atom_reindex_ter.append(atom_index)

                # This should not happen on predictions, but just in case.
                if not atom["is_present"]:
                    continue

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )
                atom_name = atom["name"]
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                atom_key = re.sub(r"\d", "", atom_name)
                if atom_key in const.ambiguous_atoms:
                    if isinstance(const.ambiguous_atoms[atom_key], str):
                        element = const.ambiguous_atoms[atom_key]
                    elif res_name in const.ambiguous_atoms[atom_key]:
                        element = const.ambiguous_atoms[atom_key][res_name]
                    else:
                        element = const.ambiguous_atoms[atom_key]["*"]
                else:
                    element = atom_key[0]

                charge = ""
                residue_index = residue["res_idx"] + 1
                pos = atom_coords[i]
                res_name_3 = "LIG" if record_type == "HETATM" else res_name[:3]
                b_factor = 1.00

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {atom_name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_index:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_index += 1

        should_terminate = chain_idx < (len(structure.chains) - 1)
        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_name_3:>3} "
                f"{chain_tag:>1}{residue_index:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

    # Dump CONECT records.
    for bond in structure.bonds:
        atom1 = structure.atoms[bond["atom_1"]]
        atom2 = structure.atoms[bond["atom_2"]]
        if not atom1["is_present"] or not atom2["is_present"]:
            continue
        atom1_idx = atom_reindex_ter[bond["atom_1"]]
        atom2_idx = atom_reindex_ter[bond["atom_2"]]
        conect_line = f"CONECT{atom1_idx:>5}{atom2_idx:>5}"
        pdb_lines.append(conect_line)

    pdb_lines.append("END")
    pdb_lines.append("")
    pdb_lines = remarks(renamed_chains, shortened_res_names) + pdb_lines
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines)
