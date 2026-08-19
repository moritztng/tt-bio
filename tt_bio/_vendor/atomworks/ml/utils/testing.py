import os

import numpy as np
from biotite.structure import AtomArray, CellList

from tt_bio._vendor.atomworks.common import immutable_lru_cache
from tt_bio._vendor.atomworks.constants import PDB_MIRROR_PATH
from tt_bio._vendor.atomworks.io import parse
from tt_bio._vendor.atomworks.ml.preprocessing.constants import CELL_SIZE
from tt_bio._vendor.atomworks.ml.preprocessing.utils.structure_utils import get_atom_mask_from_cell_list


def get_pdb_mirror_path(pdbid: str, base_dir: str = PDB_MIRROR_PATH) -> str:
    """Convenience util to get the path to a CIF file on the DIGS"""
    # Assert that the base directory exists
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"

    # Build the path to the file
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@immutable_lru_cache(maxsize=1000, deepcopy=True)
def cached_parse(pdb_id: str, **kwargs) -> dict:
    """Wrapper around parse with caching to return an independent copy of the output dict."""
    data = parse(filename=get_pdb_mirror_path(pdb_id), **kwargs)
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data


def is_clash(atom_array_1: AtomArray, atom_array_2: AtomArray, clash_distance: float = 1.0) -> bool:
    """
    Checks for clashes between two arrays. Based on atomworks.ml.preprocessing.process.DataPreprocessor.
    Recommended to pass in minimal masks of arrays to check to reduce runtime.
    """
    cell_list = CellList(atom_array_1, cell_size=CELL_SIZE)

    clashing_atom_mask = get_atom_mask_from_cell_list(atom_array_2.coord, cell_list, len(atom_array_1), clash_distance)

    return np.any(clashing_atom_mask)
