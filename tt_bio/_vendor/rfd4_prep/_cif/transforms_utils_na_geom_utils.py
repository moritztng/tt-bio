# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 5248e9a15a187c75a31b6a58d42efc0b3d6b543f.
import math

import biotite.structure as struc
import numpy as np
from tt_bio._vendor.atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from tt_bio._vendor.atomworks.io.utils.annotator import ensure_annotations
from tt_bio._vendor.atomworks.io.utils.sequence import is_purine, is_pyrimidine
from tt_bio._vendor.atomworks.ml.encoding_definitions import AF3SequenceEncoding
from tt_bio._vendor.atomworks.ml.transforms.interactions import AnnotateInteractions
from tt_bio._vendor.atomworks.ml.utils.token import (
    get_token_starts,
    is_glycine,
    is_protein_unknown,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
    spread_token_wise,
)
from biotite.structure import AtomArray

from .utils_constants import (
    ATOM_REGION_BY_RESI,
    PLANAR_ATOMS_BY_RESI,
)


def _set_canonical_na_ss_from_token_state(
    atom_array: AtomArray,
    *,
    token_starts: np.ndarray,
    state_values: np.ndarray,
    pair_inds: np.ndarray,
) -> AtomArray:
    """Write canonical NA-SS conditions from token-level state and pair indices.

    Args:
        atom_array: Structure to annotate.
        token_starts: Residue representative atom indices.
        state_values: Per-token ``BasePairEnum`` values.
        pair_inds: Directed token-index pairs of shape ``(N, 2)``.

    Returns:
        The same AtomArray with canonical NA-SS conditions set.
    """
    from .conditions import C_BP, C_NA_SS_STATE, BasePairEnum

    token_starts_with_stop = np.concatenate([np.asarray(token_starts, dtype=np.int32), [atom_array.array_length()]])
    atom_level_state = spread_token_wise(atom_array, state_values, token_starts=token_starts_with_stop)
    C_NA_SS_STATE.set_annotation(atom_array, atom_level_state.astype(np.int16))

    if len(pair_inds) > 0:
        pair_values = np.full((len(pair_inds),), BasePairEnum.PAIR.value, dtype=np.int64)
        atom_pairs = token_starts[pair_inds]
    else:
        pair_values = np.empty((0,), dtype=np.int64)
        atom_pairs = np.empty((0, 2), dtype=np.int32)

    C_BP.set_annotation(
        atom_array,
        pairs=atom_pairs.astype(np.int16),
        values=pair_values,
    )
    return atom_array


def _set_canonical_na_ss_from_token_payload(
    atom_array: AtomArray,
    *,
    token_starts: np.ndarray,
    parsed_nucleic_ss: dict[str, list],
) -> AtomArray:
    """Write canonical NA-SS conditions from token-index payloads.

    Args:
        atom_array: Structure to annotate.
        token_starts: Residue representative atom indices.
        parsed_nucleic_ss: Canonical token-level payload produced by input parsing.

    Returns:
        The same AtomArray with canonical NA-SS conditions set.
    """
    from .conditions import BasePairEnum

    n_tokens = len(token_starts)
    state_values = np.full((n_tokens,), BasePairEnum.UNSPECIFIED.value, dtype=np.int16)

    for tok in parsed_nucleic_ss.get("loop_token_indices", []):
        tok = int(tok)
        if 0 <= tok < n_tokens:
            state_values[tok] = BasePairEnum.LOOP.value

    for tok in parsed_nucleic_ss.get("specified_token_indices", []):
        tok = int(tok)
        if 0 <= tok < n_tokens and state_values[tok] == BasePairEnum.UNSPECIFIED.value:
            state_values[tok] = BasePairEnum.PAIR.value

    pair_inds: list[tuple[int, int]] = []
    for pair in parsed_nucleic_ss.get("pair_token_indices", []):
        if len(pair) != 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < n_tokens and 0 <= j < n_tokens and i != j:
            pair_inds.append((i, j))

    return _set_canonical_na_ss_from_token_state(
        atom_array,
        token_starts=np.asarray(token_starts, dtype=np.int32),
        state_values=state_values.astype(np.int16),
        pair_inds=np.asarray(pair_inds, dtype=np.int32).reshape(-1, 2),
    )


# Derived: True when the residue has any planar sidechain atoms
HAS_PLANAR_SC = {res: bool(atoms) for res, atoms in PLANAR_ATOMS_BY_RESI.items()}

AA_PLANAR_ATOMS = sorted(
    {atom for res in STANDARD_AA if res in PLANAR_ATOMS_BY_RESI for atom in PLANAR_ATOMS_BY_RESI[res]}
)

NA_PLANAR_ATOMS = sorted(
    {
        atom
        for res in (*STANDARD_RNA, *STANDARD_DNA)
        if res in PLANAR_ATOMS_BY_RESI
        for atom in PLANAR_ATOMS_BY_RESI[res]
    }
)


class NucMolInfo:
    """Constants and parameters for nucleic-acid geometry and interaction scoring.

    All parameters are set to empirically validated defaults. No constructor
    arguments are currently accepted.
    """

    def __init__(self) -> None:
        # Hbond interaction-class indices of the `hbond_count`` array:
        # `hbond_count`` array is (L, L, 3), where the last dimension
        # encodes interaction type between tokens i & j
        self.BB_BB = 0  # backbone-backbone hbond interactions
        self.BB_SC = 1  # backbone-sidechain hbond interactions
        self.SC_SC = 2  # sidechain-sidechain hbond interactions

        # We sum over the last dimension of the hbond_count array, scaling
        # count by the following weights to get the interaction score:
        self.bp_weight_BB_BB = 0.0
        self.bp_weight_BB_SC = 0.5
        self.bp_weight_SC_SC = 1.0
        self.bp_summation_weights = [self.bp_weight_BB_BB, self.bp_weight_BB_SC, self.bp_weight_SC_SC]

        # Parameters fo sigmoid function that gives us a continuous step function for
        # meeting basepair interaction criteria based on hbond counts alone (1st filter).
        # Calibrated such that:
        # >= 2 base-base H-bonds -> ~1.0
        # 1 base-base H-bond + 1 base-backbone H-bond -> ~0.5
        self.min_hbonds_for_bp = 2.0
        self.bp_hbond_coeff = 9.8  # determined heuristically
        self.bp_val_cutoff = 0.5  # minimum basepairing score for binarizing basepairs when needed

        self.base_geometry_limits = {}
        self.base_geometry_limits["D_ij"] = 20.0
        self.base_geometry_limits["H_ij"] = 1.5
        self.base_geometry_limits["P_ij"] = math.pi / 5
        self.base_geometry_limits["B_ij"] = math.pi / 5

        self.rep_atom_dict = {"protein": "CA", "rna": "C1'", "dna": "C1'"}

        # Remove spaces from atom names and canonicalize outer dict keys.
        self.vec_atom_dict = {
            "DA": {
                "W_start": "N1",
                "W_stop": "N6",
                "H_start": "N7",
                "H_stop": "N6",
                "S_start": "C1'",
                "S_stop": "N3",
                "B_start": "C1'",
                "B_stop": "N9",
            },
            "DG": {
                "W_start": "N1",
                "W_stop": "O6",
                "H_start": "N7",
                "H_stop": "O6",
                "S_start": "C1'",
                "S_stop": "N3",
                "B_start": "C1'",
                "B_stop": "N9",
            },
            "DC": {
                "W_start": "N3",
                "W_stop": "N4",
                "H_start": "C5",
                "H_stop": "N4",
                "S_start": "C1'",
                "S_stop": "O2",
                "B_start": "C1'",
                "B_stop": "N1",
            },
            "DT": {
                "W_start": "N3",
                "W_stop": "O4",
                "H_start": "C5",
                "H_stop": "O4",
                "S_start": "C1'",
                "S_stop": "O2",
                "B_start": "C1'",
                "B_stop": "N1",
            },
            "A": {
                "W_start": "N1",
                "W_stop": "N6",
                "H_start": "N7",
                "H_stop": "N6",
                "S_start": "C1'",
                "S_stop": "N3",
                "B_start": "C1'",
                "B_stop": "N9",
            },
            "G": {
                "W_start": "N1",
                "W_stop": "O6",
                "H_start": "N7",
                "H_stop": "O6",
                "S_start": "C1'",
                "S_stop": "N3",
                "B_start": "C1'",
                "B_stop": "N9",
            },
            "C": {
                "W_start": "N3",
                "W_stop": "N4",
                "H_start": "C5",
                "H_stop": "N4",
                "S_start": "C1'",
                "S_stop": "O2",
                "B_start": "C1'",
                "B_stop": "N1",
            },
            "U": {
                "W_start": "N3",
                "W_stop": "O4",
                "H_start": "C5",
                "H_stop": "O4",
                "S_start": "C1'",
                "S_stop": "O2",
                "B_start": "C1'",
                "B_stop": "N1",
            },
        }


def find_planar_positions(
    atom_array: AtomArray,
    mol_info: NucMolInfo,
    tol: float = 1e-2,
) -> dict:
    """Identify residues with planar sidechains via known canonical atom lists.

    For canonical residues the planar atoms are looked up from ``mol_info``.
    Non-canonical residues are treated conservatively and are not assigned
    planar atoms via geometric heuristics.

    Args:
        atom_array: Structure to analyse.
        mol_info: Molecular-info object supplying per-residue planar atom lists.
        tol: Unused compatibility argument retained for call-site stability.

    Returns:
        Dictionary ``{(chain_iid, res_id): [atom_name, ...]}`` mapping each
        unique residue position to its list of planar sidechain atom names.
    """
    seen_positions: set = set()
    unique_positions_list = []
    for atm in atom_array:
        pos_id = (atm.chain_iid, atm.res_id, atm.res_name)
        if pos_id not in seen_positions:
            seen_positions.add(pos_id)
            unique_positions_list.append(pos_id)

    # Get candidate planar atoms:
    planar_atom_list_dict = {}

    # for chain_iid, res_id in unique_positions_list:
    for chain_iid, res_id, res_name in unique_positions_list:
        mask = (atom_array.chain_iid == chain_iid) & (atom_array.res_id == res_id) & (atom_array.res_name == res_name)
        res_atoms = atom_array[mask]

        # If possible, speed up by using known planar atoms for this residue type:
        if res_name in PLANAR_ATOMS_BY_RESI:
            # Shared atoms between residue and known planar atoms for that residue type:
            planar_atom_list = list({atm.atom_name for atm in res_atoms} & set(PLANAR_ATOMS_BY_RESI[res_name]))
            planar_atom_list_dict[(chain_iid, res_id)] = planar_atom_list

        # Non-canonical residues are excluded from planarity-dependent handling.
        else:
            planar_atom_list_dict[(chain_iid, res_id)] = []

    return planar_atom_list_dict


def calculate_hb_counts(atom_array: AtomArray, token_level_data: dict) -> np.ndarray:
    """Count AtomWorks PLIP hydrogen bonds by residue and backbone/base region.

    Protonation and donor/acceptor typing use the same AtomWorks model as the
    interaction metrics. Both intra-chain and inter-chain contacts are included.
    The input coordinates and annotations are preserved.

    Args:
        atom_array: Structure with bonds and protonation annotations.
        token_level_data: Residue metadata including ``token_starts``.

    Returns:
        Symmetric int32 counts with shape ``(n_residues, n_residues, 3)``;
        the channels are backbone-backbone, backbone-sidechain, sidechain-sidechain.
    """
    residue_starts = token_level_data["token_starts"]
    counts = np.zeros((len(residue_starts), len(residue_starts), 3), dtype=np.int32)
    annotated = AnnotateInteractions(
        interaction_types=("hbond",),
        hbond_model="plip",
        inter_chain_only=False,
    )({"atom_array": atom_array.copy()})["atom_array"]
    pairs = np.unique(annotated.get_annotation("interaction_hbond", n_body=2).pairs, axis=0)
    if not len(pairs):
        return counts

    ensure_annotations(annotated, "is_ligand")
    regions = np.full(len(annotated), -1, dtype=np.int8)
    for atom_index in np.unique(pairs):
        atom_regions = ATOM_REGION_BY_RESI.get(annotated.res_name[atom_index])
        if atom_regions is not None:
            for region_index, region in enumerate(("bb", "sc")):
                if annotated.atom_name[atom_index] in atom_regions[region]:
                    regions[atom_index] = region_index
        elif annotated.is_ligand[atom_index]:
            regions[atom_index] = 0

    residue_pairs = np.searchsorted(residue_starts, pairs, side="right") - 1
    valid = np.all(regions[pairs] >= 0, axis=1) & (residue_pairs[:, 0] != residue_pairs[:, 1])
    donors, acceptors = residue_pairs[valid].T
    channels = regions[pairs[valid]].sum(axis=1)
    np.add.at(counts, (donors, acceptors, channels), 1)
    np.add.at(counts, (acceptors, donors, channels), 1)
    return counts


def make_coord_list(
    atom_array: AtomArray,
    residue_list: list[str],
    chain_iid_list: list[str],
    atom_list: list[str],
) -> list[list[str]]:
    """Extract per-residue representative coordinates from an AtomArray.

    All three input lists must have the same length. Missing atoms are
    filled with ``[NaN, NaN, NaN]``.

    Args:
        atom_array: Biotite AtomArray to query.
        residue_list: Residue IDs (one per token).
        chain_iid_list: Assembly-unique chain identifiers (one per token).
        atom_list: Atom names to extract.

    Returns:
        List of ``[x, y, z]`` coordinate lists, same length as input.
    """
    coord_list = []
    for res_id, chain_iid, atom_name in zip(residue_list, chain_iid_list, atom_list, strict=False):
        # Should have a unique solution, but we take the first entry either way.
        mask = (atom_array.chain_iid == chain_iid) & (atom_array.res_id == res_id) & (atom_array.atom_name == atom_name)

        # Get the coordinates for the masked atoms
        coords = atom_array.coord[mask][0:1]

        if len(coords) < 1:
            coord_list.append([float("nan"), float("nan"), float("nan")])
        else:
            coord_list.append(coords[0].tolist())

    return coord_list


def get_token_level_metadata(
    atom_array: AtomArray,
    mol_info: "NucMolInfo",
    *,
    NA_only: bool = False,
    planar_only: bool = True,
    seq_cutoff: int = 2,
    gap_length: int = 200,
) -> dict:
    """Build lightweight token-level metadata (no coordinate geometry).

    Sufficient for NA-SS reconstruction, loop labeling, and inference-time SS
    specification parsing. For geometry keys (``xyz_planar``, ``frame_xyz``,
    ``M_i``), follow up with :func:`add_token_level_geometry_data`.

    Args:
        atom_array: Structure to analyse.
        mol_info: Molecular-info constants.
        NA_only: If True, restrict filter_mask to nucleic-acid tokens.
        planar_only: If True, restrict filter_mask to tokens with planar
            sidechains.
        seq_cutoff: Sequence-distance threshold for the ``seq_neighbors``
            boolean mask.
        gap_length: Artificial gap inserted between chains for relative
            sequence position computation.

    Returns:
        Dict with keys: ``token_starts``, ``token_index``, ``is_na``,
        ``is_planar``, ``chain_list``, ``chain_iid_list``, ``resi_list``,
        ``resn_list``, ``token_id_list``, ``resi2index``, ``len_s``,
        ``seq_neighbors``, ``na_inds``, ``na_tensor_inds``,
        ``filter_mask``, ``rep_atom_list``, ``S_start_atom_list``,
        ``S_stop_atom_list``, ``include_geometry`` (False).
    """

    # Use residue starts (not token starts) so atomized atoms within one residue
    # map to a single NA-SS position.
    token_starts = struc.get_residue_starts(atom_array)
    token_level_array = atom_array[token_starts]

    token_index = np.arange(len(token_starts))

    # molecule type flags
    # Instantiate encoding locally to avoid retaining large arrays at module scope.
    sequence_encoding = AF3SequenceEncoding()
    is_rna = np.isin(token_level_array.res_name, sequence_encoding.all_res_names[sequence_encoding.is_rna_like])
    is_dna = np.isin(token_level_array.res_name, sequence_encoding.all_res_names[sequence_encoding.is_dna_like])

    is_na_arr = (is_dna | is_rna).astype(bool)

    chain_list: list[str] = []
    chain_iid_list: list[str] = []
    resi_list: list[int] = []
    ind_list: list[int] = []
    res_name_list: list[str] = []
    token_id_list: list[str] = []

    rep_atom_list: list[str | None] = []
    S_start_atom_list: list[str | None] = []
    S_stop_atom_list: list[str | None] = []
    sc_planarity_list: list[bool] = []

    for i, atm in enumerate(token_level_array):
        chain_list.append(atm.chain_id)
        chain_iid_list.append(atm.chain_iid)
        resi_list.append(int(atm.res_id))
        ind_list.append(int(i))
        res_name_list.append(atm.res_name)
        token_id_list.append(str(atm.token_id))

        if atm.is_polymer and (atm.res_name in HAS_PLANAR_SC):
            sc_planarity_list.append(bool(HAS_PLANAR_SC[atm.res_name]))
        else:
            sc_planarity_list.append(False)

        # representative & sugar-edge atoms
        if is_glycine(atm.res_name) | is_protein_unknown(atm.res_name):
            rep_atom_i = "CA"
            S_start_atom_i = None
            S_stop_atom_i = None
        elif is_standard_aa_not_glycine(atm.res_name):
            rep_atom_i = "CA"
            S_start_atom_i = "CA"
            S_stop_atom_i = "CB"
        elif is_pyrimidine(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = "C1'"
            S_stop_atom_i = "O2"
        elif is_purine(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = "C1'"
            S_stop_atom_i = "N3"
        elif is_unknown_nucleotide(atm.res_name):
            rep_atom_i = "C1'"
            S_start_atom_i = None
            S_stop_atom_i = None
        elif getattr(atm, "atomize", False):
            rep_atom_i = atm.atom_name
            S_start_atom_i = None
            S_stop_atom_i = None
        else:
            rep_atom_i = None
            S_start_atom_i = None
            S_stop_atom_i = None

        rep_atom_list.append(rep_atom_i)
        S_start_atom_list.append(S_start_atom_i)
        S_stop_atom_list.append(S_stop_atom_i)

    # residue index <-> token index map
    resi2index = {f"{c}__{r}": i for c, r, i in zip(chain_iid_list, resi_list, ind_list, strict=False)}

    # relative sequence positions w/ chain gaps
    rel_pos_list: list[int] = []
    current_chain = ""
    chn_bias = -gap_length
    for r, c in zip(resi_list, chain_iid_list, strict=False):
        if c != current_chain:
            chn_bias += gap_length
            current_chain = c
        rel_pos_list.append(int(r + chn_bias))

    rel_pos = np.asarray(rel_pos_list, dtype=np.int64)
    seq_neighbors = np.abs(rel_pos[:, None] - rel_pos[None, :]) <= int(seq_cutoff)

    na_inds = np.nonzero(is_na_arr)[0].tolist()
    na_tensor_inds = {na_i: i for i, na_i in enumerate(na_inds)}

    # Cheap planarity heuristic from residue name lookup
    is_planar_arr = np.asarray(sc_planarity_list, dtype=bool)

    # filter mask using NA_only / planar_only flags
    if NA_only and planar_only:
        filter_mask = is_na_arr & is_planar_arr
    elif NA_only and (not planar_only):
        filter_mask = is_na_arr.copy()
    elif (not NA_only) and planar_only:
        filter_mask = is_planar_arr.copy()
    else:
        filter_mask = np.ones_like(is_na_arr, dtype=bool)

    return {
        "token_starts": token_starts,
        "token_index": token_index,
        "is_na": is_na_arr,
        "is_planar": is_planar_arr,
        "chain_list": chain_list,
        "chain_iid_list": chain_iid_list,
        "resi_list": resi_list,
        "resn_list": res_name_list,
        "token_id_list": token_id_list,
        "resi2index": resi2index,
        "len_s": len(token_level_array),
        "seq_neighbors": seq_neighbors,
        "na_inds": na_inds,
        "na_tensor_inds": na_tensor_inds,
        "filter_mask": filter_mask,
        "rep_atom_list": rep_atom_list,
        "S_start_atom_list": S_start_atom_list,
        "S_stop_atom_list": S_stop_atom_list,
        "include_geometry": False,
    }


def add_token_level_geometry_data(
    atom_array: AtomArray,
    mol_info: "NucMolInfo",
    token_level_data: dict,
    *,
    NA_only: bool = False,
    planar_only: bool = True,
) -> dict:
    """Augment token-level metadata with coordinate-derived geometry fields.

    Populates ``xyz_planar``, ``xyz_S_start``, ``xyz_S_stop``,
    ``frame_xyz``, ``M_i`` and updates ``is_planar`` / ``filter_mask``
    using coordinate-derived planarity. Sets ``include_geometry=True``.

    No-ops if geometry was already computed.

    Args:
        atom_array: Structure to extract coordinates from.
        mol_info: Molecular-info constants.
        token_level_data: Dict produced by :func:`get_token_level_metadata`
            (modified in-place and returned).
        NA_only: Restrict filter_mask to nucleic-acid tokens.
        planar_only: Restrict filter_mask to tokens with planar sidechains.

    Returns:
        The same ``token_level_data`` dict, augmented with geometry keys.
    """

    if bool(token_level_data.get("include_geometry", False)):
        return token_level_data

    # Backward-compatibility: older token_level_data dicts (or user-provided ones)
    # may not contain the metadata keys this function needs.
    required_keys = (
        "chain_iid_list",
        "chain_list",
        "resi_list",
        "rep_atom_list",
        "S_start_atom_list",
        "S_stop_atom_list",
        "is_na",
    )
    if any(k not in token_level_data for k in required_keys):
        token_level_data = get_token_level_metadata(
            atom_array,
            mol_info,
            NA_only=NA_only,
            planar_only=planar_only,
        )

    chain_iid_list: list[str] = token_level_data["chain_iid_list"]
    resi_list: list[int] = token_level_data["resi_list"]
    rep_atom_list: list[str | None] = token_level_data["rep_atom_list"]
    S_start_atom_list: list[str | None] = token_level_data["S_start_atom_list"]
    S_stop_atom_list: list[str | None] = token_level_data["S_stop_atom_list"]

    planar_atom_list_dict = find_planar_positions(atom_array, mol_info)  # {(chain_iid, res_id): [atom_name, ...]}
    has_planar_sc: list[bool] = []

    xyz_planar: list[list[list[float]]] = []  # list[I] of [K_i, 3]  (K_i varies per residue)
    xyz_S_start: list[list[float]] = []  # list[I] of [3]
    xyz_S_stop: list[list[float]] = []  # list[I] of [3]

    for c, r, S_start_atm, S_stop_atm in zip(
        chain_iid_list,
        resi_list,
        S_start_atom_list,
        S_stop_atom_list,
        strict=False,
    ):
        planar_atoms_i = planar_atom_list_dict[(c, r)]
        has_planar_sc.append(bool(len(planar_atoms_i) >= 4))

        atom_array_i = atom_array[(atom_array.chain_iid == c) & (atom_array.res_id == r)]

        planar_coords_i: list[list[float]] = []
        for pl_atm_name_j in planar_atoms_i:
            pl_atom_array_ij = atom_array_i[atom_array_i.atom_name == pl_atm_name_j]
            if len(pl_atom_array_ij) == 0:
                planar_coords_i.append([float("nan"), float("nan"), float("nan")])
            else:
                planar_coords_i.append(pl_atom_array_ij[0].coord)

        xyz_planar.append(planar_coords_i if len(planar_coords_i) > 3 else [[float("nan")] * 3])

        if S_start_atm is None:
            xyz_S_start.append([float("nan"), float("nan"), float("nan")])
        else:
            S_start_atom_array_i = atom_array_i[atom_array_i.atom_name == S_start_atm]
            xyz_S_start.append(
                [float("nan"), float("nan"), float("nan")]
                if len(S_start_atom_array_i) == 0
                else S_start_atom_array_i[0].coord
            )

        if S_stop_atm is None:
            xyz_S_stop.append([float("nan"), float("nan"), float("nan")])
        else:
            S_stop_atom_array_i = atom_array_i[atom_array_i.atom_name == S_stop_atm]
            xyz_S_stop.append(
                [float("nan"), float("nan"), float("nan")]
                if len(S_stop_atom_array_i) == 0
                else S_stop_atom_array_i[0].coord
            )

        del atom_array_i

    # frame coordinates and backbone direction
    frame_xyz = np.asarray(  # [I, 3]  representative-atom coordinates
        make_coord_list(atom_array, resi_list, chain_iid_list, rep_atom_list),
        dtype=np.float32,
    )

    padded_centers = np.concatenate([frame_xyz[:1], frame_xyz, frame_xyz[-1:]], axis=0)  # [I+2, 3]
    M_i = (
        (  # [I, 3]  smoothed backbone-direction vectors
            (padded_centers[1:-1] - padded_centers[:-2]) + (padded_centers[2:] - padded_centers[1:-1])
        )
        / 2.0
    )

    is_planar_arr = np.asarray(has_planar_sc, dtype=bool)  # [I]
    token_level_data["is_planar"] = is_planar_arr

    is_na_arr = np.asarray(token_level_data["is_na"], dtype=bool)  # [I]
    if NA_only and planar_only:
        filter_mask = is_na_arr & is_planar_arr
    elif NA_only and (not planar_only):
        filter_mask = is_na_arr.copy()
    elif (not NA_only) and planar_only:
        filter_mask = is_planar_arr.copy()
    else:
        filter_mask = np.ones_like(is_na_arr, dtype=bool)
    token_level_data["filter_mask"] = filter_mask  # [I]  bool

    token_level_data.update(
        {
            "xyz_planar": xyz_planar,
            "xyz_S_start": xyz_S_start,
            "xyz_S_stop": xyz_S_stop,
            "frame_xyz": frame_xyz,
            "M_i": M_i,
            "include_geometry": True,
        }
    )

    del planar_atom_list_dict, padded_centers
    return token_level_data


# ---------------------------------------------------------------------------
# Sub-calculations used by compute_nucleic_ss
# ---------------------------------------------------------------------------


def _compute_local_frames(
    xyz_planar: list[np.ndarray],
    planar_centers: np.ndarray,
    M_i: np.ndarray,
    *,
    xyz_S_start: list | None = None,
    xyz_S_stop: list | None = None,
    compute_full_frame: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Build per-residue local coordinate frames from planar sidechain atoms.

    The base-normal direction Z_i is always computed via PCA on the planar
    atom cloud, corrected for backbone direction.  When *compute_full_frame*
    is True the sugar-edge vector is used to derive X_i and Y_i as well.

    Args:
        xyz_planar: Per-residue planar-atom coordinates, list[I] of [K_i, 3].
        planar_centers: Sidechain planar-atom centroids, [I, 3].
        M_i: Backbone-direction vectors, [I, 3].
        xyz_S_start: Sugar-edge start coordinates, list[I] of [3].
            Required when *compute_full_frame* is True.
        xyz_S_stop: Sugar-edge stop coordinates, list[I] of [3].
            Required when *compute_full_frame* is True.
        compute_full_frame: If True, also compute X_i and Y_i.
        eps: Small constant for numerical stability.

    Returns:
        Dict with ``"Z_i"`` (always), and ``"X_i"``, ``"Y_i"`` when
        *compute_full_frame* is True.  Each array has shape ``[I, 3]``.
    """
    n_tokens = len(xyz_planar)

    # Mean-centre the planar atoms per residue
    centered_points = [  # list[I] of [K_i, 3]
        np.asarray(xyz_i, dtype=np.float32) - cen_i for xyz_i, cen_i in zip(xyz_planar, planar_centers, strict=False)
    ]

    # PCA → eigenvectors per residue
    eigenvectors = np.full((n_tokens, 3, 3), np.nan, dtype=np.float32)  # [I, 3, 3]

    for i, xyz_i in enumerate(centered_points):
        xyz_i = xyz_i[~np.isnan(xyz_i).any(axis=1)]
        if xyz_i.shape[0] >= 3:
            cov_matrix = np.einsum("ij,ik->jk", xyz_i, xyz_i) / max(  # [3, 3]
                xyz_i.shape[0] - 1, 1
            )
            _, eigvecs = np.linalg.eigh(cov_matrix)  # [3, 3]
            eigenvectors[i] = eigvecs

    # Base-normal: smallest-eigenvalue direction, corrected for backbone dir
    N_i = eigenvectors[:, :, 0]  # [I, 3]
    N_i = N_i / (np.linalg.norm(N_i, axis=1, keepdims=True) + eps)

    Z_i = N_i * np.sum(M_i * N_i, axis=-1, keepdims=True)  # [I, 3]
    Z_i = Z_i / (np.linalg.norm(Z_i, axis=-1, keepdims=True) + eps)

    result: dict[str, np.ndarray] = {"Z_i": Z_i}

    if compute_full_frame:
        if xyz_S_start is None or xyz_S_stop is None:
            raise ValueError("xyz_S_start and xyz_S_stop are required for full frame")

        X_s_i = (  # [I, 3]  sugar-edge direction
            np.asarray(xyz_S_stop, dtype=np.float32) - np.asarray(xyz_S_start, dtype=np.float32)
        )
        X_s_i = X_s_i / (np.linalg.norm(X_s_i, axis=-1, keepdims=True) + eps)

        X_i = np.cross(Z_i, X_s_i)  # [I, 3]
        X_i = X_i / (np.linalg.norm(X_i, axis=-1, keepdims=True) + eps)
        result["X_i"] = X_i

        Y_i = np.cross(X_i, Z_i)  # [I, 3]
        Y_i = Y_i / (np.linalg.norm(Y_i, axis=-1, keepdims=True) + eps)
        result["Y_i"] = Y_i

    return result


def _compute_pairwise_geometry(
    Z_i: np.ndarray,
    frame_D_ij_vec: np.ndarray,
    sc_D_ij_vec: np.ndarray,
    *,
    X_i: np.ndarray | None = None,
    clamp: bool = True,
    compute_opening: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute pairwise base-step geometry between all residue pairs.

    Derives the pairwise coordinate frame (X_ij, Y_ij, Z_ij) and the
    base-pair geometry parameters: rise (H_ij), buckle (B_ij), propeller
    (P_ij), and optionally opening angle (O_ij).

    Args:
        Z_i: Per-residue base-normal vectors, [I, 3].
        frame_D_ij_vec: Pairwise backbone displacement vectors, [I, I, 3].
        sc_D_ij_vec: Pairwise sidechain-centroid displacement vectors, [I, I, 3].
        X_i: Per-residue local X-axis, [I, 3].  Required when
            *compute_opening* is True.
        clamp: Clamp cosines to [-1, 1] before ``arccos``.
        compute_opening: If True, compute opening angle O_ij.
        eps: Small constant for numerical stability.

    Returns:
        Dict with keys ``"H_ij"`` [I, I], ``"B_ij"`` [I, I],
        ``"P_ij"`` [I, I], ``"base_ori_ij"`` [I, I],
        ``"X_ij"`` [I, I, 3], ``"Y_ij"`` [I, I, 3],
        ``"Z_ij"`` [I, I, 3], and optionally ``"O_ij"`` [I, I].
    """
    # Orientation-selected pairwise Z-axis
    Z_sum = Z_i[:, None, :] + Z_i[None, :, :]  # [I, I, 3]
    Z_diff = Z_i[:, None, :] - Z_i[None, :, :]  # [I, I, 3]
    Z_ij_oris = 0.5 * np.stack((Z_sum, Z_diff), axis=0)  # [2, I, I, 3]

    base_ori_ij = (  # [I, I]  0=parallel, 1=antiparallel
        np.linalg.norm(Z_ij_oris[1], axis=-1) > np.linalg.norm(Z_ij_oris[0], axis=-1)
    ).astype(np.int64)

    Z_ij = np.where(base_ori_ij[..., None] == 0, Z_ij_oris[0], Z_ij_oris[1])  # [I, I, 3]
    Z_ij = Z_ij / (np.linalg.norm(Z_ij, axis=-1, keepdims=True) + eps)

    # Pairwise Y (inter-residue direction) and X axes
    Y_ij = frame_D_ij_vec / (np.linalg.norm(frame_D_ij_vec, axis=-1, keepdims=True) + eps)  # [I, I, 3]
    X_ij = np.cross(Z_ij, Y_ij)  # [I, I, 3]
    X_ij = X_ij / (np.linalg.norm(X_ij, axis=-1, keepdims=True) + eps)

    # Rise (H_ij)
    H_ij = np.sum(sc_D_ij_vec * Z_ij, axis=-1)  # [I, I]

    # Buckle (B_ij)
    proj_Z_i_YZ = (  # [I, I, 3]
        np.sum(Z_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        + np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
    )
    proj_Z_i_YZ_norm = proj_Z_i_YZ / (np.linalg.norm(proj_Z_i_YZ, axis=-1, keepdims=True) + eps)
    cos_buckle = np.sum(proj_Z_i_YZ_norm * (-proj_Z_i_YZ_norm.swapaxes(0, 1)), axis=-1)  # [I, I]

    # Propeller (P_ij)
    proj_Z_i_ZX = (  # [I, I, 3]
        np.sum(Z_i[:, None, :] * Z_ij, axis=-1, keepdims=True) * Z_ij
        + np.sum(Z_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
    )
    proj_Z_i_ZX_norm = proj_Z_i_ZX / (np.linalg.norm(proj_Z_i_ZX, axis=-1, keepdims=True) + eps)
    cos_propeller = np.sum(proj_Z_i_ZX_norm * (-proj_Z_i_ZX_norm.swapaxes(0, 1)), axis=-1)  # [I, I]

    if clamp:
        cos_buckle = np.clip(cos_buckle, -1.0, 1.0)
        cos_propeller = np.clip(cos_propeller, -1.0, 1.0)

    B_ij = np.arccos(cos_buckle)  # [I, I]
    P_ij = np.arccos(cos_propeller)  # [I, I]

    result: dict[str, np.ndarray] = {
        "H_ij": H_ij,
        "B_ij": B_ij,
        "P_ij": P_ij,
        "base_ori_ij": base_ori_ij,
        "X_ij": X_ij,
        "Y_ij": Y_ij,
        "Z_ij": Z_ij,
    }

    # Opening angle (O_ij) — purely diagnostic
    if compute_opening:
        if X_i is None:
            raise ValueError("X_i is required to compute opening angle")

        proj_X_i_XY = (  # [I, I, 3]
            np.sum(X_i[:, None, :] * X_ij, axis=-1, keepdims=True) * X_ij
            + np.sum(X_i[:, None, :] * Y_ij, axis=-1, keepdims=True) * Y_ij
        )
        proj_X_i_XY_norm = proj_X_i_XY / (np.linalg.norm(proj_X_i_XY, axis=-1, keepdims=True) + eps)
        cos_opening = np.sum(proj_X_i_XY_norm * proj_X_i_XY_norm.swapaxes(0, 1), axis=-1)  # [I, I]
        if clamp:
            cos_opening = np.clip(cos_opening, -1.0, 1.0)
        result["O_ij"] = np.arccos(cos_opening)  # [I, I]

    return result


def _compute_basepair_mask(
    hbond_count: np.ndarray,
    seq_neighbors: np.ndarray,
    H_ij: np.ndarray,
    B_ij: np.ndarray,
    P_ij: np.ndarray,
    mol_info: NucMolInfo,
    *,
    bool_only: bool = False,
    eps: float = 1e-8,
) -> dict[str, np.ndarray] | np.ndarray:
    """Identify base pairs by combining H-bond scores with geometry filters.

    Computes a sigmoid-based base-pair probability from weighted H-bond
    counts and gates it with rise / buckle / propeller geometry limits.

    Args:
        hbond_count: H-bond counts, [I, I, 3] (BB-BB / BB-SC / SC-SC).
        seq_neighbors: Sequence-neighbor boolean mask, [I, I].
        H_ij: Rise displacement, [I, I].
        B_ij: Buckle angle (radians), [I, I].
        P_ij: Propeller angle (radians), [I, I].
        mol_info: Molecular-info object with ``bp_summation_weights``,
            ``bp_hbond_coeff``, ``min_hbonds_for_bp``, ``bp_val_cutoff``,
            and ``base_geometry_limits``.
        bool_only: If True, return only the boolean mask array.
        eps: Small constant for numerical stability.

    Returns:
        If *bool_only*: ``np.ndarray`` of shape ``(I, I)`` (bool).
        Otherwise: dict with ``"basepairs_bool_ij"`` [I, I] (bool),
        ``"basepairs_ij"`` [I, I] (float), and
        ``"hbond_summation"`` [I, I] (float).
    """
    hbond_summation = np.tensordot(  # [I, I]
        hbond_count.astype(np.float32),
        np.asarray(mol_info.bp_summation_weights, dtype=np.float32),
        axes=([2], [0]),
    )

    logits = mol_info.bp_hbond_coeff * (  # [I, I]
        hbond_summation - (mol_info.min_hbonds_for_bp - 1)
    )
    bp_preds = (1.0 / (1.0 + np.exp(-logits))) + eps  # [I, I]

    # Geometry filters
    H_ij_filter = (  # [I, I]
        (H_ij >= -mol_info.base_geometry_limits["H_ij"]) & (H_ij <= mol_info.base_geometry_limits["H_ij"])
    )
    B_ij_filter = (  # [I, I]
        (B_ij <= mol_info.base_geometry_limits["B_ij"]) | (B_ij >= math.pi - mol_info.base_geometry_limits["B_ij"])
    )
    P_ij_filter = (  # [I, I]
        (P_ij <= mol_info.base_geometry_limits["P_ij"]) | (P_ij >= math.pi - mol_info.base_geometry_limits["P_ij"])
    )
    bp_geom_filter = H_ij_filter & B_ij_filter & P_ij_filter  # [I, I]

    if bool_only:
        basepairs_bool_ij = (  # [I, I]
            (~seq_neighbors) & bp_geom_filter & (bp_preds >= float(mol_info.bp_val_cutoff))
        )
        return basepairs_bool_ij

    basepairs_ij = (  # [I, I]
        (~seq_neighbors).astype(np.float32) * bp_geom_filter.astype(np.float32) * bp_preds.astype(np.float32)
    )
    basepairs_bool_ij = basepairs_ij >= mol_info.bp_val_cutoff  # [I, I]

    return {
        "basepairs_bool_ij": basepairs_bool_ij,
        "basepairs_ij": basepairs_ij,
        "hbond_summation": hbond_summation,
    }


def compute_nucleic_ss(
    mol_info: NucMolInfo,
    token_level_data: dict,
    hbond_count: np.ndarray,
    clamp_pairwise_params: bool = True,
    eps: float = 1e-8,
    *,
    return_local_params: bool = False,
    return_pairwise_geometry: bool = False,
    return_opening_angle: bool = False,
    return_basepairs_only: bool = False,
) -> dict[str, np.ndarray]:
    """Compute nucleic-acid pairwise base-pair geometry and filters.

    Operates in two modes:

    * **Fast annotation** (default / ``return_basepairs_only=True``): returns
      only ``basepairs_bool_ij`` and frees intermediate arrays.
    * **Diagnostic**: additionally returns local/pairwise geometry when
      ``return_pairwise_geometry``, ``return_local_params``, or
      ``return_opening_angle`` are set.

    Args:
        mol_info: Molecular-info constants (geometry limits, H-bond weights).
        token_level_data: Token-level dict with geometry (from
            :func:`add_token_level_geometry_data`).
        hbond_count: H-bond count array, shape ``(I_full, I_full, 3)``.
        clamp_pairwise_params: Clamp cosines to [-1, 1] before ``arccos``.
        eps: Small constant for numerical stability.
        return_local_params: Return per-residue X/Y/Z local frames.
        return_pairwise_geometry: Return pairwise X_ij/Y_ij/Z_ij arrays.
        return_opening_angle: Return pairwise opening angle O_ij.
        return_basepairs_only: Return only the boolean base-pair mask
            (fastest path).

    Returns:
        If ``return_basepairs_only``: ``np.ndarray`` of shape ``(I, I)``
        (bool) — the base-pair boolean mask.

        Otherwise: dict ``{"pair_params": {...}, "local_params": {...}}``
        containing the requested geometry arrays (all shape ``(I, I)`` or
        ``(I, 3)``).
    """

    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)  # [I_full]

    # --- Unpack and filter token-level data ----------------------
    M_i = np.asarray(token_level_data["M_i"], dtype=np.float32)[mask_1d]  # [I, 3]
    frame_xyz = np.asarray(token_level_data["frame_xyz"], dtype=np.float32)[mask_1d]  # [I, 3]
    xyz_S_start = [v for v, k in zip(token_level_data["xyz_S_start"], mask_1d, strict=False) if k]  # list[I] of [3]
    xyz_S_stop = [v for v, k in zip(token_level_data["xyz_S_stop"], mask_1d, strict=False) if k]  # list[I] of [3]
    xyz_planar = [v for v, k in zip(token_level_data["xyz_planar"], mask_1d, strict=False) if k]  # list[I] of [K_i, 3]

    hbond_count = np.asarray(hbond_count)[mask_1d, :][:, mask_1d]  # [I, I, 3]
    seq_neighbors = np.asarray(token_level_data["seq_neighbors"], dtype=bool)[mask_1d, :][:, mask_1d]  # [I, I]

    if len(xyz_planar) == 0:
        empty_bool = np.zeros((0, 0), dtype=bool)
        if return_basepairs_only:
            return empty_bool

        empty_float = np.zeros((0, 0), dtype=np.float32)
        empty_vec = np.zeros((0, 0, 3), dtype=np.float32)
        nucleic_ss_data: dict[str, dict[str, np.ndarray]] = {
            "pair_params": {
                "H_ij": empty_float,
                "B_ij": empty_float,
                "P_ij": empty_float,
                "base_ori_ij": np.zeros((0, 0), dtype=np.int64),
                "basepairs_bool_ij": empty_bool,
                "basepairs_ij": empty_float,
                "hbond_summation": empty_float,
            }
        }
        if return_pairwise_geometry:
            nucleic_ss_data["pair_params"]["X_ij"] = empty_vec
            nucleic_ss_data["pair_params"]["Y_ij"] = empty_vec
            nucleic_ss_data["pair_params"]["Z_ij"] = empty_vec
        if return_opening_angle:
            nucleic_ss_data["pair_params"]["O_ij"] = empty_float
        if return_local_params:
            nucleic_ss_data["local_params"] = {
                "X_i": np.zeros((0, 3), dtype=np.float32),
                "Y_i": np.zeros((0, 3), dtype=np.float32),
                "Z_i": np.zeros((0, 3), dtype=np.float32),
            }
        return nucleic_ss_data

    # --- Precompute centroids and displacement vectors -----------
    planar_centers = np.stack(  # [I, 3]
        [np.nanmean(np.asarray(xyz_i, dtype=np.float32), axis=0) for xyz_i in xyz_planar],
        axis=0,
    ).astype(np.float32)

    frame_D_ij_vec = frame_xyz[None, :, :] - frame_xyz[:, None, :]  # [I, I, 3]
    sc_D_ij_vec = planar_centers[None, :, :] - planar_centers[:, None, :]  # [I, I, 3]

    # --- CALC I: per-residue local coordinate frames -------------
    need_full_frame = return_local_params or return_opening_angle
    local_frames = _compute_local_frames(
        xyz_planar,
        planar_centers,
        M_i,
        xyz_S_start=xyz_S_start if need_full_frame else None,
        xyz_S_stop=xyz_S_stop if need_full_frame else None,
        compute_full_frame=need_full_frame,
        eps=eps,
    )
    Z_i = local_frames["Z_i"]  # [I, 3]
    X_i = local_frames.get("X_i")  # [I, 3] or None

    # --- CALC II: pairwise base-step geometry --------------------
    pw_geom = _compute_pairwise_geometry(
        Z_i,
        frame_D_ij_vec,
        sc_D_ij_vec,
        X_i=X_i,
        clamp=clamp_pairwise_params,
        compute_opening=return_opening_angle,
        eps=eps,
    )

    # --- CALC III: base-pair identification ----------------------
    bp_result = _compute_basepair_mask(
        hbond_count,
        seq_neighbors,
        pw_geom["H_ij"],
        pw_geom["B_ij"],
        pw_geom["P_ij"],
        mol_info,
        bool_only=return_basepairs_only,
        eps=eps,
    )

    if return_basepairs_only:
        return bp_result  # np.ndarray [I, I] bool

    # --- Assemble output dict ------------------------------------
    assert isinstance(bp_result, dict)

    pair_params: dict[str, np.ndarray] = {
        "H_ij": pw_geom["H_ij"],
        "B_ij": pw_geom["B_ij"],
        "P_ij": pw_geom["P_ij"],
        "base_ori_ij": pw_geom["base_ori_ij"],
        "basepairs_bool_ij": bp_result["basepairs_bool_ij"],
        "basepairs_ij": bp_result["basepairs_ij"],
        "hbond_summation": bp_result["hbond_summation"],
    }

    if return_opening_angle and "O_ij" in pw_geom:
        pair_params["O_ij"] = pw_geom["O_ij"]

    if return_pairwise_geometry:
        pair_params["X_ij"] = pw_geom["X_ij"]
        pair_params["Y_ij"] = pw_geom["Y_ij"]
        pair_params["Z_ij"] = pw_geom["Z_ij"]

    nucleic_ss_data: dict = {"pair_params": pair_params}
    if return_local_params and "Y_i" in local_frames:
        nucleic_ss_data["local_params"] = {
            "X_i": local_frames["X_i"],
            "Y_i": local_frames["Y_i"],
            "Z_i": local_frames["Z_i"],
        }

    return nucleic_ss_data


def annotate_na_ss(
    atom_array: AtomArray,
    *,
    NA_only: bool = False,
    planar_only: bool = True,
    p_canonical_bp_filter: float = 0.0,
    mol_info: NucMolInfo | None = None,
    token_level_data: dict | None = None,
) -> AtomArray:
    """Compute base pairs and write canonical NA-SS conditions onto *atom_array*.

    Uses H-bond counts and pairwise geometry filters to identify base pairs,
    then stores the result in the canonical condition representation:

    * ``C_BP`` — sparse pair edges
    * ``C_NA_SS_STATE`` — per-token ``BasePairEnum`` value

    Args:
        atom_array: Structure to annotate (modified in-place).
        NA_only: Restrict geometry filter to nucleic-acid tokens.
        planar_only: Restrict geometry filter to tokens with planar
            sidechains.
        p_canonical_bp_filter: Probability of discarding non-canonical
            base pairs (keeps only A-U, A-T, G-C).
        mol_info: Molecular-info constants; created if ``None``.
        token_level_data: Pre-computed metadata dict; augmented with
            geometry as needed.

    Returns:
        The same *atom_array* with canonical NA-SS conditions set.
    """

    if mol_info is None:
        mol_info = NucMolInfo()
    from .conditions import BasePairEnum

    # Residue representatives (0..L-1) and their corresponding atom indices.
    # Keep this aligned with get_token_level_metadata(), which uses residue starts.
    if token_level_data is not None and "token_starts" in token_level_data:
        token_starts = np.asarray(token_level_data["token_starts"], dtype=int)
    else:
        token_starts = struc.get_residue_starts(atom_array)

    token_res_names: list[str] = [str(rn) for rn in np.asarray(atom_array.res_name)[token_starts].tolist()]

    # Compute basepairs on the token graph (respecting NA_only/planar_only filtering)
    if token_level_data is None:
        token_level_data = get_token_level_metadata(
            atom_array,
            mol_info,
            NA_only=NA_only,
            planar_only=planar_only,
        )
    token_level_data = add_token_level_geometry_data(
        atom_array,
        mol_info,
        token_level_data,
        NA_only=NA_only,
        planar_only=planar_only,
    )
    # Note: this mask gives positions that are *chemically valid* for forming
    # base pairs, which is different from custom mask-generation for features
    mask_1d = np.asarray(token_level_data["filter_mask"], dtype=bool)

    subset_idxs = np.nonzero(mask_1d)[0]

    is_na_full = np.asarray(token_level_data["is_na"], dtype=bool)

    hbond_count = calculate_hb_counts(atom_array, token_level_data)
    bp_bool = np.asarray(
        compute_nucleic_ss(
            mol_info,
            token_level_data,
            hbond_count,
            clamp_pairwise_params=True,
            eps=1e-8,
            return_local_params=False,
            return_pairwise_geometry=False,
            return_opening_angle=False,
            return_basepairs_only=True,
        ),
        dtype=bool,
    )

    # Optional: filter to canonical Watson-Crick basepairs only.
    # Sampled probabilistically to allow mixed supervision during training.
    do_canonical_filter = bool(p_canonical_bp_filter and (np.random.rand() < float(p_canonical_bp_filter)))
    if do_canonical_filter:

        def _base_letter(res_name: str) -> str | None:
            rn = str(res_name).strip().upper()
            if rn in STANDARD_RNA:
                return rn
            if rn in STANDARD_DNA:
                return rn[1]  # DA/DC/DG/DT -> A/C/G/T
            return None

        allowed_pairs = {
            ("A", "U"),
            ("U", "A"),
            ("A", "T"),
            ("T", "A"),
            ("G", "C"),
            ("C", "G"),
        }
        base_letters_full: list[str | None] = [_base_letter(rn) for rn in token_res_names]

        bp_bool = np.asarray(bp_bool, dtype=bool)
        bp_rows_tmp, bp_cols_tmp = np.nonzero(bp_bool)
        for r, c in zip(bp_rows_tmp.tolist(), bp_cols_tmp.tolist(), strict=False):
            full_i = int(subset_idxs[int(r)])
            full_j = int(subset_idxs[int(c)])
            bi = base_letters_full[full_i]
            bj = base_letters_full[full_j]
            if bi is None or bj is None or (bi, bj) not in allowed_pairs:
                bp_bool[int(r), int(c)] = False
                bp_bool[int(c), int(r)] = False

    bp_bool = np.asarray(bp_bool, dtype=bool)
    bp_rows, bp_cols = np.nonzero(bp_bool)

    n_tokens = len(token_starts)
    state_values = np.full((n_tokens,), BasePairEnum.UNSPECIFIED.value, dtype=np.int16)

    # Only nucleic-acid tokens inside the chemically valid subset participate in
    # NA-SS annotation. Unpaired eligible tokens are explicit loops.
    for full_i in subset_idxs.tolist():
        if not bool(is_na_full[int(full_i)]):
            continue
        state_values[int(full_i)] = BasePairEnum.LOOP.value

    pair_inds: set[tuple[int, int]] = set()

    # We only process each unordered pair once to avoid duplicates.
    for r, c in zip(bp_rows.tolist(), bp_cols.tolist(), strict=False):
        if r == c:
            continue

        full_i = int(subset_idxs[int(r)])
        full_j = int(subset_idxs[int(c)])
        if full_i == full_j:
            continue

        # Only annotate NA-NA basepairs as nucleic secondary structure.
        if (not bool(is_na_full[int(full_i)])) or (not bool(is_na_full[int(full_j)])):
            continue

        # Enforce uniqueness: only handle (i,j) where i < j
        if full_j < full_i:
            continue

        state_values[full_i] = BasePairEnum.PAIR.value
        state_values[full_j] = BasePairEnum.PAIR.value
        pair_inds.add((full_i, full_j))
        pair_inds.add((full_j, full_i))

    return _set_canonical_na_ss_from_token_state(
        atom_array,
        token_starts=np.asarray(token_starts, dtype=np.int32),
        state_values=state_values,
        pair_inds=np.asarray(sorted(pair_inds), dtype=np.int32).reshape(-1, 2),
    )


def annotate_na_ss_from_specification(
    atom_array: AtomArray,
    specification: dict,
) -> AtomArray:
    """Write canonical NA-SS conditions from a parsed inference payload.

    Args:
        atom_array: Structure to annotate (modified in-place).
        specification: Specification dict containing the syntax-agnostic
            ``extra.parsed_nucleic_ss`` payload produced during inference
            input construction.

    Returns:
        The same *atom_array* with canonical NA-SS conditions set.
    """
    token_starts = get_token_starts(atom_array)
    spec = specification or {}
    parsed_nucleic_ss = spec.get("parsed_nucleic_ss")
    if parsed_nucleic_ss is None and isinstance(spec.get("extra"), dict):
        parsed_nucleic_ss = spec["extra"].get("parsed_nucleic_ss")
    if not parsed_nucleic_ss:
        return atom_array
    _set_canonical_na_ss_from_token_payload(
        atom_array,
        token_starts=np.asarray(token_starts, dtype=int),
        parsed_nucleic_ss=parsed_nucleic_ss,
    )
    return atom_array


def annotate_na_ss_from_data_specification(
    data: dict,
) -> AtomArray:
    """Annotate canonical NA-SS conditions from ``data["specification"]``.

    Convenience wrapper around :func:`annotate_na_ss_from_specification`.

    Args:
        data: Pipeline data dict containing ``atom_array`` and optionally
            ``specification``.

    Returns:
        The annotated AtomArray (also stored back in *data*).
    """
    atom_array = data["atom_array"]
    spec = data.get("specification", {}) or {}
    return annotate_na_ss_from_specification(atom_array, spec)
