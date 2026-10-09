"""Rebuild a chemically invalid terminal OXT before a structure is written.

Port of OpenDDE ``_repair_terminal_oxt_coordinates`` (opendde/data/utils.py, commit 6685cef,
with the 1.1.1 external-bond check on C, O and OXT). The diffusion head places OXT like any
other atom and can leave it on top of O or C, or detached from the residue. When the
predicted C/CA/O anchor is sound, OXT is placed where the CCD reference puts it relative to
that anchor; otherwise the prediction is written as it came out.

Upstream rotates the residue's ``ref_pos``. tt-bio's ``ref_pos`` OXT is the mirror ``2C - O``
(protein_atom_features), a 180 degree O-C-OXT, so the rebuild reads the CCD ideal geometry in
``_CCD_IDEAL`` instead.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MIN_TERMINAL_C_OXT_DISTANCE = 1.0
_MAX_TERMINAL_C_OXT_DISTANCE = 1.7
_MIN_TERMINAL_C_CA_DISTANCE = 1.3
_MAX_TERMINAL_C_CA_DISTANCE = 1.7
_MIN_TERMINAL_C_O_DISTANCE = 1.0
_MAX_TERMINAL_C_O_DISTANCE = 1.5
_MIN_TERMINAL_O_OXT_DISTANCE = 1.8

# C, CA, O, OXT per residue: _chem_comp_atom.pdbx_model_Cartn_{x,y,z}_ideal of the PDB
# chemical component dictionary (components.cif as cached by OpenDDE, read with gemmi). The
# C/CA/O rows equal tt_bio/data/protein_ref_conformers.json, which carries the same ideal
# coordinates without OXT.
_CCD_IDEAL = {
    "ALA": [[-0.094, 0.017, -0.716], [0.257, 0.418, 0.692], [-1.056, -0.682, -0.923], [0.661, 0.439, -1.742]],
    "ARG": [[-0.907, 2.521, -2.901], [0.004, 2.294, -1.708], [-1.827, 1.789, -3.242], [-0.588, 3.659, -3.574]],
    "ASN": [[-1.846, -0.179, -0.031], [-0.448, 0.292, -0.34], [-2.51, 0.402, 0.794], [-2.353, -1.243, -0.673]],
    "ASP": [[-1.868, -0.18, -0.029], [-0.47, 0.286, -0.344], [-2.534, 0.415, 0.786], [-2.374, -1.256, -0.652]],
    "CYS": [[-0.095, 0.006, 1.606], [0.141, 0.45, 0.186], [0.685, -0.742, 2.143], [-1.174, 0.443, 2.275]],
    "GLN": [[-0.236, 0.022, 2.344], [0.517, 0.451, 1.112], [-0.005, -1.049, 2.851], [-1.165, 0.831, 2.878]],
    "GLU": [[2.364, -0.26, 0.041], [1.138, 0.515, 0.453], [3.01, 0.096, -0.916], [2.737, -1.345, 0.737]],
    "GLY": [[-0.498, 0.029, -0.005], [0.761, -0.799, -0.008], [-0.429, 1.235, -0.023], [-1.697, -0.574, 0.018]],
    "HIS": [[1.083, -3.207, 0.905], [1.172, -1.709, 0.652], [0.04, -3.77, 1.222], [2.247, -3.882, 0.744]],
    "ILE": [[0.066, -0.032, -1.657], [-0.487, 0.519, -0.369], [-0.484, -0.958, -2.203], [1.171, 0.504, -2.197]],
    "LEU": [[0.18, -0.055, -1.836], [-0.205, 0.441, -0.467], [-0.591, -0.731, -2.474], [1.382, 0.254, -2.348]],
    "LYS": [[2.657, -0.284, -0.032], [1.394, 0.355, 0.484], [3.316, 0.275, -0.876], [3.05, -1.476, 0.446]],
    "MET": [[0.206, 0.002, -2.504], [-0.392, 0.499, -1.214], [-0.236, -0.989, -3.033], [1.232, 0.661, -3.066]],
    "PHE": [[-0.109, 0.047, 2.756], [-0.02, 0.426, 1.3], [0.879, -0.317, 3.346], [-1.286, 0.113, 3.396]],
    "PRO": [[1.408, 0.091, 0.005], [0.001, -0.107, 0.509], [1.65, 0.98, -0.777], [2.391, -0.721, 0.424]],
    "SER": [[-0.053, 0.004, 1.173], [0.1, 0.469, -0.252], [0.751, -0.76, 1.649], [-1.084, 0.44, 1.913]],
    "THR": [[-0.038, -0.09, -1.309], [0.122, -0.706, 0.056], [0.732, 0.761, -1.683], [-1.039, -0.488, -2.11]],
    "TRP": [[-0.49, 0.076, 3.357], [-0.008, 0.417, 1.97], [0.308, -0.13, 4.24], [-1.806, 0.001, 3.61]],
    "TYR": [[-0.103, 0.094, 3.201], [-0.018, 0.429, 1.734], [0.886, -0.254, 3.799], [-1.279, 0.184, 3.842]],
    "VAL": [[-0.037, -0.093, -1.288], [0.145, -0.698, 0.079], [0.703, 0.784, -1.664], [-1.022, -0.529, -2.089]],
    "UNK": [[1.576, -0.429, -3.719], [2.145, -1.162, -2.517], [0.85, 0.554, -3.635], [1.959, -0.964, -4.903]],
}
# float32, as upstream's ref_pos annotation
CCD_TERMINAL_REF = {k: np.asarray(v, dtype=np.float32) for k, v in _CCD_IDEAL.items()}


def _local_coordinate_frame(carbon, alpha_carbon, oxygen):
    """Orthonormal frame (columns: C->CA, in-plane O, normal) of one carbonyl, or None."""
    ca_axis = alpha_carbon - carbon
    ca_norm = np.linalg.norm(ca_axis)
    if not np.isfinite(ca_norm) or ca_norm < 1e-6:
        return None
    ca_axis = ca_axis / ca_norm

    oxygen_axis = oxygen - carbon
    oxygen_axis -= np.dot(oxygen_axis, ca_axis) * ca_axis
    oxygen_norm = np.linalg.norm(oxygen_axis)
    if not np.isfinite(oxygen_norm) or oxygen_norm < 1e-6:
        return None
    oxygen_axis = oxygen_axis / oxygen_norm

    normal_axis = np.cross(ca_axis, oxygen_axis)
    return np.column_stack((ca_axis, oxygen_axis, normal_axis))


def repair_terminal_oxt(coord, atom_name, res_name, chain, res_id, is_protein, bonded=None,
                        ref=None):
    """Return ``(coords, n_repaired)``: a float32 copy of ``coord`` (N, 3) with the OXT of
    each protein chain's last residue rebuilt where it fails the upstream gates.

    Per-atom arrays: ``atom_name``, ``res_name``, ``chain`` (any hashable chain key),
    ``res_id``, ``is_protein`` (bool). ``bonded`` (bool, optional) marks atoms with a
    covalent bond leaving their residue; a residue whose C, O or OXT is so bonded is left
    alone. ``ref`` maps res_name -> (4, 3) C, CA, O, OXT reference coordinates (default
    ``CCD_TERMINAL_REF``); a residue it does not name is left alone. ``coord`` is not
    modified.
    """
    out = np.array(coord, dtype=np.float32, copy=True)
    ref = CCD_TERMINAL_REF if ref is None else ref
    atom_name = np.asarray(atom_name); res_name = np.asarray(res_name)
    chain = np.asarray(chain); res_id = np.asarray(res_id)
    is_protein = np.asarray(is_protein, dtype=bool)
    repaired = 0

    for chain_id in np.unique(chain[is_protein]):
        chain_mask = (chain == chain_id) & is_protein
        terminal_res_id = np.max(res_id[chain_mask])
        terminal_mask = chain_mask & (res_id == terminal_res_id)

        matches = {n: np.flatnonzero(terminal_mask & (atom_name == n))
                   for n in ("CA", "C", "O", "OXT")}
        if any(len(ix) != 1 for ix in matches.values()):
            continue
        idx = {n: ix[0] for n, ix in matches.items()}
        if bonded is not None and any(bonded[idx[n]] for n in ("C", "O", "OXT")):
            continue
        ref_coordinates = ref.get(str(res_name[idx["C"]]))
        if ref_coordinates is None or not np.all(np.isfinite(ref_coordinates)):
            continue

        required = np.array([idx["C"], idx["CA"], idx["O"], idx["OXT"]])
        pred_coordinates = out[required]
        if not np.all(np.isfinite(pred_coordinates[:3])):
            logger.warning("Skipped terminal OXT repair for chain %s residue %s because the "
                           "predicted C/CA/O coordinates are not finite", chain_id,
                           terminal_res_id)
            continue

        c_oxt_distance = np.linalg.norm(pred_coordinates[3] - pred_coordinates[0])
        o_oxt_distance = np.linalg.norm(pred_coordinates[3] - pred_coordinates[2])
        if (np.isfinite(c_oxt_distance)
                and _MIN_TERMINAL_C_OXT_DISTANCE <= c_oxt_distance <= _MAX_TERMINAL_C_OXT_DISTANCE
                and np.isfinite(o_oxt_distance)
                and o_oxt_distance >= _MIN_TERMINAL_O_OXT_DISTANCE):
            continue

        c_ca_distance = np.linalg.norm(pred_coordinates[1] - pred_coordinates[0])
        c_o_distance = np.linalg.norm(pred_coordinates[2] - pred_coordinates[0])
        if not (_MIN_TERMINAL_C_CA_DISTANCE <= c_ca_distance <= _MAX_TERMINAL_C_CA_DISTANCE
                and _MIN_TERMINAL_C_O_DISTANCE <= c_o_distance <= _MAX_TERMINAL_C_O_DISTANCE):
            logger.warning("Skipped terminal OXT repair for chain %s residue %s because the "
                           "predicted C-CA/C-O distances are invalid (%.3f/%.3f A)", chain_id,
                           terminal_res_id, c_ca_distance, c_o_distance)
            continue

        pred_frame = _local_coordinate_frame(*pred_coordinates[:3])
        ref_frame = _local_coordinate_frame(*ref_coordinates[:3])
        if pred_frame is None or ref_frame is None:
            continue

        ref_c_oxt = ref_coordinates[3] - ref_coordinates[0]
        ref_c_oxt_distance = np.linalg.norm(ref_c_oxt)
        ref_o_oxt_distance = np.linalg.norm(ref_coordinates[3] - ref_coordinates[2])
        if not (_MIN_TERMINAL_C_OXT_DISTANCE <= ref_c_oxt_distance <= _MAX_TERMINAL_C_OXT_DISTANCE
                and ref_o_oxt_distance >= _MIN_TERMINAL_O_OXT_DISTANCE):
            continue

        rotation = pred_frame @ ref_frame.T
        out[idx["OXT"]] = pred_coordinates[0] + rotation @ ref_c_oxt
        repaired += 1

    return out, repaired


def repair_terminal_oxt_from_feats(coord, feats, atom_name, res_name, where=None):
    """``repair_terminal_oxt`` on a Protenix-family feature dict: chain from ``asym_id``,
    residue from ``residue_index``, protein from ``mol_type`` (a dict without it is returned
    unchanged). ``token_bonds`` resolves only to tokens, so an atom counts as externally bonded
    when its token is bonded to a token of another residue; for a standard residue (one token)
    that is any bond on the residue. ``where`` names the output in the log line."""
    from tt_bio.protenix_data import MOL_TYPE_IDS

    out = np.array(coord, dtype=np.float32, copy=True)
    if "mol_type" not in feats:
        return out, 0
    a2t = feats["atom_to_token_idx"].cpu().numpy()
    asym = feats["asym_id"].cpu().numpy(); resid = feats["residue_index"].cpu().numpy()
    bonded = None
    tb = feats.get("token_bonds")
    if tb is not None:
        tb = tb.cpu().numpy() > 0
        other_res = (asym[:, None] != asym[None, :]) | (resid[:, None] != resid[None, :])
        bonded = (tb & other_res).any(1)[a2t]
    out, repaired = repair_terminal_oxt(out, atom_name, res_name, asym[a2t], resid[a2t],
                                        feats["mol_type"].cpu().numpy()[a2t] == MOL_TYPE_IDS["protein"],
                                        bonded=bonded)
    if repaired:
        logger.warning("Rebuilt %d invalid terminal OXT coordinate(s) before saving %s",
                       repaired, where)
    return out, repaired
