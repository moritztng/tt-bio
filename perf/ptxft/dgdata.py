"""Real structures -> Protenix distogram targets, and the bin grid verified not assumed.

The objective is a distogram cross-entropy against real structures, so the target has to
sit on the SAME bin grid the head predicts on. Get that wrong and the loss is wrong while
still going down, which is the most convincing way this row could fail.

The grid is the AlphaFold/OpenFold one, `pts_to_distogram` in
tt_bio/_vendor/openfold3/core/utils/tensor_utils.py:49: 64 bins whose 63 boundaries are
`linspace(2.3125, 21.6875, 63)` on the CB-CB distance. It is taken from there rather than
from a paper, and `--calibrate` then checks it against the model itself: if the grid is
right, the base model's argmax bin centre tracks the true CB distance; if it is wrong, the
relation is offset or scaled and the check says so.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request

import numpy as np

MIN_BIN, MAX_BIN, N_BINS = 2.3125, 21.6875, 64
BOUNDARIES = np.linspace(MIN_BIN, MAX_BIN, N_BINS - 1)
# Bin centres, with the two open-ended bins given the boundary itself. Used only for the
# calibration readout, never in the loss.
CENTRES = np.concatenate([[BOUNDARIES[0]],
                          0.5 * (BOUNDARIES[1:] + BOUNDARIES[:-1]),
                          [BOUNDARIES[-1]]])

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fetch_cif(pdb_id: str, cache_dir: str) -> str:
    """Download an mmCIF from RCSB once and cache it. No auth, no key."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as fh:
            fh.write(data)
    return path


def parse_chain(cif_path: str, chain: str | None = None):
    """(sequence, CB coords, mask) for one polypeptide chain of an mmCIF.

    CB rather than CA because that is what the AlphaFold-family distogram is defined on,
    with glycine falling back to CA since it has no CB. Only the FIRST model and only
    ATOM records are read; altloc keeps the first. Written against the atom_site loop
    directly rather than through a parser, because the only fields needed are six and a
    dependency that reads the whole file is a dependency that can disagree with itself
    across versions.
    """
    cols, rows, in_loop, header = [], [], False, False
    opener = gzip.open if cif_path.endswith(".gz") else open
    with opener(cif_path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                in_loop, header = True, True
                cols.append(line.strip().split(".", 1)[1])
                continue
            if header and not line.startswith("_atom_site."):
                header = False
            if in_loop:
                if line.startswith("#") or line.startswith("loop_") or not line.strip():
                    break
                rows.append(line.split())
    if not rows:
        raise ValueError(f"no atom_site loop in {cif_path}")
    idx = {c: i for i, c in enumerate(cols)}
    need = ["group_PDB", "label_atom_id", "label_comp_id", "label_asym_id",
            "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z"]
    for n in need:
        if n not in idx:
            raise ValueError(f"{cif_path} atom_site loop has no {n}")
    model_col = idx.get("pdbx_PDB_model_num")

    res = {}
    for r in rows:
        if len(r) < len(cols) or r[idx["group_PDB"]] != "ATOM":
            continue
        if model_col is not None and r[model_col] != "1":
            continue
        asym = r[idx["label_asym_id"]]
        if chain is not None and asym != chain:
            continue
        comp = r[idx["label_comp_id"]]
        if comp not in THREE_TO_ONE:
            continue
        seq_id = r[idx["label_seq_id"]]
        if seq_id in (".", "?"):
            continue
        atom = r[idx["label_atom_id"]].strip('"')
        key = (asym, int(seq_id))
        slot = res.setdefault(key, {"comp": comp})
        if atom in ("CB", "CA"):
            slot.setdefault(atom, (float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]),
                                   float(r[idx["Cartn_z"]])))
    if not res:
        raise ValueError(f"no standard residues in {cif_path}"
                         + (f" chain {chain}" if chain else ""))
    # One chain only: the longest, so a crystal with several copies is unambiguous.
    by_chain = {}
    for (asym, sid) in res:
        by_chain.setdefault(asym, []).append(sid)
    pick = chain or max(by_chain, key=lambda c: len(by_chain[c]))
    sids = sorted(by_chain[pick])
    seq, xyz, mask = [], [], []
    for sid in sids:
        slot = res[(pick, sid)]
        seq.append(THREE_TO_ONE[slot["comp"]])
        pt = slot.get("CB") or slot.get("CA")
        xyz.append(pt if pt else (0.0, 0.0, 0.0))
        mask.append(pt is not None)
    return "".join(seq), np.asarray(xyz, np.float64), np.asarray(mask, bool)


def distogram_target(xyz: np.ndarray, mask: np.ndarray):
    """(bin index [N,N] int64, pair mask [N,N] bool) on the head's own 64-bin grid."""
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    b = np.searchsorted(BOUNDARIES, d, side="right").astype(np.int64)
    pm = mask[:, None] & mask[None, :]
    return b, pm


def cross_entropy(logits: np.ndarray, target: np.ndarray, pair_mask: np.ndarray):
    """Mean CE over masked pairs, in float64 on host, plus the analytic seed.

    The loss VALUE is computed on host in float64 from the device logits so a bf16 scalar
    reduction cannot swamp a small improvement, which is the whole quantity being
    measured. The seed handed back to the tape is the exact analytic gradient
    (softmax(x) - onehot)/M, so nothing differentiates an approximation of the loss.
    """
    x = logits.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    n, _ = target.shape
    ii, jj = np.nonzero(pair_mask)
    m = ii.size
    if m == 0:
        raise ValueError("the pair mask is empty; no residue has a CB")
    loss = float(-np.log(np.clip(p[ii, jj, target[ii, jj]], 1e-300, None)).sum() / m)
    seed = p.copy()
    seed[ii, jj, target[ii, jj]] -= 1.0
    seed *= pair_mask[..., None] / m
    return loss, seed, p


def argmax_distance(logits: np.ndarray) -> np.ndarray:
    """The predicted distance implied by the argmax bin, for --calibrate only."""
    return CENTRES[logits.argmax(-1)]
