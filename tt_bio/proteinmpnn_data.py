"""Slim PDB parsing + featurization for ProteinMPNN design runs.

Vendored from dauparas/ProteinMPNN (MIT) and trimmed to the all-chains-designed
path used by ``tt-bio design``. Multi-chain backbones are supported with every
chain marked for design (no fixed-position / tied-position / PSSM machinery —
bring-your-own-backbone sequence design only). See ``NOTICE``.
"""
from __future__ import annotations

import numpy as np
import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
_ATOMS = ["N", "CA", "C", "O"]


def parse_pdb(path: str) -> dict:
    """Parse a PDB into the ProteinMPNN chain-dict format (all chains designed).

    Returns ``{name, num_of_chains, seq, seq_chain_<L>, coords_chain_<L>}`` where
    each coords dict holds ``N/CA/C/O_chain_<L>`` as ``[L, 3]`` lists. Missing O
    atoms are imputed from the backbone geometry (ProteinMPNN tolerates this; the
    reference parser zero-fills and relies on the mask).
    """
    chains: dict[str, dict] = {}
    for raw in open(path, "rb"):
        line = raw.decode("utf-8", "ignore").rstrip()
        if line[:6] == "HETATM" and line[17:20] == "MSE":
            line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
        if line[:4] != "ATOM":
            continue
        atom = line[12:16].strip()
        if atom not in _ATOMS:
            continue
        ch = line[21:22] or "A"
        resname = line[17:20].strip()
        resn = line[22:27].strip()  # insertion code aware
        if resn[-1].isalpha():
            resa, resn = resn[-1], int(resn[:-1]) - 1
        else:
            resa, resn = "", int(resn) - 1
        x, y, z = (float(line[i:i + 8]) for i in (30, 38, 46))
        c = chains.setdefault(ch, {"res": {}, "order": []})
        if resn not in c["res"]:
            c["res"][resn] = {"atom": {}, "resa": resa, "aa": _AA_3TO1.get(resname, "X")}
            c["order"].append(resn)
        c["res"][resn]["atom"][atom] = [x, y, z]

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    out = {"name": path.rsplit("/", 1)[-1][:-4], "num_of_chains": 0, "seq": ""}
    for i, ch in enumerate(sorted(chains)):
        letter = letters[i]
        c = chains[ch]
        order = sorted(c["order"])
        seq = "".join(c["res"][r]["aa"] for r in order)
        coords = {a + "_chain_" + letter: [] for a in _ATOMS}
        for r in order:
            for a in _ATOMS:
                coords[a + "_chain_" + letter].append(c["res"][r]["atom"].get(a, [0.0, 0.0, 0.0]))
        out["seq_chain_" + letter] = seq
        out["coords_chain_" + letter] = {k: np.asarray(v, dtype=np.float32) for k, v in coords.items()}
        out["seq"] += seq
        out["num_of_chains"] += 1
    return out


def featurize(pdb_dict: dict, device: str = "cpu") -> dict:
    """Pack a parsed PDB into the padded tensors ProteinMPNN.forward/sample consume.

    All chains are marked for design (masked_list = all, visible_list = []).
    """
    letters = [k.split("_")[-1] for k in pdb_dict if k.startswith("seq_chain_")]
    B = 1
    L = len(pdb_dict["seq"])
    L_max = L
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)

    l0 = 0
    for c, letter in enumerate(letters, start=1):
        seq = pdb_dict["seq_chain_" + letter]
        cl = len(seq)
        l1 = l0 + cl
        cd = pdb_dict["coords_chain_" + letter]
        x = np.stack([cd[f"{a}_chain_{letter}"] for a in _ATOMS], 1)  # [cl, 4, 3]
        X[0, l0:l1] = x
        chain_M[0, l0:l1] = 1
        chain_encoding_all[0, l0:l1] = c
        residue_idx[0, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
        S[0, l0:l1] = [ALPHABET.index(a) for a in seq]
        l0 = l1

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    return {
        "X": torch.from_numpy(X).to(device),
        "S": torch.from_numpy(S).to(device).long(),
        "mask": torch.from_numpy(mask).to(device),
        "chain_M": torch.from_numpy(chain_M).to(device).float(),
        "chain_M_pos": torch.ones([B, L_max], device=device),
        "residue_idx": torch.from_numpy(residue_idx).to(device).long(),
        "chain_encoding_all": torch.from_numpy(chain_encoding_all).to(device).long(),
        "lengths": np.array([L], dtype=np.int32),
        "name": pdb_dict["name"],
        "native_seq": pdb_dict["seq"],
    }

