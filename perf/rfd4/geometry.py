"""Oracle-free geometry metrics, read straight off a design CIF.

These are the harness's own columns (`geom.*`), not upstream's, and the grade labels them that
way. They exist for two reasons: a grade is possible before an oracle is wired to a card, and a
collapsed or NaN-poisoned design is caught by arithmetic nobody has to trust a model for.

Only the mmCIF `atom_site` loop is read, so a generated `.cif` or `.cif.zst` works with no
parser dependency.
"""
from __future__ import annotations

import math
import pathlib
import subprocess

import numpy as np

CA_BOND_LO, CA_BOND_HI = 3.60, 4.10  # trans peptide CA-CA is ~3.80 A
CA_CLASH = 3.60  # two CAs this close, >=3 apart in sequence, are on top of each other


def _read_text(path: pathlib.Path) -> str:
    if path.suffix == ".zst":
        return subprocess.run(
            ["zstd", "-dc", str(path)], check=True, capture_output=True, text=True
        ).stdout
    return path.read_text()


def parse_atom_site(path: pathlib.Path) -> dict[str, list]:
    """Return the atom_site loop as column name -> list of raw string values."""
    text = _read_text(path)
    lines = text.splitlines()
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    for line in lines:
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if in_loop:
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                break
            parts = _split_cif_row(s)
            if len(parts) == len(cols):
                rows.append(parts)
    return {c: [r[i] for r in rows] for i, c in enumerate(cols)}


def _split_cif_row(s: str) -> list[str]:
    out, cur, quote = [], "", ""
    for ch in s:
        if quote:
            if ch == quote:
                quote = ""
            else:
                cur += ch
        elif ch in "'\"":
            quote = ch
        elif ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _float(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def design_geometry(path: pathlib.Path, chain: str | None = None) -> dict[str, float]:
    """Geometry columns for one design. `chain` restricts to the chain of interest."""
    site = parse_atom_site(path)
    if not site or "Cartn_x" not in site:
        return {}
    xyz = np.array(
        [[_float(x), _float(y), _float(z)] for x, y, z in zip(site["Cartn_x"], site["Cartn_y"], site["Cartn_z"])]
    )
    n_nonfinite = int(np.count_nonzero(~np.isfinite(xyz)))
    atom = np.array(site.get("label_atom_id", []))
    chains = np.array(site.get("label_asym_id", [""] * len(xyz)))
    comp = np.array(site.get("label_comp_id", [""] * len(xyz)))
    seq_id = np.array([_float(v) for v in site.get("label_seq_id", ["nan"] * len(xyz))])

    sel = atom == "CA"
    if chain is not None:
        sel &= chains == chain
    ca = xyz[sel]
    ca = ca[np.isfinite(ca).all(axis=1)]
    out: dict[str, float] = {
        "geom.n_nonfinite_coords": float(n_nonfinite),
        "geom.n_ca": float(ca.shape[0]),
        "geom.frac_unknown_residue": _unk_fraction(comp[sel]) if sel.any() else float("nan"),
    }
    if ca.shape[0] < 3:
        out["geom.radius_of_gyration"] = float("nan")
        out["geom.frac_ca_bond_outlier"] = float("nan")
        out["geom.ca_clash_frac"] = float("nan")
        return out

    out["geom.radius_of_gyration"] = float(np.sqrt(((ca - ca.mean(axis=0)) ** 2).sum(axis=1).mean()))

    # Consecutive CAs, within one chain only: a chain break is not a bond.
    sel_chain = chains[sel][: ca.shape[0]]
    sel_seq = seq_id[sel][: ca.shape[0]]
    step = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    same_chain = sel_chain[1:] == sel_chain[:-1]
    adjacent = np.abs(np.diff(sel_seq)) == 1
    bonded = same_chain & (adjacent | ~np.isfinite(np.diff(sel_seq)))
    if bonded.any():
        d = step[bonded]
        out["geom.frac_ca_bond_outlier"] = float(np.mean((d < CA_BOND_LO) | (d > CA_BOND_HI)))
    else:
        out["geom.frac_ca_bond_outlier"] = float("nan")

    dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    idx = np.arange(ca.shape[0])
    far_in_seq = np.abs(idx[:, None] - idx[None, :]) >= 3
    cross_chain = sel_chain[:, None] != sel_chain[None, :]
    considered = far_in_seq | cross_chain
    n_pairs = int(np.count_nonzero(np.triu(considered, 1)))
    n_clash = int(np.count_nonzero(np.triu(considered & (dist < CA_CLASH), 1)))
    out["geom.ca_clash_frac"] = float(n_clash / n_pairs) if n_pairs else float("nan")
    return out


def _unk_fraction(comps: np.ndarray) -> float:
    if comps.size == 0:
        return float("nan")
    return float(np.mean(np.isin(comps, ["UNK", "UNL", "UNX"])))


def sequence_of(path: pathlib.Path, chain: str | None = None) -> str:
    """One-letter sequence of the CA atoms, used for the diversity guard."""
    site = parse_atom_site(path)
    if not site:
        return ""
    atom = np.array(site.get("label_atom_id", []))
    comp = np.array(site.get("label_comp_id", []))
    chains = np.array(site.get("label_asym_id", [""] * len(atom)))
    sel = atom == "CA"
    if chain is not None:
        sel &= chains == chain
    return "".join(THREE_TO_ONE.get(c, "X") for c in comp[sel])


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def nan_safe_mean(values) -> float:
    vals = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")
