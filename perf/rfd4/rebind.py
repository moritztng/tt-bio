"""Bind a reported quality number to the artifact it claims to come from.

A grade reads `per_example_metrics.csv`. Nothing checks that the numbers in it describe the CIFs
sitting beside it, that each design was counted once, or that the code which produced them is the
code we ship. This module checks all three, from the files themselves:

* `rederive` recomputes the two headline columns off the CIFs the row names -- backbone RMSD
  between the design and its refold (Kabsch on protein backbone atoms), and sequence identity
  between the design and its `_input.cif` -- and reports every disagreement with the CSV.
* each `designed_structure_path` must appear exactly once across the arm's CSVs, so a design
  copied into two trees cannot be counted twice.
* every artifact read is recorded by sha256, and an arm's source commit is rejected unless it is
  an ancestor of `origin/main`.

Method: Magnus Bauer's `scripts/rfd4_release_validation_aggregate.py` (rfd4-experimental @
7fbfc2704), which re-derives sequence identity and CA RMSD from the CIFs and fails on disagreement.
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry  # noqa: E402

#: `atomworks.constants.PROTEIN_BACKBONE_ATOM_NAMES`, the `is_protein_backbone` mask the refold
#: metric aligns and scores on (`tt_bio/rfd4/eval_refold.py:103-114`).
BACKBONE = ("N", "CA", "C", "O", "OXT")
RMSD_COLUMN = "refolded_backbone_rmsd.rmsd"
IDENTITY_COLUMN = "sequence_similarity_to_input.sequence_identity_to_input"
#: Our Kabsch against biotite's on the same atoms: a tolerance for arithmetic, not for a different
#: number. A real disagreement is a tenth of an Angstrom, not a thousandth.
RMSD_TOL = 5e-3
IDENTITY_TOL = 1e-6


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def on_main(sha: str, repo=".") -> bool:
    """Is `sha` an ancestor of origin/main. A number from code that is not on main is not ours."""
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"],
                       cwd=str(repo), capture_output=True)
    return r.returncode == 0


def _atoms(path):
    """Protein backbone atoms of a CIF: `(keys, coords)`, keyed by chain, residue and atom name."""
    site = geometry.parse_atom_site(pathlib.Path(path))
    if not site:
        return [], np.zeros((0, 3))
    names = np.asarray(site.get("label_atom_id", []), dtype=object)
    keep = np.isin(names, BACKBONE)
    xyz = np.array([[geometry._float(v) for v in site[c]] for c in ("Cartn_x", "Cartn_y", "Cartn_z")]).T
    chain = np.asarray(site.get("label_asym_id", []), dtype=object)
    seq = np.asarray(site.get("label_seq_id", []), dtype=object)
    keys = [(str(chain[i]), str(seq[i]), str(names[i])) for i in np.flatnonzero(keep)]
    return keys, xyz[keep]


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD of `b` superimposed on `a` (both (n, 3)), the fit biotite's `superimpose` makes."""
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(bc.T @ ac)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    fitted = bc @ rot
    return float(np.sqrt(((fitted - ac) ** 2).sum(axis=1).mean()))


def rederive_rmsd(design, refolded) -> dict:
    """Backbone RMSD between a design and its refold, on the atoms both carry."""
    ka, xa = _atoms(design)
    kb, xb = _atoms(refolded)
    if not ka or not kb:
        return {"why": "no protein backbone atoms in one of the two structures"}
    index = {k: i for i, k in enumerate(kb)}
    shared = [(i, index[k]) for i, k in enumerate(ka) if k in index]
    if len(shared) != len(ka) or len(ka) != len(kb):
        # Upstream scores only when the two masks select the same atoms and returns nothing otherwise;
        # a partial overlap is a different quantity, so it is reported as unavailable, not as a number.
        return {"why": f"atoms differ: design {len(ka)}, refold {len(kb)}, shared {len(shared)}"}
    a = xa[[i for i, _ in shared]]
    b = xb[[j for _, j in shared]]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"why": "non-finite coordinates"}
    return {"value": kabsch_rmsd(a, b), "n_atoms": len(shared)}


def _residues(path) -> list[str]:
    """Residue names at the CA atoms, upstream's `_protein_sequence`. Compared as three-letter
    names, not one-letter codes: two different non-standard residues both read `X`."""
    site = geometry.parse_atom_site(pathlib.Path(path))
    if not site:
        return []
    atom = np.asarray(site.get("label_atom_id", []), dtype=object)
    comp = np.asarray(site.get("label_comp_id", []), dtype=object)
    return [str(c) for c in comp[atom == "CA"]]


def rederive_identity(design, input_cif) -> dict:
    """Sequence identity of a design against its input, over residues both name."""
    pred, ref = _residues(design), _residues(input_cif)
    if not pred or not ref or len(pred) != len(ref):
        return {"why": f"sequence lengths differ: design {len(pred)}, input {len(ref)}"}
    same = sum(1 for x, y in zip(pred, ref) if x == y)
    return {"value": same / len(pred), "n_residues": len(pred)}


def _sidecar(design: pathlib.Path, suffix: str):
    for ext in (".cif", ".cif.gz", ".cif.zst"):
        p = design.parent / (design.name.split(".")[0].replace("_generated", "") + suffix + ext)
        if p.exists():
            return p
    return None


def rederive(rows: list[dict], tol_rmsd: float = RMSD_TOL, tol_identity: float = IDENTITY_TOL) -> dict:
    """Every reported headline number in `rows`, against the artifacts the row names.

    A row whose artifact is absent is `unavailable` (named, with why); a row whose artifact says
    something else is a `mismatch`. Both are reported; neither is silently dropped.
    """
    checked, mismatches, unavailable, artifacts = 0, [], [], {}
    for row in rows:
        design = row.get("designed_structure_path")
        if not isinstance(design, str):
            continue
        design_path = pathlib.Path(design)
        for column, tol, other in ((RMSD_COLUMN, tol_rmsd, "refolded"), (IDENTITY_COLUMN, tol_identity, "_input")):
            reported = row.get(column)
            if not (isinstance(reported, float) and math.isfinite(reported)):
                continue
            if column == RMSD_COLUMN:
                partner = row.get("refolded_backbone_rmsd.refolded_structure_path")
                partner = pathlib.Path(partner) if isinstance(partner, str) else _sidecar(design_path, "_refolded")
            else:
                partner = _sidecar(design_path, "_input")
            if partner is None or not design_path.exists() or not pathlib.Path(partner).exists():
                unavailable.append({"design": design, "column": column,
                                    "why": f"no {other} artifact beside the design"})
                continue
            got = (rederive_rmsd(design_path, partner) if column == RMSD_COLUMN
                   else rederive_identity(design_path, partner))
            if "value" not in got:
                unavailable.append({"design": design, "column": column, "why": got["why"]})
                continue
            checked += 1
            artifacts[str(design_path)] = sha256(design_path)
            artifacts[str(partner)] = sha256(partner)
            if abs(got["value"] - reported) > tol:
                mismatches.append({"design": design, "column": column, "reported": round(reported, 6),
                                   "rederived": round(got["value"], 6), "tolerance": tol,
                                   "artifact": str(partner)})
    return {"checked": checked, "mismatches": mismatches, "unavailable": unavailable,
            "artifact_sha256": artifacts}


def duplicates(rows: list[dict]) -> list[str]:
    """Designs counted more than once in one arm."""
    seen: dict[str, int] = {}
    for row in rows:
        key = row.get("designed_structure_path")
        if isinstance(key, str):
            seen[key] = seen.get(key, 0) + 1
    return sorted(k for k, n in seen.items() if n > 1)


def check_arm(arm: dict, sha: str | None = None, repo=".", allow_unbound: bool = False) -> dict:
    """The rebinding verdict for one arm: numbers re-derived, designs counted once, source on main."""
    out = rederive(arm["rows"])
    out["duplicate_designs"] = duplicates(arm["rows"])
    out["source_sha"] = sha
    out["source_on_main"] = None if sha is None else on_main(sha, repo)
    reasons = []
    if out["mismatches"]:
        reasons.append(f"{len(out['mismatches'])} reported numbers disagree with their artifact")
    if out["duplicate_designs"]:
        reasons.append(f"{len(out['duplicate_designs'])} designs counted more than once")
    if sha is None and not allow_unbound:
        reasons.append("no source commit given for this arm")
    elif sha is not None and not out["source_on_main"]:
        reasons.append(f"source commit {sha[:9]} is not on main")
    out["reasons"] = reasons
    out["verdict"] = "BOUND" if not reasons else "REBIND_FAILED"
    return out
