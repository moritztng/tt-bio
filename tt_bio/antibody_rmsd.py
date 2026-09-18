"""Per-region antibody Fv RMSD, reproducing ABodyBuilder3's published evaluation.

This is the measuring instrument for the ABodyBuilder3 reproduction, and it is validated the
only way an instrument can honestly be validated: by recomputing *their* released per-structure
numbers from *their* released structures. It must never be adjusted until it agrees with a model
of ours.

Upstream is Kenlay et al., Bioinformatics 40(10):btae576, structures and per-structure
``evaluate.csv`` at Zenodo ``10.5281/zenodo.11354577`` (CC-BY-4.0).

Two conventions here are surprising, and both were recovered by matching their numbers rather
than assumed:

* **"Backbone" means N, CA, C, CB -- not N, CA, C, O.** Their ``extract_backbone_coordinates``
  (``abodybuilder3/loss/aligned_rmsd.py``) slices ``positions[:, :, :4]`` and documents the input
  as "(B, n, 14/37, 3)". Under atom14 that slice is N, CA, C, O; under atom37 it is N, CA, C, CB
  (``residue_constants.atom_types``). The published evaluation used atom37, so the carbonyl O is
  absent and CB is present, and glycine contributes three atoms rather than four. Scoring with
  N, CA, C, O instead moves CDR-H3 by about 0.07 A, which is comfortably enough to invalidate a
  0.20 A accuracy bar.
* **Superposition is per chain, over that chain's whole backbone.** Heavy and light are each
  superposed onto the truth independently and every region of that chain is then measured under
  its chain's transform. Framework-only superposition, the other common convention, is not what
  they did.

Regions come from the dataset, which carries ANARCI IMGT numbering with IMGT region boundaries.
``region_indices_from_pdb`` recovers the residue-to-region mapping from the PDB residue numbers
that their writer emits (heavy ``1..n_h``, light ``501..``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from tt_bio.align import rigid_transform

#: Atom names their evaluation scores, in atom37 order. See the module docstring: this is
#: ``atom37[:4]``, so CB rather than the carbonyl O.
BACKBONE_ATOMS = ("N", "CA", "C", "CB")

#: The true backbone, for anything that is not reproducing their table.
TRUE_BACKBONE_ATOMS = ("N", "CA", "C", "O")

HEAVY_REGIONS = ("fwh1", "cdrh1", "fwh2", "cdrh2", "fwh3", "cdrh3", "fwh4")
LIGHT_REGIONS = ("fwl1", "cdrl1", "fwl2", "cdrl2", "fwl3", "cdrl3", "fwl4")
REGION_NAMES = HEAVY_REGIONS + LIGHT_REGIONS

#: Where the light chain's residue numbering starts in their PDB writer, which uses the jump to
#: mark the chain break.
LIGHT_RESSEQ_START = 501


def read_atoms(path: str | Path, atoms=BACKBONE_ATOMS) -> dict:
    """Parse a PDB into ``{(chain, resseq, icode): {atom_name: (x, y, z)}}``.

    Only ``atoms`` are kept. Residues are keyed by their written number, so a residue their
    PDB-fixer dropped for having no resolved atoms simply does not appear.
    """
    wanted = set(atoms)
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in wanted:
            continue
        out.setdefault((line[21], int(line[22:26]), line[26]), {})[name] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    return out


def residue_indices(residue_keys, regions) -> dict:
    """Map each residue key to its index in the per-residue region list.

    Two numbering conventions appear in the released output, and a file can be missing residues,
    so neither position nor residue number works on its own:

    * Their inference stage writes heavy ``1..n_h`` and light ``501..``, the jump marking the
      chain break. Their PDB fixer then drops residues with no resolved atoms, so a chain can be
      short and position within it is no longer the index -- the residue number is.
    * The ABodyBuilder2 baseline structures carry real IMGT numbering with insertion codes, which
      is sparse, so residue-number arithmetic does not apply and position does.

    A chain holding exactly as many residues as the region list expects is mapped by position;
    a short chain is mapped by residue number. Doing this per file and then pairing on the index
    is what lets a truth numbered ``501..`` be compared with a prediction that restarts at 1.
    """
    n_heavy = sum(1 for r in regions if r in HEAVY_REGIONS)
    expected = {"H": n_heavy, "L": len(regions) - n_heavy}
    out = {}
    for chain in ("H", "L"):
        keys = [k for k in residue_keys if (k[0] == "H") == (chain == "H")]
        base = 0 if chain == "H" else n_heavy
        if len(keys) == expected[chain]:
            out.update({k: base + i for i, k in enumerate(keys)})
            continue
        for key in keys:
            idx = key[1] - 1 if chain == "H" else n_heavy + key[1] - LIGHT_RESSEQ_START
            if not 0 <= idx < len(regions):
                raise ValueError(
                    f"residue {key} maps outside the region list (n_heavy={n_heavy}); neither "
                    f"positional nor numbered assignment applies"
                )
            out[key] = idx
    return out


def gather(true_atoms: dict, pred_atoms: dict, regions, atoms=BACKBONE_ATOMS):
    """Pair up the atoms present in both structures.

    Returns ``(chains, region_of_atom, true_xyz, pred_xyz)``. Residues are paired through their
    index into the region list, and an atom is scored only when both structures have it, which
    drops the OXT and hydrogens their PDB-fixer adds and the atoms the crystal never resolved.
    """
    true_idx = residue_indices(list(true_atoms), regions)
    pred_idx = residue_indices(list(pred_atoms), regions)
    by_index = {i: k for k, i in pred_idx.items()}
    chains, region_of, t_xyz, p_xyz = [], [], [], []
    for t_key, idx in true_idx.items():
        p_key = by_index.get(idx)
        if p_key is None:
            continue
        for name in atoms:
            if name in true_atoms[t_key] and name in pred_atoms[p_key]:
                chains.append(t_key[0])
                region_of.append(regions[idx])
                t_xyz.append(true_atoms[t_key][name])
                p_xyz.append(pred_atoms[p_key][name])
    return (
        chains,
        region_of,
        torch.tensor(t_xyz, dtype=torch.float64),
        torch.tensor(p_xyz, dtype=torch.float64),
    )


def per_region_rmsd(chains, region_of, true_xyz, pred_xyz) -> dict[str, float]:
    """Superpose each chain onto the truth, then score every region under its chain's transform.

    Returns ``rmsd_H``/``rmsd_L`` for the whole chain and ``rmsd_fwh``/``rmsd_cdrh1`` and so on
    per region, matching the column names in their ``evaluate.csv``.
    """
    chains = torch.tensor([c == "H" for c in chains])
    regions = list(region_of)
    out: dict[str, float] = {}
    for is_heavy, chain, tag in ((True, "H", "h"), (False, "L", "l")):
        sel = chains if is_heavy else ~chains
        if not sel.any():
            continue
        rot, mean_pred, mean_true = rigid_transform(pred_xyz[sel], true_xyz[sel])
        aligned = (pred_xyz[sel] - mean_pred) @ rot + mean_true
        sq = ((aligned - true_xyz[sel]) ** 2).sum(-1)
        here = [r for r, s in zip(regions, sel.tolist()) if s]
        out[f"rmsd_{chain}"] = float(sq.mean().sqrt())
        for group, name in (
            (tuple(f"fw{tag}{i}" for i in "1234"), f"rmsd_fw{tag}"),
            *(((f"cdr{tag}{i}",), f"rmsd_cdr{tag}{i}") for i in "123"),
        ):
            mask = torch.tensor([r in group for r in here])
            if mask.any():
                out[name] = float(sq[mask].mean().sqrt())
    return out


def score_pdb_pair(true_pdb, pred_pdb, regions, atoms=BACKBONE_ATOMS) -> dict[str, float]:
    """Per-region RMSD between two PDBs written by their inference stage.

    ``regions`` is the per-residue region list, which ships in the released
    ``<variant>/plddt/<structure>.pt`` and in the dataset's own per-structure files.
    """
    return per_region_rmsd(
        *gather(read_atoms(true_pdb, atoms), read_atoms(pred_pdb, atoms), regions, atoms)
    )
