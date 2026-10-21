"""The oracle-bound half of `rfd4-evaluate`: refolding metrics and pass criteria.

A refold is read, never produced, here. Upstream's `OracleMetric.compute` folds a design only when
its confidence JSON is absent from the design's output directory and otherwise loads what is there
(`oracles/oracle_metric.py:471-520`), so a refold written by `tt_bio.rfd4.oracle`'s Tenstorrent
oracle, by an earlier run, or by AF3 itself is scored identically. That cache is the seam this
module reads:

    <out_dir>/evaluation_samples/<example_id>/<example_id>[_<save_id>]_<oracle_name>_confidences.json
    <out_dir>/evaluation_samples/<example_id>/<example_id>[_<save_id>]_<oracle_name>_refolded.cif

The confidences file is a flat dict of the oracle's already-reduced metric keys, which is exactly
what `tt_bio.rfd4.oracle.extract_confidence_metrics` returns, plus `chain_of_interest`.

Both sides lose their polymer terminal oxygens before any comparison
(`metrics/utils/metrics_utils.py:349-358`), so a refold that grew or dropped an OXT still compares.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tt_bio.rfd4 import eval_metrics as M
from tt_bio.rfd4.eval_io import read_design

# atomworks `constants.py`: what `is_protein_backbone` counts, OXT included (it is removed before
# the comparison, so it only matters for a design that kept one in an atomized residue).
PROTEIN_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O", "OXT")

#: Upstream's default oracle config, `configs/metrics/all_metrics.yaml`: AF3 under three keys.
#: (config key, oracle_name, gate tags). The config key prefixes the columns, `oracle_name` the
#: file; a pass criterion reads `oracle_ppi.*` whichever oracle produced it.
ORACLES = (
    ("oracle", "af3", {
        "af3_refolding", "unconditional_refolding", "motif_scaffold_refolding",
        "internal_coordinate_motif_scaffold_refolding", "sm_binder_refolding", "ame_refolding",
        "ame_no_ligand_refolding", "ame_original_benchmark_refolding", "disco_refolding",
    }),
    ("oracle_ppi", "af3_ppi", {"ppi_refolding", "antibody"}),
    ("oracle_dna", "af3_dna", {"dna_refolding"}),
)  # fmt: skip


def remove_terminal_oxygen(design):
    """atomworks `ml/transforms/filters.py:555-595`, as upstream calls it on both sides.

    A chain's OXT (protein) or OP3 (nucleic acid) goes only when that chain has more than one
    residue and the atom is not atomized, so a free amino acid keeps its OXT.
    """
    atoms = design.atoms
    chains = atoms.chain_id
    starts = M.residue_starts(atoms)
    labels, counts = np.unique(chains[starts], return_counts=True)
    multi_residue = np.isin(chains, labels[counts > 1])
    terminal = ((atoms.atom_name == "OXT") & design.is_protein) | (
        (atoms.atom_name == "OP3") & design.is_polymer & ~design.is_protein
    )
    return ~(terminal & multi_residue & ~design.atomize)


def masks(design, coi):
    """The AtomWorks mask vocabulary the refolding configs are written in, for one structure."""
    atoms = design.atoms
    is_protein_backbone = design.is_protein & np.isin(atoms.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    is_nucleic_acid = design.is_polymer & ~design.is_protein
    return {
        "is_protein": design.is_protein,
        "is_protein_backbone": is_protein_backbone,
        "is_nucleic_acid": is_nucleic_acid,
        "is_dna": is_nucleic_acid & np.char.startswith(atoms.res_name.astype(str), "D"),
        "is_ligand": ~design.is_polymer,
        "is_chain_of_interest": (
            np.zeros(atoms.array_length(), dtype=bool) if coi is None else chain_iid(atoms) == coi
        ),
        "C_CRD": design.is_crd,
        "SA_CRDSRC": design.crd_src,
        "is_CA": atoms.atom_name == "CA",
        "all": np.ones(atoms.array_length(), dtype=bool),
    }


def chain_iid(atoms):
    """atomworks' chain instance id: the chain plus its assembly transform, `A` -> `A_1`.

    Every structure on the evaluate path comes out of assembly 1 under the identity transform, so
    the suffix is always `_1`. It is the label upstream writes in the `chain_of_interest` column
    and the one the per-chain oracle keys are named after, so the port has to carry it too.
    """
    return np.char.add(atoms.chain_id.astype(str), "_1")


#: `configs/metrics/refolding_metrics/all_refolding_metrics.yaml`, in file order:
#: (name, tags that switch it on, align mask, subset mask). Each mask names keys of `masks()`;
#: a leading `~` negates. `required_tags_all: [is_refold:untransformed]` holds for all of these,
#: which this port satisfies by scoring only untransformed refolds.
REFOLDING = (
    (
        "refolded_backbone_rmsd",
        {
            "unconditional_refolding", "ppi_refolding", "antibody", "dna_refolding",
            "motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding",
            "sm_binder_refolding", "ame_refolding", "ame_no_ligand_refolding",
            "ame_original_benchmark_refolding", "disco_refolding",
        },
        ["is_protein_backbone"],
        ["is_protein_backbone"],
    ),
    (
        "refolded_target_aligned_binder_rmsd",
        {"ppi_refolding", "dna_refolding", "antibody"},
        ["~is_chain_of_interest", ("C_CRD", "is_dna")],
        ["is_chain_of_interest", "is_protein"],
    ),
    (
        "refolded_target_backbone_aligned_binder_rmsd",
        {"ppi_refolding"},
        ["~is_chain_of_interest", "C_CRD", "is_protein_backbone"],
        ["is_chain_of_interest", "is_protein"],
    ),
    (
        "refolded_backbone_aligned_ligand_rmsd",
        {"sm_binder_refolding"},
        ["is_protein_backbone"],
        ["is_ligand"],
    ),
)  # fmt: skip

#: The motif pair, `subdivide_by_annotation: condition_CoordinateGroup_1_atom` with the design's
#: coordinates taken from `annotation_coords-with-insertion_1_atom`: each coordinate group is
#: resolved for symmetry, aligned and scored on its own, and the groups are pooled as
#: sqrt(sum(rmsd^2 * n) / sum(n)) (`gt_structure_comparison.py:402-462`).
MOTIF = (
    (
        "refolded_motif_allatom_rmsd",
        {
            "motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding", "ame_refolding",
            "ame_no_ligand_refolding", "ame_original_benchmark_refolding",
        },
        ["SA_CRDSRC", "is_protein"],
    ),
    (
        "refolded_motif_ca_rmsd",
        {"motif_scaffold_refolding", "internal_coordinate_motif_scaffold_refolding"},
        ["SA_CRDSRC", "is_protein", "is_CA"],
    ),
)  # fmt: skip

#: Refolding metrics this port does not compute, with the tags that switch them on. Apo/holo needs
#: a second refold from `ApoOnlyTransform`; the DISCO one needs `LigandCentroidDistanceMetric`.
REFOLDING_NOT_PORTED = {
    "refolded_apo_holo_rmsd": {"apo_refolding"},
}  # fmt: skip

#: `configs/metrics/pass_criteria/all_pass_criteria.yaml`: (name, gate tags, prohibited tags,
#: {metric key: bounds}). Bounds are upstream's own, read off the config rather than the docs --
#: `developability.rosetta_developability.sap_score` is 50.0 there and 75 in the published table.
_MOTIF_THRESHOLDS = {
    "refolded_backbone_rmsd.rmsd": {"max": 2.0},
    "refolded_motif_allatom_rmsd.rmsd": {"max": 2.0},
    "refolded_motif_ca_rmsd.rmsd": {"max": 1.0},
}

PASS_CRITERIA = (
    ("motif_scaffold", {"motif_scaffold_refolding"}, set(), _MOTIF_THRESHOLDS),
    ("internal_coords", {"internal_coordinate_motif_scaffold_refolding"}, set(), _MOTIF_THRESHOLDS),
    ("unconditional", {"unconditional_refolding"}, set(), {"refolded_backbone_rmsd.rmsd": {"max": 1.5}}),
    ("ppi", {"ppi_refolding"}, set(), {
        "oracle_ppi.ptm_coi": {"min": 80.0},
        "oracle_ppi.min_ipae_coi": {"max": 1.5},
        "refolded_target_backbone_aligned_binder_rmsd.rmsd": {"max": 2.5},
    }),
    ("antibody", {"antibody"}, set(), {
        "oracle_ppi.min_ipae_coi": {"max": 2.0},
        "refolded_target_aligned_binder_rmsd.rmsd": {"max": 2.0},
    }),
    ("dna", {"dna_refolding"}, set(), {
        "oracle_dna.min_ipae_coi": {"max": 1.5},
        "refolded_target_aligned_binder_rmsd.rmsd": {"max": 8.0},
    }),
    ("sm_binder", {"sm_binder_refolding"}, set(), {
        "oracle.min_ipae_coi": {"max": 1.5},
        "oracle.iptm_coi": {"min": 80.0},
        "refolded_backbone_rmsd.rmsd": {"max": 1.5},
        "refolded_backbone_aligned_ligand_rmsd.rmsd": {"max": 5.0},
    }),
    ("ame", {"ame_refolding"}, {"no_ligand"}, {
        "oracle.min_ipae_coi": {"max_exclusive": 1.5},
        "refolded_backbone_rmsd.rmsd": {"max": 2.0},
        "refolded_motif_allatom_rmsd.rmsd": {"max": 1.5},
        "clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A": {"max": 0},
    }),
    ("disco", {"disco_refolding"}, set(), {
        "refolded_backbone_rmsd.rmsd": {"max": 2.0},
        "max_ligand_centroid_distance.ligand_centroid_distance": {"max": 2.0},
    }),
    ("ame_no_ligand", {"ame_refolding"}, set(), {
        "refolded_backbone_rmsd.rmsd": {"max": 2.0},
        "refolded_motif_allatom_rmsd.rmsd": {"max": 1.5},
    }, {"no_ligand"}),
)  # fmt: skip

#: Pass criteria whose thresholds read a metric this port does not compute.
PASS_NOT_PORTED = {
    "developability": {"developability"},
}


def _resolve(mask_spec, table):
    """A config mask expression as this port spells it: terms ANDed, a tuple inside it ORed."""
    out = None
    for term in mask_spec:
        if isinstance(term, tuple):
            part = np.zeros_like(table["all"])
            for name in term:
                part |= table[name]
        elif term.startswith("~"):
            part = ~table[term[1:]]
        else:
            part = table[term]
        out = part if out is None else (out & part)
    return out


def refold_paths(samples_dir, example_id, oracle_name):
    """The cached confidences and refold for one design, or `(None, None)` if it has none."""
    design_dir = Path(samples_dir) / example_id
    stem = f"{example_id}_{oracle_name}"
    confidences = design_dir / f"{stem}_confidences.json"
    if not confidences.exists():
        return None, None
    refolded = next(
        (p for s in (".cif", ".cif.gz", ".cif.zst", ".bcif") if (p := design_dir / f"{stem}_refolded{s}").exists()),
        None,
    )
    return confidences, refolded


def oracle_metrics(samples_dir, example_id, tags):
    """Every active oracle's cached columns for one design, prefixed by its config key.

    Returns `(columns, refolds)`: the columns upstream's oracle manager would have produced, and
    the parsed refold per oracle key for the refolding metrics to score.
    """
    columns, refolds, coi = {}, {}, None
    for key, oracle_name, gate in ORACLES:
        if not (tags is None or tags & gate):
            continue
        confidences, refolded = refold_paths(samples_dir, example_id, oracle_name)
        if confidences is None:
            continue
        values = json.loads(confidences.read_text())
        coi = coi or values.get("chain_of_interest")
        # Upstream strips every key ending in `chain_of_interest` out of the row and folds them
        # into the single `chain_of_interest` column (`proteina.py:268-290`), so the oracle-
        # prefixed one is never a column of its own.
        columns.update(
            {f"{key}.{name}": value for name, value in values.items() if not name.endswith("chain_of_interest")}
        )
        columns[f"{key}.refolded_structure_path"] = str(refolded) if refolded else None
        columns[f"{key}.confidence_json_path"] = str(confidences)
        if refolded is not None:
            refolds[key] = refolded
    return columns, refolds, coi


def refolding_metrics(design, refolds, tags, coi):
    """Score every active refolding metric of every refold this design has.

    One pass per refold, as upstream does, each handed the design as ground truth and that refold
    as the prediction (`metrics_utils.py:349-440`).
    """
    out = {}
    design_keep = remove_terminal_oxygen(design)
    gt = design.atoms[design_keep]
    gt_masks = {name: mask[design_keep] for name, mask in masks(design, coi).items()}
    for _key, path in sorted(refolds.items()):
        refold = read_design(path)
        keep = remove_terminal_oxygen(refold)
        pred = refold.atoms[keep]
        pred_masks = {name: mask[keep] for name, mask in masks(refold, coi).items()}
        for name, gate, align, subset in REFOLDING:
            if not (tags is None or tags & gate):
                continue
            result = M.structure_comparison(
                gt,
                _resolve(align, gt_masks),
                _resolve(subset, gt_masks),
                pred,
                _resolve(align, pred_masks),
                _resolve(subset, pred_masks),
                design.is_polymer[design_keep],
            )
            out.update({f"{name}.{k}": v for k, v in result.items()})
        if tags is None or "disco_refolding" in tags:
            result = ligand_centroid_distance(gt, gt_masks, pred, pred_masks)
            out.update({f"max_ligand_centroid_distance.{k}": v for k, v in result.items()})
        for name, gate, mask in MOTIF:
            if not (tags is None or tags & gate):
                continue
            motif_gt = gt.copy()
            motif_gt.coord = design.coords_with_insertion[design_keep].astype(np.float32)
            result = motif_rmsd(
                motif_gt,
                _resolve(mask, gt_masks),
                design.crd_group[design_keep],
                pred,
                _resolve(mask, pred_masks),
                design.is_polymer[design_keep],
            )
            out.update({f"{name}.{k}": v for k, v in result.items()})
    return out


def design_motif_rmsd(design):
    """Inline `design_motif_allatom_rmsd` (`design_metrics.yaml:68-79`): the design scored against
    its own conditioning, i.e. ground truth = the design with `coords-with-insertion` as its
    coordinates, prediction = the design, both masked to `SA_CRDSRC`, per coordinate group."""
    gt = design.atoms.copy()
    gt.coord = design.coords_with_insertion.astype(np.float32)
    return motif_rmsd(gt, design.crd_src, design.crd_group, design.atoms, design.crd_src, design.is_polymer)


def _ligand_molecules(atoms, is_ligand):
    """`gt_structure_comparison._get_ligand_molecules`: centroid and residue-name set per bonded
    ligand molecule, in molecule order."""
    from tt_bio.rfd4.eval_symmetry import _molecules

    lig = atoms[is_ligand]
    if lig.array_length() == 0:
        return [], []
    molecules, _ = _molecules(lig, np.zeros(lig.array_length(), dtype=bool))
    return (
        [np.mean(lig.coord[m], axis=0) for m in molecules],
        [frozenset(lig.res_name[m].tolist()) for m in molecules],
    )


def ligand_centroid_distance(gt, gt_masks, pred, pred_masks):
    """`max_ligand_centroid_distance` (`LigandCentroidDistanceMetric`, aggregation max): the refold
    is fit to the design on protein backbone, then each ligand molecule's centroid is compared.
    NaN when either side has no ligand, the counts differ, or the residue-name sets are duplicated
    or mismatched, as upstream; {} when the alignment fails."""
    from biotite.structure import superimpose

    g = gt_masks["is_protein_backbone"] & M._finite(gt)
    p = pred_masks["is_protein_backbone"] & M._finite(pred)
    if g.sum() != p.sum() or g.sum() == 0:
        return {}
    _, transform = superimpose(gt.coord[g], pred.coord[p])
    moved = pred.copy()
    moved.coord = transform.apply(pred.coord)
    gt_centroids, gt_fps = _ligand_molecules(gt, gt_masks["is_ligand"])
    pred_centroids, pred_fps = _ligand_molecules(moved, pred_masks["is_ligand"])
    nan = {"ligand_centroid_distance": float("nan")}
    if not gt_centroids or not pred_centroids or len(gt_centroids) != len(pred_centroids):
        return nan
    if len(set(gt_fps)) != len(gt_fps) or len(set(pred_fps)) != len(pred_fps) or gt_fps != pred_fps:
        return nan
    distances = [float(np.linalg.norm(a - b)) for a, b in zip(gt_centroids, pred_centroids)]
    return {"ligand_centroid_distance": float(np.max(distances))}


def motif_rmsd(gt, gt_mask, groups, pred, pred_mask, gt_is_polymer):
    """`StructureComparisonMetric._compute_per_group` with align and subset the same mask.

    Each value of the group annotation (read on the design) is a slice of both structures; symmetry
    is resolved inside the slice, then aligned and scored on the mask there. A slice with no
    result, or none of its atoms left, drops out, and the rest pool by atom count."""
    total, weighted = 0, 0.0
    for value in np.unique(groups):
        sel = groups == value
        result = M.structure_comparison(
            gt[sel], gt_mask[sel], gt_mask[sel], pred[sel], pred_mask[sel], pred_mask[sel], gt_is_polymer[sel],
            return_n_atoms=True,
        )
        if not result or result["n_atoms"] == 0:
            continue
        total += result["n_atoms"]
        weighted += result["rmsd"] ** 2 * result["n_atoms"]
    if total == 0:
        return {}
    return {"rmsd": float(np.sqrt(weighted / total))}


def pass_criteria(metrics, tags):
    """`oracles/pass_criterion_metric.py:118-150`: every threshold must hold, a missing key is
    indeterminate and scores `pass=0` with `all_metrics_present=0`."""
    out = {}
    for name, gate, prohibited, thresholds, *required_all in PASS_CRITERIA:
        if tags is not None and (not (tags & gate) or tags & prohibited):
            continue
        if tags is not None and required_all and not required_all[0] <= tags:
            continue
        present, passed = True, True
        for key, bounds in thresholds.items():
            value = metrics.get(key)
            if value is None:
                present = passed = False
                continue
            if (lo := bounds.get("min")) is not None and not value >= lo:
                passed = False
            if (hi := bounds.get("max")) is not None and not value <= hi:
                passed = False
            if (lo := bounds.get("min_exclusive")) is not None and not value > lo:
                passed = False
            if (hi := bounds.get("max_exclusive")) is not None and not value < hi:
                passed = False
        out[f"{name}.pass"] = float(passed)
        out[f"{name}.all_metrics_present"] = float(present)
    return out


def not_ported(tags):
    """The oracle-side metrics a tag set switches on that this module does not compute."""
    gates = {**REFOLDING_NOT_PORTED, **PASS_NOT_PORTED}
    return [name for name, gate in gates.items() if tags is None or tags & gate]
