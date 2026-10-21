"""Upstream RFD4 `rfd4-evaluate` inline metrics, reimplemented on numpy + biotite.

Upstream computes these in `rfproteina/metrics/design_metrics.py` against atomworks 3.0.4, which
needs python 3.12 and is a release-gated bump on tt-bio's side (it breaks RF3's featurizer). The
metrics themselves do not need it: what atomworks contributes is three small rules, vendored here
with the upstream file and line they come from, so `tt-bio evaluate` runs on tt-bio's own stack.

Exactness is not asserted by this module. It is proven by running upstream's `rfd4-evaluate` and
this port on the same designs and diffing the two CSVs with `perf/rfd4/eval_compare.py`.

Column names are upstream's, character for character, including the `1p5A` threshold spelling and
the `bond_length.` prefix inside the `bond_length_metrics` namespace.
"""

import numpy as np

# Vendored from atomworks `io/utils/bonds.py:660-694` (filter_bonds_by_distance defaults): a bond is
# unreasonable when it is longer than the threshold for the elements at its two ends.
BOND_THRESHOLD_CHNO = 1.8
BOND_THRESHOLD_CHNOPS = 2.4
BOND_THRESHOLD_OTHER = 3.6

_CHNO = frozenset({"C", "H", "N", "O"})
_CHNOPS = frozenset({"C", "H", "N", "O", "P", "S"})

CLASH_THRESHOLD = 1.5

CLASH_CATEGORIES = (
    "intra_residue_polymer",
    "intra_residue_ligand",
    "inter_residue_polymer_polymer",
    "inter_residue_polymer_ligand",
    "inter_residue_ligand_ligand",
)


def threshold_key(threshold: float) -> str:
    """`1.5` -> `'1p5A'`. Upstream `design_metrics.py:176`: the key avoids `.` (its namespace
    separator) and `,` (the CSV delimiter)."""
    return f"{threshold:.1f}".replace(".", "p") + "A"


def _chain_identifier(atoms):
    categories = atoms.get_annotation_categories()
    return atoms.chain_iid if "chain_iid" in categories else atoms.chain_id


def _bond_threshold(elem_a, elem_b):
    """Per-bond threshold, vectorised over the two element arrays."""
    a = np.char.upper(elem_a.astype(str))
    b = np.char.upper(elem_b.astype(str))
    in_chno = np.isin(a, list(_CHNO)) & np.isin(b, list(_CHNO))
    in_chnops = np.isin(a, list(_CHNOPS)) & np.isin(b, list(_CHNOPS))
    out = np.full(a.shape, BOND_THRESHOLD_OTHER)
    out[in_chnops] = BOND_THRESHOLD_CHNOPS
    out[in_chno] = BOND_THRESHOLD_CHNO
    return out


def _bond_spatial_masks(atoms, a1, a2, is_polymer):
    """Upstream `design_metrics.py:14-47`. Mutually exclusive except that
    `between_chain.nonpolymer_to_polymer` is an XOR sub-category of between-chain."""
    chain = _chain_identifier(atoms)
    same_chain = chain[a1] == chain[a2]
    same_res = same_chain & (atoms.res_id[a1] == atoms.res_id[a2])
    cross_chain = ~same_chain
    poly_a1, poly_a2 = is_polymer[a1], is_polymer[a2]
    return {
        "within_residue": same_res,
        "between_residue.within_chain": ~same_res & same_chain,
        "between_chain.polymer_polymer": cross_chain & poly_a1 & poly_a2,
        "between_chain.nonpolymer_to_polymer": cross_chain & (poly_a1 ^ poly_a2),
    }


def compute_bond_length_stats(atoms, is_polymer, verbose: bool = True) -> dict:
    """`bond_length_metrics`, upstream `design_metrics.py:50-105`.

    A bond whose either endpoint has a non-finite coordinate is preserved, never flagged
    (atomworks `bonds.py:702`). With no bonds at all the metric contributes no columns, which is
    how upstream leaves the cell empty rather than writing a zero.
    """
    if atoms.bonds is None:
        return {}
    bonds = atoms.bonds.as_array()
    n_orig = len(bonds)
    if n_orig == 0:
        return {}

    a1, a2 = bonds[:, 0], bonds[:, 1]
    coords = atoms.coord
    dist = np.linalg.norm(coords[a1] - coords[a2], axis=1)
    finite = np.isfinite(coords[a1]).all(axis=1) & np.isfinite(coords[a2]).all(axis=1)
    is_unreasonable = finite & (dist > _bond_threshold(atoms.element[a1], atoms.element[a2]))

    out = {"bond_length.frac_unreasonable": float(is_unreasonable.sum() / n_orig)}
    if not verbose:
        return out

    for name, mask in _bond_spatial_masks(atoms, a1, a2, is_polymer).items():
        n_cat = int(mask.sum())
        out[f"bond_length.frac_unreasonable.{name}"] = (
            float(int((is_unreasonable & mask).sum()) / n_cat) if n_cat > 0 else float("nan")
        )
    return out


def _clash_category_masks(atoms, i_idx, j_idx, is_polymer):
    """Upstream `design_metrics.py:184-218`: same-vs-different residue x the two molecule types."""
    chain = _chain_identifier(atoms)
    poly_i, poly_j = is_polymer[i_idx], is_polymer[j_idx]
    both_polymer = poly_i & poly_j
    both_ligand = ~poly_i & ~poly_j
    polymer_ligand = poly_i ^ poly_j
    same_res = (chain[i_idx] == chain[j_idx]) & (atoms.res_id[i_idx] == atoms.res_id[j_idx])
    return {
        "intra_residue_polymer": same_res & both_polymer,
        "intra_residue_ligand": same_res & both_ligand,
        "inter_residue_polymer_polymer": ~same_res & both_polymer,
        "inter_residue_polymer_ligand": ~same_res & polymer_ligand,
        "inter_residue_ligand_ligand": ~same_res & both_ligand,
    }


def compute_clash_stats(atoms, is_polymer, is_crd=None, threshold: float = CLASH_THRESHOLD) -> dict:
    """`clash_metrics`, upstream `design_metrics.py:107-174`.

    A clash is a non-bonded pair closer than `threshold`; a pair whose two atoms are both
    coordinate-conditioned (fixed input, atomworks `C_CRD`) is excluded, because the model did not
    place either of them. The reported number is the fraction of *atoms* that have at least one
    clashing partner, not the fraction of pairs.

    Upstream reaches this through biotite's `CellList`, which raises on non-finite coordinates;
    that behaviour is kept here deliberately, since a grading arm must see a design with unresolved
    atoms fail rather than score it on a subset.
    """
    from biotite.structure import CellList

    n_atoms = atoms.array_length()
    coords = atoms.coord
    key = threshold_key(threshold)

    bonded_keys = np.empty(0, dtype=np.intp)
    if atoms.bonds is not None:
        bonds = atoms.bonds.as_array()
        if len(bonds) > 0:
            b1 = np.minimum(bonds[:, 0], bonds[:, 1]).astype(np.intp)
            b2 = np.maximum(bonds[:, 0], bonds[:, 1]).astype(np.intp)
            bonded_keys = b1 * n_atoms + b2

    neighbors = CellList(atoms, cell_size=threshold).get_atoms(coords, radius=threshold)
    n, max_k = neighbors.shape
    i_rep = np.repeat(np.arange(n), max_k)
    j_flat = neighbors.ravel()
    valid = (j_flat >= 0) & (i_rep < j_flat)
    i_idx, j_idx = i_rep[valid], j_flat[valid]

    if is_crd is None:
        is_crd = np.zeros(n_atoms, dtype=bool)
    keep = ~np.isin(i_idx * n_atoms + j_idx, bonded_keys) & ~(is_crd[i_idx] & is_crd[j_idx])
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    pair_dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=-1) if len(i_idx) else np.empty(0)

    def frac_atoms_clashing(pair_mask):
        min_dists = np.full(n_atoms, np.inf)
        if pair_mask.any():
            np.minimum.at(min_dists, i_idx[pair_mask], pair_dists[pair_mask])
            np.minimum.at(min_dists, j_idx[pair_mask], pair_dists[pair_mask])
        return float((min_dists < threshold).sum()) / n_atoms

    out = {f"frac_atoms_clashing_{key}": frac_atoms_clashing(np.ones(len(i_idx), dtype=bool))}
    masks = (
        _clash_category_masks(atoms, i_idx, j_idx, is_polymer)
        if len(i_idx)
        else {c: np.zeros(0, dtype=bool) for c in CLASH_CATEGORIES}
    )
    for category in CLASH_CATEGORIES:
        out[f"frac_atoms_clashing_{category}_{key}"] = frac_atoms_clashing(masks[category])
    return out


# --- residue / token level ---------------------------------------------------------------------

# atomworks `constants.py:488`.
STANDARD_AA = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)  # fmt: skip

# rfproteina SSEnum values; only equality and run-counting matter here.
SSE_NONE, SSE_HELIX, SSE_SHEET, SSE_LOOP = 0, 1, 2, 3


def _residue_change(atoms):
    """True at every atom where chain, residue id, insertion code or residue name changes."""
    change = np.zeros(atoms.array_length(), dtype=bool)
    if len(change):
        change[0] = True
        for name in ("chain_id", "res_id", "ins_code", "res_name"):
            values = atoms.get_annotation(name)
            change[1:] |= values[1:] != values[:-1]
    return change


def residue_starts(atoms):
    return np.flatnonzero(_residue_change(atoms))


def token_starts(atoms, atomize):
    """atomworks `ml/utils/token.py:18`: a new token at every residue change and at every atomized atom."""
    return np.flatnonzero(_residue_change(atoms) | atomize)


def chain_of_interest(design):
    """rfproteina `utils/chain.py:29-103`. Returns (chain, unambiguous)."""
    chains = design.atoms.chain_id
    unique = np.unique(chains)
    if len(unique) < 2:
        return unique[0], True
    protein = [c for c in unique if design.is_protein[chains == c].all()]
    if len(protein) == 1:
        return protein[0], True
    if not protein:
        return sorted(unique)[0], False
    seq_frac = {c: float(np.mean(design.c_seq[chains == c] != "<M>")) for c in protein}
    least = min(seq_frac.values())
    candidates = [c for c in protein if seq_frac[c] == least]
    if len(candidates) == 1:
        return candidates[0], True
    internal = (design.crd_group != -1) & (design.crd_group != 0)
    crd_frac = {c: float(design.is_crd[chains == c].mean() + internal[chains == c].mean()) for c in candidates}
    least = min(crd_frac.values())
    candidates = [c for c in candidates if crd_frac[c] == least]
    if len(candidates) == 1:
        return candidates[0], True
    return sorted(candidates)[0], False


def compute_amino_acid_composition(design) -> dict:
    """`amino_acid_composition`, rfproteina `metrics/amino_acid_composition.py:26-41`."""
    starts = residue_starts(design.atoms)
    names = design.atoms.res_name[starts]
    input_unknown = (design.c_seq[starts] != "<M>") & (design.c_seq[starts] == "UNK")
    standard = names[np.isin(names, STANDARD_AA)]
    n = len(standard)
    return {
        "n_generated_unknown_residues": float(np.count_nonzero((names == "UNK") & ~input_unknown)),
        "n_standard_residues": float(n),
        "ala_fraction": float(np.mean(standard == "ALA")) if n else float("nan"),
        "val_fraction": float(np.mean(standard == "VAL")) if n else float("nan"),
        "gly_fraction": float(np.mean(standard == "GLY")) if n else float("nan"),
    }


def compute_sequence_adherence(design) -> dict:
    """`sequence_adherence`, rfproteina `metrics/sequence_adherence_metric.py:52-73`."""
    starts = residue_starts(design.atoms)
    conditioned = design.c_seq[starts] != "<M>"
    if not conditioned.any():
        return {"sequence_adherence": float("nan"), "n_violations": 0.0}
    target = design.c_seq[starts][conditioned]
    generated = design.atoms.res_name[starts][conditioned]
    n_violations = int((target != generated).sum())
    return {"sequence_adherence": 1.0 - n_violations / len(target), "n_violations": float(n_violations)}


def compute_sequence_probability_adherence(design) -> dict:
    """`sequence_probability_adherence`, rfproteina `sequence_adherence_metric.py:76-106`.

    Only the exact 0 and 1 constraints are checked inline; upstream pools soft errors post hoc."""
    from tt_bio.rfd4.eval_io import SEQUENCE_PROBABILITY_TOKENS

    starts = residue_starts(design.atoms)
    values = design.seq_prob[starts]
    masks = np.isfinite(values)
    outcomes = design.atoms.res_name[starts][:, None] == np.asarray(SEQUENCE_PROBABILITY_TOKENS)[None, :]
    zero = masks & (values == 0.0)
    one = masks & (values == 1.0)
    n_zero, n_one = int(zero.sum()), int(one.sum())
    v_zero, v_one = int(outcomes[zero].sum()), int((~outcomes[one]).sum())
    return {
        "zero_adherence": 1.0 - v_zero / n_zero if n_zero else float("nan"),
        "one_adherence": 1.0 - v_one / n_one if n_one else float("nan"),
        "n_zero_conditions": float(n_zero),
        "n_zero_violations": float(v_zero),
        "n_one_conditions": float(n_one),
        "n_one_violations": float(v_one),
    }


def _sse_fractions(valid, suffix):
    if len(valid) == 0:
        return {f"{k}{suffix}": float("nan") for k in ("helix_fraction", "sheet_fraction", "loop_fraction", "num_ss_elements")}
    n = len(valid)
    runs = 1 + int(np.count_nonzero(valid[1:] != valid[:-1]))
    return {
        f"helix_fraction{suffix}": float((valid == SSE_HELIX).sum() / n),
        f"sheet_fraction{suffix}": float((valid == SSE_SHEET).sum() / n),
        f"loop_fraction{suffix}": float((valid == SSE_LOOP).sum() / n),
        f"num_ss_elements{suffix}": float(runs),
    }


def _psea(atoms):
    """rfproteina `sse_metrics.py:19-79`: per-residue SSEnum and validity, NONE everywhere on failure."""
    from biotite.structure import annotate_sse

    n_res = len(residue_starts(atoms))
    sse = np.full(n_res, SSE_NONE, dtype=np.int8)
    valid = np.zeros(n_res, dtype=bool)
    try:
        codes = annotate_sse(atoms)
        if len(codes) == n_res:
            sse[codes == "a"] = SSE_HELIX
            sse[codes == "b"] = SSE_SHEET
            sse[codes == "c"] = SSE_LOOP
            valid = codes != ""
    except Exception:
        pass
    return sse, valid


def compute_sse_accuracy(design) -> dict:
    """`sse_accuracy`, rfproteina `sse_metrics.py:186-276`: secondary-structure conditioning
    against P-SEA on the same generated structure. Atom-level, so `n_conditioned_atoms` counts
    atoms that are conditioned and have a valid P-SEA code. No C_SEC column, or nothing
    conditioned, contributes no columns."""
    if design.c_sec is None:
        return {}
    conditioned = design.c_sec != SSE_NONE
    if not conditioned.any():
        return {}
    sse, valid = _psea(design.atoms)
    residue_index = np.cumsum(_residue_change(design.atoms)) - 1
    atom_sse, atom_valid = sse[residue_index], valid[residue_index]
    compare = conditioned & atom_valid
    if not compare.any():
        nan = float("nan")
        return {"overall_accuracy": nan, "helix_recall": nan, "sheet_recall": nan, "loop_recall": nan,
                "n_conditioned_atoms": int(conditioned.sum())}
    target, predicted = design.c_sec[compare], atom_sse[compare]

    def recall(cls):
        sel = target == cls
        return float((predicted[sel] == cls).mean()) if sel.any() else float("nan")

    return {
        "overall_accuracy": float((target == predicted).mean()),
        "helix_recall": recall(SSE_HELIX),
        "sheet_recall": recall(SSE_SHEET),
        "loop_recall": recall(SSE_LOOP),
        "n_conditioned_atoms": int(compare.sum()),
    }


def compute_sse_distribution(design) -> dict:
    """`sse_distribution`, rfproteina `metrics/sse_metrics.py:19-160`.

    P-SEA (biotite `annotate_sse`) gives one code per residue; it is spread to atoms and then read
    at *token* starts, so an atomized residue counts once per atom. Any P-SEA failure leaves every
    residue unassigned, exactly as upstream's bare `except` does."""
    atoms = design.atoms
    sse, valid = _psea(atoms)
    residue_index = np.cumsum(_residue_change(atoms)) - 1
    tokens = token_starts(atoms, design.atomize)
    token_sse = sse[residue_index[tokens]]
    token_valid = valid[residue_index[tokens]]
    coi, _ = chain_of_interest(design)
    coi_mask = token_valid & (atoms.chain_id[tokens] == coi)
    return {**_sse_fractions(token_sse[token_valid], ""), **_sse_fractions(token_sse[coi_mask], "_coi")}


# --- structure comparison (StructureComparisonMetric, no per-group subdivision) -----------------

PROTEIN_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O", "OXT")  # atomworks `constants.py:715`


def _finite(atoms):
    return np.isfinite(atoms.coord).all(axis=-1)


def structure_comparison(gt, gt_align, gt_subset, pred, pred_align, pred_subset, gt_is_polymer=None) -> dict:
    """rfproteina `gt_structure_comparison.py:194-285`, one structure per side.

    With `gt_is_polymer` given, the ground truth's identical subunits and equivalent atoms are
    first matched to the prediction (`eval_symmetry.resolve`, upstream's default
    `resolve_symmetry=("subunit", "atom")`). Each mask is applied to its own side and ANDed with
    finite coordinates there. Kabsch (biotite `superimpose`, gt fixed) on the align atoms, applied
    to the whole prediction; then positional RMSD over the subset atoms. A length mismatch on either
    step returns {} (no column), an empty subset returns NaN, exactly as upstream. `gt_align=None`
    skips alignment."""
    from biotite.structure import rmsd, superimpose

    if gt_is_polymer is not None and gt.array_length() == pred.array_length():
        from tt_bio.rfd4.eval_symmetry import resolve

        gt = gt.copy()
        gt.coord = resolve(gt, pred, gt_is_polymer, gt_align, gt_subset)
    pred_coord = pred.coord.copy()
    if gt_align is not None:
        g = gt_align & _finite(gt)
        p = pred_align & _finite(pred)
        if g.sum() != p.sum() or g.sum() == 0:
            return {}
        _, transform = superimpose(gt.coord[g], pred.coord[p])
        pred_coord = transform.apply(pred_coord)
    g = gt_subset & _finite(gt)
    p = pred_subset & np.isfinite(pred_coord).all(axis=-1)
    if g.sum() != p.sum():
        return {}
    if g.sum() == 0:
        return {"rmsd": float("nan")}
    return {"rmsd": float(rmsd(gt.coord[g], pred_coord[p]))}


def compute_ca_rmsd_to_input(design, input_design) -> dict:
    """`ca_rmsd_to_input`, configs `design_metrics.yaml:109-115`: align and score on protein CA."""
    if input_design is None:
        raise ValueError("ca_rmsd_to_input needs the design's <stem>_input.cif")
    g = input_design.is_protein & (input_design.atoms.atom_name == "CA")
    p = design.is_protein & (design.atoms.atom_name == "CA")
    return structure_comparison(input_design.atoms, g, g, design.atoms, p, p, input_design.is_polymer)


def _protein_sequence(design):
    atoms = design.atoms[design.is_protein]
    if atoms.array_length() == 0:
        return None
    return atoms.res_name[residue_starts(atoms)]


def compute_sequence_similarity_to_input(design, input_design) -> dict:
    """`sequence_similarity_to_input`, rfproteina `design_metrics.py:282-318`."""
    if input_design is None:
        # upstream dereferences the missing ground truth and raises, which drops the design's
        # whole inline block; NaN here would score a design upstream refuses to score
        raise ValueError("sequence_similarity_to_input needs the design's <stem>_input.cif")
    pred = _protein_sequence(design)
    ref = _protein_sequence(input_design)
    if pred is None or ref is None or len(pred) != len(ref):
        return {"sequence_identity_to_input": float("nan")}
    return {"sequence_identity_to_input": float(np.mean(pred == ref))}


def compute_hotspot_obedience(design, max_hotspot_distance=4.5) -> dict:
    """`hotspot_obedience`, rfproteina `metrics/hotspots.py:18-88`, configured at 4.5 A.

    The fraction of hotspot atoms with a generated atom of another chain within the cutoff. Per
    hotspot chain, the candidates are every atom off that chain that is not coordinate-
    conditioned, so a fixed partner chain never counts as the binder. NaN with no hotspots.
    """
    hot = design.hotspot
    if hot is None or not hot.any():
        return {"hotspot_obedience": float("nan")}
    coord = design.atoms.coord
    chains = design.atoms.chain_id
    obeyed = 0
    for chain in np.unique(chains[hot]):
        on_chain = chains == chain
        binder = coord[~on_chain & ~design.is_crd]
        hotspots = coord[hot & on_chain]
        if len(binder) == 0:
            continue
        distance = np.linalg.norm(binder[None, :, :] - hotspots[:, None, :], axis=-1)
        obeyed += int(np.any(distance < max_hotspot_distance, axis=1).sum())
    return {"hotspot_obedience": float(obeyed / hot.sum())}
