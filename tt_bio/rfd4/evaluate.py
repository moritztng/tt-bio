"""`tt-bio evaluate --model rfd4`: upstream's `rfd4-evaluate`, on tt-bio's own stack.

Same inputs (a directory of generated structures, searched recursively), same `metrics_tags`
gating, same output: `<out_dir>/evaluation_metrics/per_example_metrics[_job<N>].csv`, one row per
design, upstream's column names, rounded to 4 decimals with NaN written as an empty cell.

What upstream does and this keeps, on purpose:
* the inline block is all-or-nothing per design. If any inline metric raises, the design gets no
  inline columns and `pre_oracle_pass = 0` (rfproteina `evaluate.py:192-193`). Hiding that would
  make the two arms disagree exactly where a grader needs them to agree.
* a pre-oracle filter whose metric is missing counts as a failure (`pass_criterion_metric.py:131`).

Metrics not ported yet are listed in `NOT_PORTED`; asking for a tag that gates one of them is an
error rather than a silently shorter CSV.
"""

import csv
import hashlib
import logging
import math
from pathlib import Path

from tt_bio.rfd4 import eval_metrics as M
from tt_bio.rfd4.eval_io import read_design

log = logging.getLogger(__name__)

_STRUCTURE_SUFFIXES = (".cif", ".cif.gz", ".cif.zst", ".bcif", ".pdb", ".pdb.gz")


def _gate(any_of=(), all_of=(), none_of=()):
    """rfproteina `metric.py:397-419`: required-any, required-all, prohibited."""
    any_of, all_of, none_of = set(any_of), set(all_of), set(none_of)

    def applies(tags):
        if tags is None:
            return True
        return (
            (not all_of or tags >= all_of)
            and (not any_of or bool(tags & any_of))
            and not (tags & none_of)
        )

    return applies


# Inline metrics in upstream's config order (configs/metrics/inline_metrics/design_metrics.yaml),
# each with its gate and its port.
INLINE = (
    ("amino_acid_composition", _gate(), M.compute_amino_acid_composition),
    (
        "bond_length_metrics",
        _gate(any_of=["design"], none_of=["small_molecule_only", "structure_prediction"]),
        lambda d: M.compute_bond_length_stats(d.atoms, d.is_polymer),
    ),
    (
        "clash_metrics",
        _gate(any_of=["design", "structure_prediction"], none_of=["small_molecule_only"]),
        lambda d: M.compute_clash_stats(d.atoms, d.is_polymer, d.is_crd),
    ),
    ("sequence_adherence", _gate(any_of=["design"]), M.compute_sequence_adherence),
    ("sequence_probability_adherence", _gate(any_of=["design"]), M.compute_sequence_probability_adherence),
    (
        "sse_distribution",
        _gate(any_of=["design"], none_of=["small_molecule_only", "structure_prediction"]),
        M.compute_sse_distribution,
    ),
    (
        "sse_accuracy",
        _gate(any_of=["design"], none_of=["small_molecule_only", "structure_prediction"]),
        M.compute_sse_accuracy,
    ),
)

# Upstream inline metrics this port does not compute yet, with the tags that switch them on.
NOT_PORTED = {
    "hotspot_obedience": _gate(any_of=["hotspot_conditioning"]),
    "rasa_adherence": _gate(any_of=["rasa_conditioning"]),
    "centroid_conditioning_adherence": _gate(any_of=["centroid_conditioning"]),
    "interchain_interactions": _gate(any_of=["interchain_interactions"]),
    "posebusters_metrics": _gate(any_of=["small_molecule_only"]),
    "chiral_accuracy": _gate(any_of=["has_small_molecule"]),
    "design_motif_allatom_rmsd": _gate(
        any_of=["3d_coordinate_conditioning", "unindexed", "internal_coordinate_conditioning"]
    ),
    "design_unindexed_motif_allatom_rmsd": _gate(all_of=["unindexed"]),
    "unindexed_insertion_mapping": _gate(all_of=["unindexed"]),
    "contact_molecular_surface": _gate(any_of=["cms"]),
    "sequence_similarity_to_input": _gate(any_of=["partial_diffusion"]),
    "ca_rmsd_to_input": _gate(any_of=["partial_diffusion"]),
    "cath_metric": _gate(any_of=["cath"]),
    "structure_prediction_lddt": _gate(any_of=["structure_prediction"]),
    "sequence_developability": _gate(any_of=["developability"]),
    "rosetta_developability": _gate(any_of=["developability"]),
    "antibody_contacts": _gate(any_of=["antibody"]),
}

# configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml: (key, gate, {column: max}).
PRE_ORACLE_FILTERS = (
    ("unknown_residue_filter", _gate(), {"amino_acid_composition.n_generated_unknown_residues": 0}),
    ("sequence_adherence_filter", _gate(any_of=["design"]), {"sequence_adherence.n_violations": 0}),
)
PRE_ORACLE_NOT_PORTED = {
    "unindexed_insertion_mapping_filter": _gate(all_of=["unindexed"]),
    "insertion_rmsd_filter": _gate(all_of=["unindexed", "insertion_rmsd_filter"]),
    "motif_rmsd_filter": _gate(all_of=["motif_rmsd_filter"]),
    "ligand_clash_filter": _gate(all_of=["ligand_clash_filter"], none_of=["no_ligand"]),
}

# Tags that switch on the refolding oracles; none of them run in this port yet.
ORACLE_TAGS = {
    "unconditional_refolding", "ppi_refolding", "dna_refolding", "motif_scaffold_refolding",
    "internal_coordinate_motif_scaffold_refolding", "sm_binder_refolding", "ame_refolding",
    "disco_refolding",
}  # fmt: skip


def parse_tags(raw):
    """rfproteina `evaluate.py:69-75`: a string is split on commas, a list is taken as is."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip().strip("[]")
        return [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    return list(raw)


def unsupported(tags):
    tagset = None if tags is None else set(tags)
    names = [n for n, g in {**NOT_PORTED, **PRE_ORACLE_NOT_PORTED}.items() if g(tagset)]
    if tagset is None or tagset & ORACLE_TAGS:
        names.append("refolding oracles")
    return names


def structure_files(inputs):
    root = Path(inputs)
    files = sorted(p for p in root.glob("**/*") if p.is_file() and p.name.lower().endswith(_STRUCTURE_SUFFIXES))
    if not files:
        raise ValueError(f"No structure files found in {inputs}")
    return files


def structure_stem(path):
    name = path.name
    for suffix in sorted(_STRUCTURE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def example_id(path, inputs):
    """rfproteina `evaluate.py:129-134`: stem plus the first 8 hex of sha256(relative path as .cif)."""
    rel = path.relative_to(inputs)
    stem = structure_stem(rel)
    identity = rel.with_name(stem + ".cif")
    return f"{stem}_{hashlib.sha256(str(identity).encode()).hexdigest()[:8]}"


def inline_metrics(design, tags):
    tagset = None if tags is None else set(tags)
    out = {}
    for name, gate, fn in INLINE:
        if gate(tagset):
            out.update({f"{name}.{k}": v for k, v in fn(design).items()})
    return out


def pre_oracle_pass(metrics, tags):
    tagset = None if tags is None else set(tags)
    for _, gate, limits in PRE_ORACLE_FILTERS:
        if not gate(tagset):
            continue
        for column, maximum in limits.items():
            value = metrics.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)) or value > maximum:
                return 0.0
    return 1.0


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(round(value, 4))
    return value


def evaluate(inputs, metrics_tags, out_dir, task_csv=None, job_id=None, gen_njobs=1):
    """Run the ported inline metrics over `inputs` and write upstream's CSV. Returns its path."""
    tags = parse_tags(metrics_tags)
    missing = unsupported(tags)
    if missing:
        raise NotImplementedError(
            "these upstream metrics are switched on by the requested tags but are not ported yet: "
            + ", ".join(missing)
        )
    inputs = Path(inputs)
    files = structure_files(inputs)
    tasks = {}
    if task_csv is not None:
        with open(task_csv, newline="") as fh:
            tasks = {r["designed_structure_path"]: r["task"] for r in csv.DictReader(fh)}
    if job_id is not None:
        chunk = (len(files) + gen_njobs - 1) // gen_njobs
        files = files[job_id * chunk : (job_id + 1) * chunk]
        if not files:
            raise ValueError("No structure files remain after chunking.")

    out_dir = Path(out_dir)
    samples_dir = out_dir / "evaluation_samples"
    metrics_dir = out_dir / "evaluation_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        eid = example_id(path, inputs)
        row = {
            "example_id": eid,
            "dataset_name": inputs.name,
            "task": tasks.get(str(path), "all"),
            "designed_structure_path": str(path),
            "output_base_dir": str(samples_dir),
            "metrics_tags": ",".join(tags) if tags else "",
        }
        try:
            metrics = inline_metrics(read_design(path), tags)
        except Exception as exc:  # upstream drops the whole inline block for this design
            log.warning("Design metrics failed for %s: %s", eid, exc)
            metrics = {}
        row.update(metrics)
        row["pre_oracle_pass"] = pre_oracle_pass(metrics, tags)
        # rfproteina `proteina.py:283-290` fills this from metric keys ending in `chain_of_interest`;
        # only the oracles emit one, so on the inline-only path it is always empty.
        row["chain_of_interest"] = None
        rows.append(row)

    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    suffix = f"_job{job_id}" if job_id is not None else ""
    path = metrics_dir / f"per_example_metrics{suffix}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})
    return path
