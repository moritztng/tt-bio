"""The RFD4 design-level metric set, per feature family, taken from upstream.

Every column name below is upstream's, read off RFD4-Proteina at the pinned sha
2db07bb444d91695f8c10e6acb7648e9297f9221:

    configs/metrics/inline_metrics/design_metrics.yaml   (design metrics, no oracle)
    configs/metrics/refolding_metrics/all_refolding_metrics.yaml
    configs/metrics/refolding_oracles/{af2,af3,of3}.yaml
    configs/metrics/pass_criteria/all_pass_criteria.yaml
    docs/reference/metrics.md, docs/how-to/cookbook.md  (which tag turns which metric on)

`rfd4-evaluate inputs=<dir> metrics_tags='[...]' out_dir=<dir>` writes
`evaluation_metrics/per_example_metrics.csv`, one row per design, one column per metric. That CSV
is this harness's input on both arms: the TT side computes the same columns from the same
upstream code, with the oracle running on Tenstorrent (rfd4-beam's wiring). Nothing here
re-implements an upstream metric; the `geom.*` columns are the harness's own and are marked as
such, they exist so a grade is still possible before an oracle is wired up.

kind:
  dist  -- continuous, graded distributionally with a KS test (bar: p >= 0.05)
  rate  -- a 0/1 pass column, graded with a one-sided Fisher exact test (bar: p >= 0.05)
  guard -- a hard threshold on the TT arm alone, no reference comparison
oracle: True means the column only exists once a refolding oracle has run.
"""
from __future__ import annotations

# Tags to pass to `rfd4-evaluate` so the columns a family needs actually get computed.
# A metric whose tag is missing is skipped silently upstream (docs/how-to/refolding.md), which is
# why the harness fails a family whose columns are absent rather than treating them as "no data".
FEATURE_TAGS = {
    "unconditional": ["design", "unconditional_refolding"],
    "binder": ["design", "ppi_refolding", "hotspot_conditioning"],
    "antibody_cdr": ["design", "antibody"],
    "nanobody_cdr": ["design", "antibody"],
    "enzyme": ["design", "ame_refolding", "motif_scaffold_refolding"],
    "motif_scaffold": ["design", "motif_scaffold_refolding", "3d_coordinate_conditioning"],
    "small_molecule": ["design", "sm_binder_refolding", "interchain_interactions", "has_small_molecule"],
    "nucleic_acid": ["design", "dna_refolding"],
    # `partial_diffusion` is what gates sequence_similarity_to_input and ca_rmsd_to_input, the
    # comparisons against the structure that was partially diffused (sequence recovery, for an
    # inverse-folding run). It gates no refold, so a family that grades a refold column must also
    # carry a refolding tag -- without it the refold never runs and the column reads `missing`.
    "inverse_folding": ["design", "partial_diffusion"],
    "partial_diffusion": ["design", "partial_diffusion", "unconditional_refolding"],
}

# Graded on every family: cheap, no oracle, and they are what a broken port breaks first.
COMMON = [
    ("clash_metrics.frac_atoms_clashing_1p5A", "dist", False),
    # Upstream emits this one with the metric's own name repeated: the class is
    # BondLengthMetric, its sub-metric key is `bond_length`. Verified against a real
    # `rfd4-evaluate metrics_tags=[design]` CSV, not from the docs, which drop the middle part.
    ("bond_length_metrics.bond_length.frac_unreasonable", "dist", False),
    ("amino_acid_composition.ala_fraction", "dist", False),
    ("amino_acid_composition.gly_fraction", "dist", False),
    ("sse_distribution.helix_fraction", "dist", False),
    ("sse_distribution.sheet_fraction", "dist", False),
    ("sse_distribution.num_ss_elements", "dist", False),
    ("geom.radius_of_gyration", "dist", False),
    ("geom.frac_ca_bond_outlier", "dist", False),
    ("geom.ca_clash_frac", "dist", False),
    # Emitted by the evaluate pipeline itself, not by a metric config: 1 if the design cleared
    # every pre-oracle filter its tags activated. A filter a design cannot satisfy fails every
    # design and upstream then skips ALL downstream metrics, so a CSV with no metric columns is
    # usually this and not a missing config. Graded as a rate, because a port whose designs stop
    # clearing the filters is worse even when every surviving metric looks fine.
    ("pre_oracle_pass", "rate", False),
    ("amino_acid_composition.n_generated_unknown_residues", "guard", False),
    ("geom.n_nonfinite_coords", "guard", False),
    ("geom.chain_not_found", "guard", False),
]

# Family-specific, on top of COMMON.
SPECIFIC = {
    "unconditional": [
        ("refolded_backbone_rmsd.rmsd", "dist", True),
        # There is no bare `plddt`: the oracle emits overall_plddt (whole prediction) and
        # plddt_coi (chain of interest). A monomer's grade wants the whole prediction.
        ("oracle.overall_plddt", "dist", True),
        ("unconditional.pass", "rate", True),
    ],
    "binder": [
        ("oracle_ppi.ptm_coi", "dist", True),
        ("oracle_ppi.iptm_coi", "dist", True),
        ("oracle_ppi.min_ipae_coi", "dist", True),
        ("refolded_target_backbone_aligned_binder_rmsd.rmsd", "dist", True),
        ("hotspot_obedience.hotspot_obedience", "dist", False),
        ("ppi.pass", "rate", True),
    ],
    "antibody_cdr": [
        ("oracle_ppi.min_ipae_coi", "dist", True),
        ("refolded_target_aligned_binder_rmsd.rmsd", "dist", True),
        ("antibody_contacts.cdr_contact_fraction", "dist", False),
        ("antibody.pass", "rate", True),
    ],
    "enzyme": [
        ("refolded_backbone_rmsd.rmsd", "dist", True),
        ("refolded_motif_allatom_rmsd.rmsd", "dist", True),
        ("oracle.min_ipae_coi", "dist", True),
        ("clash_metrics.frac_atoms_clashing_inter_residue_polymer_ligand_1p5A", "dist", False),
        ("ame.pass", "rate", True),
    ],
    "motif_scaffold": [
        ("design_motif_allatom_rmsd.rmsd", "dist", False),
        ("refolded_backbone_rmsd.rmsd", "dist", True),
        ("refolded_motif_allatom_rmsd.rmsd", "dist", True),
        ("refolded_motif_ca_rmsd.rmsd", "dist", True),
        ("motif_scaffold.pass", "rate", True),
    ],
    "small_molecule": [
        ("oracle.iptm_coi", "dist", True),
        ("oracle.min_ipae_coi", "dist", True),
        ("refolded_backbone_rmsd.rmsd", "dist", True),
        ("refolded_backbone_aligned_ligand_rmsd.rmsd", "dist", True),
        ("chiral_accuracy.tetrahedral_percent_correct_chirality", "dist", False),
        ("sm_binder.pass", "rate", True),
    ],
    "nucleic_acid": [
        ("oracle_dna.min_ipae_coi", "dist", True),
        ("refolded_target_aligned_binder_rmsd.rmsd", "dist", True),
        ("dna.pass", "rate", True),
    ],
    "inverse_folding": [
        ("sequence_similarity_to_input.sequence_identity_to_input", "dist", False),
        ("ca_rmsd_to_input.rmsd", "dist", False),
        ("sequence_adherence.sequence_adherence", "dist", False),
    ],
    "partial_diffusion": [
        ("sequence_similarity_to_input.sequence_identity_to_input", "dist", False),
        ("ca_rmsd_to_input.rmsd", "dist", False),
        ("refolded_backbone_rmsd.rmsd", "dist", True),
    ],
}
SPECIFIC["nanobody_cdr"] = SPECIFIC["antibody_cdr"]

# A guard is a bar on the TT arm by itself: a reference arm that is equally broken must not
# excuse it. max_value is inclusive.
GUARDS = {
    "amino_acid_composition.n_generated_unknown_residues": {"max_mean": 0.0},
    "geom.n_nonfinite_coords": {"max_mean": 0.0},
    # Upstream named a chain of interest the design file does not contain. The harness keeps the
    # whole structure so the design is still graded, but every geom.* column then describes the
    # complex rather than the binder, which is not the measurement anyone asked for.
    "geom.chain_not_found": {"max_mean": 0.0},
}

# Designs that are all the same design are not 30 designs. Diversity is graded in quality.py
# against the reference arm with the same one-sided Fisher exact a pass rate gets; it carries no
# constant of its own beyond P_BAR, because unique_frac quantizes in steps of 1/n.

P_BAR = 0.05
N_MIN = 30


def columns_for(feature: str) -> list[tuple[str, str, bool]]:
    if feature not in SPECIFIC:
        raise KeyError(f"unknown feature family {feature!r}; known: {sorted(SPECIFIC)}")
    return COMMON + SPECIFIC[feature]


def oracle_columns(feature: str) -> list[str]:
    return [c for c, _, needs_oracle in columns_for(feature) if needs_oracle]


def geometry_columns(feature: str) -> list[str]:
    return [c for c, _, _ in columns_for(feature) if c.startswith("geom.")]
