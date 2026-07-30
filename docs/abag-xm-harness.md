# AbAg-XM harness: fold, label, score, release

The harness turns a set of antibody-antigen targets into a ranking benchmark: every target is
folded 50 times per generator, every sample is labelled against the native structure, and all
ranker scores are merged into one table. It produced the AbAg-XM dataset (see
`docs/abag-xm-dataset-card.md`); use it to reproduce a fold, score your own ranker against the
released labels, or add a new generator.

## Inputs

- `examples/abag_xm/<pdb_id>.yaml` — one fold definition per target: the antigen chain `A` and
  the antibody chains (`H`, `L` where present), sequences only.
- `docs/implementation-parity-data/abag-xm-targets.parquet` — the target manifest: declared
  interface chains, cluster assignments, release dates, flags.
- Ground-truth mmCIFs, one per target, fetched from the PDB. Not in git; see the dataset card
  for the released copy.
- Tool pins that matter for label identity: DockQ 2.1.3, ANARCI (IMGT numbering), MMseqs2 18.
  Full pin list in `docs/PROVENANCE.md`.

## Pipeline

**1. Generate (device).** One fold per target per generator:

```bash
tt-bio predict examples/abag_xm/9lz0.yaml --model boltz2 --diffusion_samples 50 \
    --msa_dir ~/abag_xm/msa_cache --msa_db_path ~/.boltz/msa_db \
    --seed 42 --override --write_pae
```

`scripts/abag_xm_generate.py` drives all 164 targets and appends one JSON line per fold to
`progress.jsonl` so a restart skips finished work. Every generator received byte-identical MSAs
(the shared cache), which is what makes cross-generator label comparisons valid.

**2. Label (CPU).** Per fold:

```bash
python3 scripts/abag_xm_labels.py <results_dir> <native.cif> examples/abag_xm/<target>.yaml \
    --out ~/abag_xm/tier_a/labels/<gen>_<target>.json
```

Per sample this computes DockQ on the declared interface from the manifest (never a
wave-average over the assembly — the card explains why), epitope Jaccard, interface lDDT,
per-CDR RMSD, and PAE-derived scores; per fold it adds the 50x50 pairwise DockQ/TM matrix,
PSS, and basin clustering. `scripts/abag_xm_label_census.py` audits every null label and
assigns it a cause; `docs/abag-xm-label-census.md` is the current census.

**3. Score rankers (CPU).**

```bash
python3 scripts/abag_xm_ranker_scores.py --all --with_deeprank \
    --out ~/abag_xm/tier_a/ranker_scores.csv
```

Joins native confidences, the PAE-derived rankers (`pdockq2`, `ipsae`, `anticonf`), PSS, the
learned ranker DeepRank-Ab (Apache-2.0; cached under `tier_a/deeprank_json_cache`), and the
oracle labels into one row per sample. To score your own ranker, join on
`(target, gen, rank)` — `rank` is confidence-ordered, rank 0 is the generator's top-1.

**4. Release tables + assembly.**

```bash
python3 scripts/abag_xm_publish.py --out_dir ~/abag_xm/release   # assemble + preflight only
```

Builds `labels.parquet` / `ensembles.parquet` / `targets.parquet` / `leak_audit.parquet`,
stages coordinates and PAE, fills the dataset card, and runs preflight. Upload is a separate
gated step (`--go`) and is not part of the harness.

## Adding a generator

A generator is anything that satisfies the output contract below for each fold yaml; it does
not need to be a tt-bio model, but the tt-bio `predict` CLI already emits this layout, so a
new `tt_bio` model adapter is the shortest path.

Per target, produce a results directory `<gen>_results_<target>/` containing:

- `structures/<target>_model_<rank>.cif` — one mmCIF per sample, named by confidence rank
  (rank 0 = the sample the generator itself would pick).
- `structures/<target>_model_<rank>_pae.npz` — per-sample PAE (needed for `pdockq2`/`ipsae`;
  folds without it still get DockQ/sequence labels).
- `results.json` — a one-element list; the entry carries `status` and `all_runs`, a list of
  per-sample dicts with `rank`, `confidence_score` (the number used for ranked success), and
  ideally `iptm`, `ptm`, `complex_plddt` (the native-confidence baselines).

Then register the directory-prefix mapping in the scripts that translate folder names to
generator names (`abag_xm_ranker_scores.py`, `abag_xm_build_release_tables.py`,
`abag_xm_stage_release.py`, `abag_xm_label_patch.py`: the `GEN_DIRS`/`DIR_TO_GEN` dicts), and
run phases 2-4. Labels are generator-agnostic — nothing in phase 2 changes.

`scripts/abag_xm_acceptance.py` is the integrity gate: it checks every fold's record
(structures present, PAE count, provenance fields) and exits non-zero on any outstanding
fold. Run it before trusting a merged table.
