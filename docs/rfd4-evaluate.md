# Evaluating RFD4 designs

`tt-bio evaluate --model rfd4` scores designs you already have the way RFD4-Proteina's own
`rfd4-evaluate` does, and writes the same CSV. It needs no device.

```
tt-bio evaluate designs/ --metrics_tags design --out_dir eval/
```

`designs/` is searched recursively for `.cif`, `.cif.gz`, `.cif.zst`, `.bcif` and `.pdb` files. The
result is `eval/evaluation_metrics/per_example_metrics.csv`, one row per design, with upstream's
column names, values rounded to four decimals and an empty cell where upstream writes NaN.

## Flags

| tt-bio | upstream | meaning |
| --- | --- | --- |
| `INPUTS` | `inputs=` | directory of generated structures |
| `--metrics_tags` | `metrics_tags=` | comma-separated tags that switch metrics on |
| `--out_dir` | `out_dir=` | output root |
| `--task_csv` | `task_csv=` | maps `designed_structure_path` to `task`; otherwise `task` is `all` |
| `--job_id`, `--gen_njobs` | `job_id=`, `gen_njobs=` | split the inputs into shards; adds `_job<N>` to the CSV name |

Compare two sets of designs only when both were scored with the same tags: a metric whose tag was not
passed produces no column.

## What is covered

The `design` tag is complete and matches upstream to all four published decimals:

* `clash_metrics`: fraction of atoms with a non-bonded partner under 1.5 Å, overall and split by
  residue and molecule type.
* `bond_length_metrics`: fraction of bonds longer than their element-dependent limit (1.8 Å for
  C/H/N/O, 2.4 Å with P or S, 3.6 Å otherwise), overall and by bond category.
* `amino_acid_composition`: unknown residues the model generated, standard residue count, and the
  alanine, valine and glycine fractions.
* `sequence_adherence` and `sequence_probability_adherence`: whether residues conditioned to an
  identity, or to an exact 0/1 probability, came out that way.
* `sse_distribution` and `sse_accuracy`: helix, sheet and loop fractions and element count from
  P-SEA, and agreement with any secondary-structure conditioning.
* the `unknown_residue_filter` and `sequence_adherence_filter` pre-oracle filters, as
  `pre_oracle_pass`.

The `partial_diffusion` tag adds `ca_rmsd_to_input` and `sequence_similarity_to_input`, both read
against the `<name>_input.cif` written beside the design.

## Refolding

`--metrics_tags design,unconditional_refolding` (or `ppi_refolding`) adds the refolding RMSDs, the
oracle's confidence columns and the pass criteria that read them. This command folds nothing: it
reads the refold the oracle already wrote under

```
<out_dir>/evaluation_samples/<example_id>/<example_id>_<oracle>_refolded.cif
<out_dir>/evaluation_samples/<example_id>/<example_id>_<oracle>_confidences.json
```

which is the same cache upstream resumes from, so scoring the same `out_dir` a second time costs
nothing and a refold from any oracle is scored identically. A design with no refold there is named
in the error rather than written out with the columns missing.

Tags that switch on anything else (motif RMSDs, apo/holo, hotspot and unindexed metrics, CATH,
Rosetta developability, antibody contacts) are refused with the list of what is missing, rather
than producing a shorter CSV that looks complete.

## Behaviour kept from upstream on purpose

If any one metric fails on a design, that design gets none of its metric columns and
`pre_oracle_pass` is 0. Upstream does the same, so the two tools disagree only where the numbers
do. The usual causes are a structure with unresolved atoms (non-finite coordinates) and a generated
CIF copied without the `<name>_input.cif` written next to it.
