<!-- This file becomes README.md in the HuggingFace dataset repo. It is the dataset card.
     Values written in double braces require the finished slab and must be filled from the
     generated artifacts before publication. Do not estimate them. -->
---
license: cc-by-4.0
task_categories:
  - other
tags:
  - biology
  - protein-structure
  - antibody
  - model-ranking
pretty_name: AbAg-XM
size_categories:
  - 10K<n<100K
---

# AbAg-XM: cross-model antibody-antigen co-folding ensembles

Most antibody-antigen structure prediction is scored one model at a time, against one predictor.
AbAg-XM is built for the question underneath that: **given many samples from several predictors,
can you tell which one is right?** Each target is folded 50 times by each of three open predictors,
and every sample carries both its predicted confidence and its measured accuracy, so a ranking
method can be tested against ground truth rather than against another prediction.

{{N_TARGETS}} antibody-antigen complexes x 3 generators x 50 samples =
{{N_SAMPLES_TOTAL}} scored structures.

## What is in it

| file | one row per | contents |
|---|---|---|
| `targets.parquet` | target | sequences, declared interface chains, release date, provenance |
| `labels.parquet` | (target, generator, sample) | DockQ, epitope Jaccard, interface lDDT, CDR RMSD, PAE-derived scores, native confidences |
| `ensembles.parquet` | (target, generator) | condensed 1225-pair similarity matrix, pairwise structural similarity (PSS), basin clustering |
| `structures/<generator>/<target>/` | sample | gzipped mmCIF coordinates |
| `pae/<generator>/<target>/` | sample | predicted aligned error, float16 |

Generators: **Boltz-2**, **Protenix-v2**, **OpenDDE**.

```python
from datasets import load_dataset
labels = load_dataset("{{HF_REPO}}", data_files="labels.parquet")["train"]
```

Coordinates are ~50,000 files. Fetch a subset with `huggingface_hub.snapshot_download(...,
allow_patterns="structures/boltz2/9abc/*")` rather than cloning the whole repo.

## The temporal split is the point

Every target was released after the training cutoff of every generator (2021 and 2023 cutoffs
against 2026 target releases). Nothing here can have been memorised, which is what makes the
accuracy labels usable as a ranking benchmark rather than a recall test.

## Accuracy labels are per-interface, never wave-averaged

DockQ is computed for the **declared antibody-antigen interface**, with an explicit chain map, not
averaged over all interfaces in the assembly. This matters more than it sounds.

We measured it against wave-averaged scoring on the same 200 models, across four antigens from
PSBench's `Multimer_7_2024_8_2025` set. **Wave-averaged DockQ calls 200 of 200 models
CAPRI-acceptable (>= 0.23); per-interface DockQ calls 106 of 200.** Across those four targets the
per-interface median spans 0.006 to 0.899, a factor of 151, while the wave-averaged median spans
0.517 to 0.873, a factor of 1.7. It cannot separate a target whose models are mostly right from one
whose models are essentially all wrong, because the near-rigid intra-antibody interface dominates
the average and is modelled well either way.

A ranking method tuned on wave-averaged labels is being tuned on the wrong signal.

For the same reason, the per-sample baselines restricted to the **declared chain pair** are the
PAE-derived `pae_ipsae` and `pae_pdockq2`. Use those when you want an interface-level baseline. The
native confidences in the same table (`conf_iptm`, `conf_ptm`, `conf_confidence_score`,
`conf_complex_plddt`) are whole-complex quantities, because that is all any of the three generators
reports per sample, so they are a global-confidence baseline and not an interface one.

## Limitations worth knowing before you use it

- **MSAs are uniref30-only and unpaired.** No paired MSA, no environmental database. Absolute
  accuracy is therefore *not* comparable to published numbers produced with full ColabFold search;
  comparisons within this dataset are valid because every generator received byte-identical input.
- **Three generators, not a survey.** Chosen for permissive licensing so their outputs can be
  redistributed.
- **50 samples per fold** bounds how well the oracle gap can be estimated for any single target.
- **`fold_seq_light` is null for 59 of the targets**, which is a fact about the construct and not
  missing data: those are heavy-only or single-chain antibodies. It is null exactly when `has_HL`
  is false. `fold_seq_antigen` and `fold_seq_heavy` are populated for every target.
- Per-sample provenance (`host`, `tt_bio_commit`, `host_threads`, `paired_msa`, sampling path) is
  recorded in `labels.parquet`. Rows produced under a recovered or non-standard configuration are
  flagged; filter on provenance if your analysis is sensitive to it.

## Licence and attribution

The dataset is released under **CC-BY-4.0**.

Structures were produced with Boltz-2 (MIT), Protenix-v2 (Apache-2.0) and OpenDDE (Apache-2.0).
Ground-truth structures are from the PDB (public domain). If you use AbAg-XM, please cite those
predictors alongside this dataset.

Generated on Tenstorrent Blackhole accelerators with [tt-bio]({{TT_BIO_URL}}).

## Reproducing it

The generation, labelling and scoring scripts live in tt-bio under `scripts/abag_xm_*`. Every fold
records the commit that produced it, so any row can be traced to the exact code path.
