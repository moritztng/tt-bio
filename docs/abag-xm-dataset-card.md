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

{{N_TARGETS}} antibody-antigen complexes ({{N_SCORABLE}} DockQ-scorable) x 3 generators x
50 samples = {{N_SAMPLES_TOTAL}} scored structures.

## What is in it

| file | one row per | contents |
|---|---|---|
| `targets.parquet` | target | sequences, declared interface chains, release date, provenance, leak flags |
| `labels.parquet` | (target, generator, sample) | DockQ, epitope Jaccard, interface lDDT, CDR RMSD, PAE-derived scores, native confidences, DeepRank-Ab |
| `ensembles.parquet` | (target, generator) | condensed 1225-pair similarity matrix, pairwise structural similarity (PSS), basin clustering |
| `leak_audit.parquet` | target | pre-cutoff homology identities (CDR-H3, antigen), best-hit entries, flags |
| `antigen_dedup.parquet` | target | antigen UniProt accession(s), accession multiplicity, duplicate-group and dedup keep flags |
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

Date purity is not homology purity, so we audited that too: every target's CDR-H3 against
ANARCI-numbered SAbDab chains released before each cutoff, and every antigen against all
pre-cutoff PDB protein chains (MMseqs2). 34 of the 164 targets have a pre-2023 complex (16 of
them pre-2021) carrying a full-length 100%-identity CDR-H3 together with a >=92%-identity
antigen: {{LEAK_FLAGGED_PRE2023}}. A generator can have seen those homologs, so read those
targets' numbers as recall-adjacent. For a strict subset, drop them: 130 targets remain
(`leak_flag_pre2023` in `targets.parquet`; per-target identities in `leak_audit.parquet`).
Zero of the 164 targets appear in ABAG-Rank's training set.

## Scorable subset: 161 of 164

9ly2, 9ly3 and 9lz2 are anti-phosphoepitope antibodies: their native interface is carried by
phosphoserine residues, which the DockQ parser discards, so no scorable interface atoms exist.
Their structures, confidences and ranker scores ship normally, but their DockQ labels are
null. Every success-rate table uses the 161 DockQ-scorable targets as its denominator and
says so.

## Accuracy labels are per-interface, never wave-averaged

DockQ is computed for the **declared antibody-antigen interface**, with an explicit chain map, not
averaged over all interfaces in the assembly. This matters more than it sounds.

We measured it against wave-averaged scoring on the same 350 models: seven antigens from PSBench's
`Multimer_7_2024_8_2025` set, 50 of each target's 200 AF3 models sampled with seed 42.
**Wave-averaged DockQ calls 350 of 350 models CAPRI-acceptable (>= 0.23); per-interface DockQ calls
216 of 350.** Across those seven targets the per-interface median spans 0.006 to 0.899, a factor of
152, while the wave-averaged median spans 0.339 to 0.873, a factor of 2.6. Wave-averaging cannot
separate a target whose models are mostly right from one whose models are essentially all wrong,
because the near-rigid intra-antibody interface dominates the average and is modelled well either
way. The two labels agree on rank (pooled Spearman +0.888); they disagree on scale, and scale is
what a threshold reads.

A ranking method tuned on wave-averaged labels is being tuned on the wrong signal.

For the same reason, the per-sample baselines restricted to the **declared chain pair** are the
PAE-derived `pae_ipsae` and `pae_pdockq2`. Use those when you want an interface-level baseline. The
native confidences in the same table (`conf_iptm`, `conf_ptm`, `conf_confidence_score`,
`conf_complex_plddt`) are whole-complex quantities, because that is all any of the three generators
reports per sample, so they are a global-confidence baseline and not an interface one.

## Labels check out against a second DockQ implementation

Every label is DockQ 2.1.3 on the declared interface, and every number in this dataset rests on
that one quantity, so the labels were re-scored with tinyprot, an independent DockQ
implementation: 120 stratified (target, generator, sample) triples spanning the full score range,
Pearson r = 0.99968, median absolute deviation 2.8e-8, and zero disagreements that flip a sample
across the 0.23 or 0.8 thresholds. The cross-validation did expose one real bug, a
chain-assignment error on a single target (9q1l, two generator folds). Those labels were
recomputed and the full-panel re-audit is clean. Full report in `docs/abag-xm-dockq-xval.md` of
the tt-bio repo.

## Validated against the published baseline

The 164 targets are set-identical to OpenDDE's ARK benchmark panel, so their published result is
a direct check on the whole pipeline. Ranked success at N=5 for opendde-abag is 67.1% here
against OpenDDE's published 66.4% on the same targets at their published configuration. The
+0.7pp gap is one target's worth (1/161 = 0.62pp), the 9q1l label correction above, plus
sub-rounding.

## Some targets share antigens

Not every antigen is a distinct protein. 138 of the 164 antigens map to at least one UniProt
accession (the 26 null-mapping ones are engineered constructs, a reported class of their own, not
auto-duplicates), and 24 of those accessions are shared by more than one target, covering 78 of
the 164 targets. The largest group is the SARS-CoV-2 spike (P0DTC2) with 12 targets. Per-target
averages are therefore not fully independent, and popular antigens pull the headline numbers
toward themselves.

Headline metrics are accordingly reported both ways. Ranked success at N=5 for opendde-abag is
0.671 on the full panel, 0.656 with one target per accession (109 scorable targets), and 0.631
under the stricter sequence-level dedup (80 scorable). The full panel stays primary, because
panel identity with ARK is what the baseline validation above rests on; the deduplicated view is
the sensitivity analysis. Per-target accessions, multiplicities and keep flags are in
`antigen_dedup.parquet`; the full audit is `docs/abag-xm-antigen-dedup.md` in the tt-bio repo.

## Limitations worth knowing before you use it

- **MSAs are uniref30-only and unpaired.** No paired MSA, no environmental database. Absolute
  accuracy is therefore *not* comparable to published numbers produced with full ColabFold search;
  comparisons within this dataset are valid because every generator received byte-identical input.
- **Three generators, not a survey.** Chosen for permissive licensing so their outputs can be
  redistributed. The closest sibling set is SCALE (~200k scFv-antigen predictions over
  training-era SAbDab targets); it studies the same oracle-selection gap at larger scale.
  AbAg-XM's targets all postdate every generator's training cutoff and carry a homology
  audit, which is what a generalization claim needs.
- **CoFold Arena is complementary, not comparable.** CoFold Arena (cofoldarena.ai, weekly
  PDB-synced) scores a 5-sample / top-1 / single-seed operating point with each model's own
  confidence ranker; AbAg-XM measures the full oracle-vs-N curve to N=50 and ranker transfer
  on a date-purity-audited panel. Its stricter methodology is adopted here: their paired
  bootstrap with rank ranges is how the ranker intervals are computed, their
  one-antibody-per-antigen panel rule is the dedup audit above, and their scorer (tinyprot)
  is the label cross-validator above. Percentages are not comparable across the two.
- **50 samples per fold** bounds how well the oracle gap can be estimated for any single target.
- **`fold_seq_light` is null for 59 of the targets**, which is a fact about the construct and not
  missing data: those are heavy-only or single-chain antibodies. It is null exactly when `has_HL`
  is false. `fold_seq_antigen` and `fold_seq_heavy` are populated for every target.
- Fold provenance (`host`, `tt_bio_commit`, `host_threads`, `paired_msa`, `mps`) rides on every
  label row. 437 folds batched 5 samples per pass, 55 batched 3; the sampling path is identical
  either way, and `mps` is recorded so you can stratify on it. Rows produced under a recovered or
  non-standard configuration are flagged; filter on provenance if your analysis is sensitive to it.

## Licence and attribution

The dataset is released under **CC-BY-4.0**.

Structures were produced with Boltz-2 (MIT), Protenix-v2 (Apache-2.0) and OpenDDE (Apache-2.0).
Ground-truth structures are from the PDB (public domain). If you use AbAg-XM, please cite those
predictors alongside this dataset.

Generated on Tenstorrent Blackhole accelerators with [tt-bio]({{TT_BIO_URL}}).

## Reproducing it

The generation, labelling and scoring scripts live in tt-bio under `scripts/abag_xm_*`. Every fold
records the commit that produced it, so any row can be traced to the exact code path.
