---
license: cc-by-4.0
pretty_name: AbAg-XM
size_categories:
- 100K<n<1M
task_categories:
- other
tags:
- biology
- protein-structure
- antibody
- structure-prediction
- benchmark
configs:
- config_name: samples
  default: true
  data_files:
  - split: train
    path: samples/*.parquet
- config_name: targets
  data_files:
  - split: train
    path: targets/targets.parquet
- config_name: structures
  data_files:
  - split: train
    path: structures/*/*.parquet
---

# AbAg-XM

<a href="https://tenstorrent.com"><img src="https://huggingface.co/datasets/Tenstorrent/abag-xm/resolve/main/tt-logo-card.png" alt="Tenstorrent" height="42"></a>

*Computed on [Tenstorrent](https://tenstorrent.com) hardware with [TT-Bio](https://github.com/moritztng/tt-bio).*


335,360 antibody-antigen structure predictions from four independently trained models, every one
scored against the experimental structure with DockQ. 512 samples per target per model, no cell
shallower than 512.

**The targets are [2026ARK-AB](https://arxiv.org/abs/2607.03787), the antibody-antigen benchmark
released with OpenDDE.** 164 PDB targets, 404 interfaces, 159 clusters at 40% MMseqs2 entity
clustering. We did not assemble that set and take no credit for it. What this dataset adds is the
predictions and labels on top of it. **If you use this dataset, cite 2026ARK-AB as well.**

This is the substrate for one finding: **sampling scales and selection does not.** The pool of 512
gets steadily better as you draw more, and the structure a model's own confidence hands you stops
improving after about 32 samples. The analysis is at https://moritztng.github.io/abag-scaling/, and
every interval and control is in
[FINDINGS.md](https://github.com/moritztng/abag-scaling/blob/main/FINDINGS.md).

| model | targets | random | oracle@16 | oracle@512 | delivered@16 | delivered@512 | gap@512 |
|---|---|---|---|---|---|---|---|
| boltz2 | 161 | 0.2246 | 0.3370 | 0.4480 | 0.2449 | 0.2514 | +0.1966 [+0.1654, +0.2301] |
| opendde-abag | 160 | 0.4991 | 0.5613 | 0.6211 | 0.5047 | 0.4978 | +0.1234 [+0.1034, +0.1451] |
| protenix-v2 | 161 | 0.3009 | 0.4164 | 0.5673 | 0.3201 | 0.3191 | +0.2481 [+0.2169, +0.2808] |
| esmfold2 | 161 | 0.2512 | 0.3387 | 0.4431 | 0.2795 | 0.2852 | +0.1579 [+0.1377, +0.1792] |

Mean DockQ. `oracle@k` is the expected best DockQ among k samples, `delivered@k` the expected DockQ
of the one the model's own confidence returns from those k. The oracle is an upper bound computed
with the answer key; it is not achievable. Intervals are paired bootstrap over targets, B=20000.

Reproducing `delivered` needs one convention: where two samples tie on `selector`, the tie is
resolved against the selector, so it is credited with the lower-DockQ one. For the three
co-folders, whose confidence is effectively continuous, that never bites. For esmfold2 it does:
its pLDDT selector is quantised to 4 decimals, which leaves about 181 distinct values per 512 and
a top-selector tie on 20 of the 161 scorable targets. Break those ties the other way and
esmfold2's delivered@512 reads 0.2878 rather than 0.2852.

## Computed on Tenstorrent

All 335,360 folds ran on **[Tenstorrent](https://tenstorrent.com)** hardware, a 32-chip Wormhole Galaxy, using
**[TT-Bio](https://github.com/moritztng/tt-bio)**, our open-source stack for running structure
prediction models on Tenstorrent. Four models, 512 samples per target, 164 targets. At this scale
cost per prediction is what decides whether a study like this is affordable at all, which is most of
why this dataset exists.

## The benchmark this builds on

The 164 targets are **2026ARK-AB**, the antibody-antigen benchmark released with OpenDDE: 164 PDB
targets, 404 antibody-antigen interfaces, 159 clusters at 40% MMseqs2 entity clustering. **We did not
assemble this target set.** It is defined in
[Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine](https://arxiv.org/abs/2607.03787),
and the target list is in [aurekaresearch/OpenDDE](https://github.com/aurekaresearch/OpenDDE) under
`benchmarks/2026ARK_AB/`. If you use this dataset, please cite that work as well.

**AbAg-XM** is our name for the 335,360 predictions and DockQ labels generated on top of that
benchmark, not a name for the benchmark itself. XM is for cross-model: the same targets folded by four
independently trained models, which is what this dataset adds.

## Load it

```python
from datasets import load_dataset

# the scores: 335,360 rows, 17 MB
s = load_dataset("Tenstorrent/abag-xm", split="train")

# the panel: 164 targets with sequences, chain maps and reference structures
t = load_dataset("Tenstorrent/abag-xm", "targets", split="train")

# one target's 512 predicted structures (52 MB), rather than all 34 GB
one = load_dataset("Tenstorrent/abag-xm",
                   data_files="structures/boltz2/9dsg.parquet", split="train")
print(one[0]["cif"][:200])
```

The default config is the scores, so the obvious command does not start a 34 GB download.
Reference structures, fold inputs and MSAs are plain files:

```bash
hf download Tenstorrent/abag-xm --repo-type dataset --include "natives/*" --local-dir .
hf download Tenstorrent/abag-xm --repo-type dataset --include "inputs/9dsg.yaml" --local-dir .
```

## Configs

| config | rows | size | what it is for |
|---|---|---|---|
| `samples` (default) | 335,360 | 17 MB | every (model, target, sample) with its confidence and its DockQ. Reproduces every published number |
| `targets` | 164 | 48 KB | the panel: sequences, chain maps, residue counts, which reference structure each target scores against |
| `structures` | 335,360 | 34.4 GB | the predicted mmCIF text, sharded `structures/{model}/{target}.parquet`. Join to `samples` on `sample_id` |

Plain folders: `natives/` (164 experimental reference structures, 139 MB), `inputs/` (the 164 fold
inputs, byte-identical to what the campaign ran), `msa/` (353 gzipped a3m alignments, 212 MB, one
per distinct chain sequence).

## Schema: `samples`

| column | type | units | meaning |
|---|---|---|---|
| `model` | string | | `boltz2`, `opendde-abag`, `protenix-v2`, `esmfold2` |
| `target` | string | | PDB entry ID, lowercase |
| `chunk` | int8 | | 0-7. The fold job the sample came from; `seed = base + 1000 * chunk` |
| `rank` | int8 | | 0-63 within the chunk |
| `sample_id` | string | | `{target}_c{chunk}_r{rank}`. Joins `structures` |
| `selector` | float32 | | the model's own shipped confidence: mean pLDDT for esmfold2, `confidence_score` otherwise. Rank by this to reproduce what a user gets |
| `confidence_score` | float32 | | AF3-style composite. Null for esmfold2, which has no interface head |
| `ptm` | float32 | | predicted TM-score |
| `iptm` | float32 | | interface pTM. Null for esmfold2 |
| `complex_plddt` | float32 | | mean pLDDT over the complex. Null for esmfold2 |
| `dockq` | float32 | [0,1] | DockQ against the experimental structure. 0.23 acceptable, 0.49 medium, 0.80 high. Null on the 3 unscorable targets |
| `irmsd` | float32 | Å | interface backbone RMSD |
| `lrmsd` | float32 | Å | ligand RMSD |
| `fnat` | float32 | [0,1] | fraction of native contacts recovered |
| `interface_lddt` | float32 | [0,1] | lDDT over interface atom pairs. Null on 4 targets and a few scattered poses, see limitations |
| `cdr_h1_rmsd`, `cdr_h2_rmsd`, `cdr_h3_rmsd` | float32 | Å | per-CDR-loop RMSD after alignment. Null on 4 targets for H1, 5 for H2 and H3, see limitations |
| `epitope_jaccard` | float32 | [0,1] | overlap of predicted and native antigen contact residue sets. Null on the 7 targets with no resolvable native epitope, see limitations |
| `seed` | int32 | | diffusion seed for the chunk |
| `mps` | int8 | chips | chips per fold job. Null where the fleet recorded `auto` |
| `wall_s` | int32 | s | wall time of the 64-sample chunk, not of one sample. All 64 rows of a chunk share it |
| `hardware` | string | | `wh-galaxy` on every row |
| `code_sha` | string | | TT-Bio commit that produced the fold |

`DockQ` is the CAPRI-calibrated composite of `fnat`, `irmsd` and `lrmsd` on the antibody-antigen
interface. `interface_lddt`, the CDR RMSDs and `epitope_jaccard` are our own implementations.
The score columns are stored as float32; the largest deviation from the float64 values the scorer
produced is 7.63e-06 Å, on `lrmsd`.

`structures`: `sample_id`, `chunk`, `rank`, `cif`. The `cif` string is byte-identical to the file
the model wrote.

`targets`: `target`, `chains`, `seq_antigen`, `seq_h`, `seq_l` (null on 2-chain nanobody targets),
`n_res_total`, `n_res_antigen`, `n_res_h`, `n_res_l`, `native_file`, `native_chain1`,
`native_chain2`, `interface`, `chain_map` (JSON, native chain -> model chain as the scorer resolved
it), `msa_a3m` (JSON, chain id -> a3m file), `dockq_scorable`, `note`.

## How it was produced

* **Models.** Boltz-2, Protenix-v2 and OpenDDE-abag are AlphaFold3-style all-atom diffusion
  co-folders; ESMFold2 is a single-sequence folder. All four ran through
  [TT-Bio](https://github.com/moritztng/tt-bio) at commit `e2edf05da6`, stamped in the `code_sha`
  column of every row.
* **Hardware.** A 32-chip Tenstorrent Wormhole Galaxy. `hardware` is `wh-galaxy` on every published
  row.
* **Sampling.** 8 independent fold jobs of 64 samples each per (model, target) = 512.
  `seed = base + 1000 * chunk`, with disjoint per-model bases (opendde-abag 20000, protenix-v2
  30000, boltz2 40000, esmfold2 50000) and no shared seed between models. The seed is a function of
  (model, chunk) and is therefore the same across all 164 targets: independence across targets comes
  from the inputs differing, not the draws differing. The diffusion noise is target-shaped, so no
  two targets receive the same noise.
* **MSAs.** The three co-folders used unpaired MSAs from an offline ColabFold search against
  UniRef30, cached per chain sequence and shipped in `msa/`. The filename is
  `sha256(sequence)[:16] + ".a3m.gz"`, so a chain's alignment is findable from its sequence alone.
  ESMFold2 ran single-sequence with 10 recycling steps and 100 sampling steps; it uses no MSA.
* **Scoring.** DockQ 2.1.3 against the experimental structure in `natives/`, with the resolved chain
  map recorded per target in the `targets` config so a re-score is reproducible.
* **Completeness.** 164 targets x 4 models = 656 cells. 655 ship, every one exactly 512 samples deep
  with 512 distinct structures and no placeholder values. Nothing was padded, truncated or quietly
  filled with fewer samples.

Only the 512-sample rung ships. The campaign's shallower rungs nest inside it: rung k is chunks
`0 .. k/64 - 1` of the same pool and the structures are the same files, so filter `chunk < k/64` to
reconstruct any of them. Shipping them as separate rows would count the same prediction up to four
times.

## Correction applied 2026-08-14

**11,776 `epitope_jaccard` values changed from 0.0 to null.** Where the native antigen chain does not
resolve there is no native epitope to intersect against, and the scorer wrote `0.0` instead of
leaving the value empty. Those rows read as "the model missed the epitope entirely" when the truth is
"not computable". They cover 7 targets (9kwy, 9ly2, 9ly3, 9lz2, 9ull, 9ulm, 9ynx) in all four models.

Rows were selected on the scorer's own `native_epitope_size == 0`, never on the value being zero.
**The exact zeros on other targets are real measurements** and none of them changed, nor did any
other value in any other column: 11,776 cells moved across the whole `samples` config and nothing
else did. There were 44,433 of those real zeros at this commit. The rescore below then filled rows
that had been empty, some of them with real zeros, and the count is now 52,440.

The dataset was public with the old values from about 10:41 UTC on 2026-08-14 until this commit. To
tell which copy you hold, read `epitope_jaccard` on target 9kwy: null is the corrected data, 0.0 is
the old data. Either re-pull, or drop `epitope_jaccard` on those seven targets.

## Rescored secondary metrics, 2026-08-14

**418,271 `interface_lddt`, CDR-RMSD and `epitope_jaccard` values changed from null to a value.**
Nothing else moved. The gaps were never a property of the folds. The labelling ran across two hosts
and one of them had no PyYAML and no ANARCI, so every metric that needs them came back empty on the
cells that host scored, which is why the missing values fell on whole 64-sample jobs rather than on
individual poses. All 111,616 affected predictions were still on disk, so the repair was to rescore
them with the same scripts, the same interpreter and the same structures that produced every value
already in this dataset.

That makes it a fill, and it is asserted as one. Re-running those scorers on 22,368 already-populated
values reproduced all 22,368 bit-for-bit, and the upload moved nothing: `value_changed` is 0 and
`value -> null` is 0 on every column of every file. The fills are 41,472 in boltz2, 249,301 in
esmfold2 and 127,498 in protenix-v2. opendde-abag was already complete and its file is unchanged.

`epitope_jaccard` stays null on the seven targets named above. The rescore does compute it there, as
`0.0`, which is exactly the artifact the correction above removed, so those 14,336 rows are left
empty on purpose.

**The stated reason for the three unscorable targets was also wrong, and is corrected.** This card
said 9ly2, 9ly3 and 9lz2 are 3-way Ab:Ag hetero-hexamers whose antibody-antigen interface the scorer
cannot resolve. The chain map is in fact correct and explicitly declared. The real mechanism is
chemical: 71/71, 78/78 and 48/48 of the antigen-side contact atoms on their declared interface sit on
phosphoserine, which DockQ drops. No value changed with that correction, only the explanation.

Earlier pulls carry the old, ragged columns. To tell which copy you hold, count non-null
`cdr_h3_rmsd` in `samples/esmfold2.parquet`: 81,408 in this data, 15,360 in the old.

## Known limitations

* **164 targets.** Enough to separate the four models' oracle-delivered gaps with intervals that do
  not overlap zero. Not enough to support per-epitope-class or per-germline claims, and not a
  general statement about co-folding beyond antibody-antigen complexes.
* **Three targets carry no DockQ.** 9ly2, 9ly3 and 9lz2 are anti-phosphoepitope antibodies. Every
  contact atom on the antigen side of their declared interface sits on a phosphoserine (71/71, 78/78
  and 48/48), and DockQ scores only standard residues, so the interface has nothing to score. The
  fold input carries unmodified serine at those positions, so the quantity does not exist on the
  prediction side either. Their confidence values ship; `dockq` is null. 161 targets are scorable, in
  every model.
* **`epitope_jaccard` is null on seven targets.** 9kwy, 9ly2, 9ly3, 9lz2, 9ull, 9ulm and 9ynx have
  no resolvable native antigen chain, so there is no native epitope set to compare a prediction
  against and no overlap exists to measure, in any model. Read those nulls as "not computable", not
  as misses, and exclude them from any `epitope_jaccard` aggregate. Exact zeros on the other targets
  are real: the antibody docked somewhere else.
* **One cell is absent.** opendde-abag / 9sbb. Its galaxy folds sit in a pTM 0.668-0.697 basin
  against ~0.91 on a refold of the identical input, DockQ 0.023 against 0.880 under the same fixed
  scorer, and a scan over the whole panel found it the only such case. It is a pipeline artifact,
  not model behaviour, and no published number used it. That is why `samples` has 655 x 512 rows,
  not 656 x 512.
* **The secondary metrics are near-complete, and null where the quantity does not exist.** `dockq`,
  `irmsd`, `lrmsd` and `fnat` are populated on every scorable sample in all four models. Mean
  per-target depth out of 512, for boltz2 / esmfold2 / opendde-abag / protenix-v2: `interface_lddt`
  499.4 / 498.9 / 499.4 / 499.4, `cdr_h1_rmsd` 496.8 / 496.8 / 496.8 / 496.8, `cdr_h2_rmsd` and
  `cdr_h3_rmsd` 496.4 / 496.4 / 496.3 / 496.4, `epitope_jaccard` 490.1 / 490.1 / 490.0 / 490.1. The
  shortfall is named targets, the same ones in every model. `interface_lddt` is null on 9ly2, 9ly3,
  9lz2 and 9mz8, whose native antigen chain does not resolve. The CDR RMSDs are null on 9l9y, 9mnu,
  9msc and 9udq, whose native heavy chain cannot be IMGT-numbered, and H2 and H3 additionally on
  9lwc. `epitope_jaccard` is null on the seven targets above. Beyond those, a thin per-pose residual
  remains, scattered rather than by target: `interface_lddt` on 13 / 108 / 0 / 16 samples and
  `cdr_h1_rmsd` on 439 / 450 / 434 / 449. `cdr_h2_rmsd`, `cdr_h3_rmsd` and `epitope_jaccard` have
  none. Quote these metrics at their own depth, and read the nulls on the named targets as "not
  computable" rather than as misses.
* **512 is a decision cap, not a measured knee.** The oracle's gain per doubling is still positive at
  the top rung, so the ceiling has not saturated. Nothing here says 512 is where sampling stops
  paying.
* **6 of 656 cells were folded by a slightly different engine tree.** The four largest targets
  (9j4c, 9i3p, 9ivj, 9q7y) needed device-memory fixes that had not yet landed on the frozen tree.
  Each cell is single-tree, so no *pool* is internally mixed; the inhomogeneity is strictly between
  cells. The individual fixes are bit-exact at their own gates, which is not the same as whole-tree
  numerical equivalence, so we state it rather than call it cosmetic. The affected cells:
  opendde-abag 9i3p / 9ivj / 9q7y / 9j4c, protenix-v2 9j4c, esmfold2 9j4c.
* **Confidence is the model's own, unmodified.** We did not train, tune or recalibrate a selector.
  `delivered` is what each model ships, which is the point: the gap is a property of released
  models, not of a selector we chose.

## Licence and attribution

The dataset, meaning the score tables, the packaging and the derived metrics, is released under
**CC-BY-4.0**. The material it is built from carries its own terms, all of them permissive:

| component | source | licence |
|---|---|---|
| Boltz-2 predictions | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) | MIT (code and weights) |
| Protenix-v2 predictions | [bytedance/Protenix](https://github.com/bytedance/Protenix) | Apache-2.0 (code and model parameters) |
| OpenDDE-abag predictions | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) | Apache-2.0 |
| ESMFold2 predictions | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) | MIT, subject to the Biohub [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/) |
| `natives/` reference structures | the [Protein Data Bank](https://www.wwpdb.org/) | CC0 1.0. Please cite the original depositors |
| `msa/` alignments | UniRef30, via ColabFold | UniProt content, CC-BY-4.0 |

None of the four model licences restricts redistribution of model outputs. Predictions are
hypotheses, not measurements; treat them as such.

## Citation

If you use this dataset, please cite it:

> Thüning, M. (2026). *AbAg-XM: 335,360 DockQ-labelled antibody-antigen structure predictions
> from four models*. Tenstorrent. https://huggingface.co/datasets/Tenstorrent/abag-xm

```bibtex
@misc{abagxm2026,
  title  = {AbAg-XM: 335,360 DockQ-labelled antibody-antigen structure predictions
            from four models},
  author = {Th\"uning, Moritz},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Tenstorrent/abag-xm},
  note   = {Analysis: https://moritztng.github.io/abag-scaling/}
}
```

The analysis built on this dataset is at
[moritztng.github.io/abag-scaling](https://moritztng.github.io/abag-scaling/); citing the dataset
covers both. The `natives/` reference structures come from the PDB under CC0, so if you use those,
please also cite the original depositors.
