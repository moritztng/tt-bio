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

335,360 antibody-antigen structure predictions from four independently trained models, every one
scored against the experimental structure with DockQ. 164 targets, 512 samples per target per
model, no cell shallower than 512.

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

## Schema — `samples`

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
| `interface_lddt` | float32 | [0,1] | lDDT over interface atom pairs. Partial coverage, see limitations |
| `cdr_h1_rmsd`, `cdr_h2_rmsd`, `cdr_h3_rmsd` | float32 | Å | per-CDR-loop RMSD after alignment. Partial coverage |
| `epitope_jaccard` | float32 | [0,1] | overlap of predicted and native antigen contact residue sets. Partial coverage |
| `seed` | int32 | | diffusion seed for the chunk |
| `mps` | int8 | chips | chips per fold job. Null where the fleet recorded `auto` |
| `wall_s` | int32 | s | wall time of the 64-sample chunk, not of one sample. All 64 rows of a chunk share it |
| `hardware` | string | | `wh-galaxy` on every row |
| `code_sha` | string | | tt-bio commit that produced the fold |

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
  [tt-bio](https://github.com/moritztng/tt-bio) at commit `e2edf05da6`, stamped in the `code_sha`
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
* **One cell is absent.** opendde-abag / 9sbb. Its galaxy folds sit in a pTM 0.668-0.697 basin
  against ~0.91 on a refold of the identical input, DockQ 0.023 against 0.880 under the same fixed
  scorer, and a scan over the whole panel found it the only such case. It is a pipeline artifact,
  not model behaviour, and no published number used it. That is why `samples` has 655 x 512 rows,
  not 656 x 512.
* **The secondary metrics are ragged, badly so for esmfold2.** `dockq` and `irmsd` are populated on
  every scorable sample in all four models. `interface_lddt`, `epitope_jaccard` and the CDR RMSDs
  are not: mean per-target depth for `cdr_h3_rmsd` is 419.8 for boltz2, 505.6 for opendde-abag,
  330.0 for protenix-v2 and **95.4 for esmfold2** out of 512. Quote those metrics at their own
  depth. No epitope or CDR claim from this dataset is a claim at n=512.
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
