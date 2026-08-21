# Nesso-1

Nesso-1 predicts protein-ligand binding affinity. It is a coarse-grained model: there is no
structure module and no atom-resolution decoder, so it does not produce coordinates. The only
3D quantity it predicts is a soft distogram, and both the pocket selection and the affinity head
read that distogram rather than a pose. If you want a structure, fold with Boltz-2; if you want
an affinity number quickly, this is the cheaper model.

Upstream is [recursionpharma/nesso](https://github.com/recursionpharma/nesso) (Apache-2.0),
weights `recursionpharma/nesso` on the Hub. The host featurization pipeline is vendored under
`tt_bio/_vendor/nesso/`; the model itself is `tt_bio/nesso1.py`.

## Status

The torch reference is bit-exact against upstream, and `tt-bio affinity` reproduces the committed
upstream scalars end to end from a YAML. On device the model is deterministic run to run: the
solo-vs-solo delta is exactly 0.0 at every size and in both precisions.

Speed is settled, and it is not a win against a GPU. The device work is 6.50 s at 512 aa against
1.05 s of H200 device time, so the port sits **6.19x off an H200** and does not clear the 4x gap
tt-bio holds itself to. That is a floor rather than a first attempt: the workload is tt-bio's own
tuned pair-only pairformer block run 304 times per prediction, measured on the same block class and
the same lever settings the fold runs, and the last named lever left (the trimul tail fused at
`k_tiles=4`) came out bit-exact and 1.74x slower. End to end one prediction is 8.4 s at 532 tokens
on one p150a.

Two of the shipped fused kernels do not serve this path at all: the triangle-attention persistent
mask needs a batch-broadcast bias and the affinity stack builds one bias per row, and the trimul
tail fusion refuses this block's contraction width. Both refuse for the same reason at every size,
and the floor above already reflects that, so it is a description of the workload rather than
headroom to go and get.

It is still the right model to run here, because the alternative is much worse. Scoring the same
protein-ligand pair through Boltz-2 affinity costs 386.25 s per prediction at 512 aa on the same
card, so Nesso-1 is **59x cheaper for the same question** and still 3.35x cheaper than Boltz-2
affinity on an H200. Pick it for the cost of the answer on this hardware, not because it beats a
GPU.

## Does it rank binders

Yes, and by the same margin as the reference implementation. On DAVIS (kinase inhibitor Kd from
Therapeutics Data Commons, non-censored measurements only), within-target Pearson of the predicted
affinity against pKd, 30 compounds per target on one Blackhole p150a:

| target | Pearson | Spearman | MW-only control |
|---|---|---|---|
| ABL1p (1167 aa) | 0.732 | 0.594 | 0.172 |
| YSK4 (1328 aa) | 0.593 | 0.636 | 0.178 |
| mean | **0.662** | 0.615 | 0.175 |

The same protocol on an H200 with the upstream implementation gives 0.636 mean Pearson against the
same 0.175 control, so the port ranks compounds as well as the reference does. Per compound the two
arms agree to 0.987-0.994 correlation with a worst single-ligand difference of 0.37, which also
covers a conformer difference: each arm embeds its own ligand with ETKDG.

Within-target is the metric the technical report uses. Pooling across targets would mostly measure
the between-target offset, which is not the ranking task. Censored measurements are excluded because
DAVIS reports every non-binder as exactly 10000 nM, which would turn Pearson into a statement about
how many ties there are.

`scripts/nesso1_port/davis_validate.py` reproduces it; the selection is a pure function of
`davis.csv`, so the compounds are the same ones the reference scored, in the same order.

## Which trunk precision

bf16 is the default. The table is the worst of eleven output scalars, device against the torch CPU
fp32 reference on the same input, divided by upstream's own 0.058 run-to-run spread on the same
scalar (so 1.0 means "as far off as upstream is from itself"):

| tokens | target | fp32 | bf16 | bf16 speedup |
|---|---|---|---|---|
| 61 | tyr48 | 1.30 | 3.17 | 2.8x |
| 148 | CDK2 128 aa | 0.15 | 2.38 | 4.4x |
| 276 | CDK2 256 aa | 0.18 | 0.88 | 6.5x |
| 337 | AURKC + DAVIS binder | 2.86 | 3.01 | 5.9x |
| 532 | CDK2 512 aa | 2.04 | 1.13 | 6.1x |
| 1044 | CDK2 1024 aa | out of DRAM | 25.5 s | - |

Two things decide it. From 276 tokens up bf16 is not the worse arm: it wins outright at 532, and at
337 the two are within 5% of each other on the worst scalar while bf16 is closer on nine of the
eleven (both arms are offset the same way there, which is a device-vs-CPU difference and not a dtype
one). And fp32 cannot be the default everywhere regardless of accuracy, because at 1044 tokens it
asks for a 20.6 GB DRAM buffer against a 4.28 GB bank and dies.

Below ~150 tokens fp32 is clearly the more faithful arm (0.15 against 2.38 xR at 148 tokens) and it
only costs a few seconds there, so `--trunk fp32` is the right choice for a small complex you are
reporting rather than ranking.

## Running it

```bash
tt-bio affinity complex.yaml                       # one input
tt-bio affinity ligands/ --out_dir screen          # a directory: model stays resident
tt-bio affinity complex.yaml --trunk fp32          # more faithful under ~150 tokens, ~6x slower
tt-bio affinity complex.yaml --accelerator cpu     # the torch reference
```

Writes `<id>_affinity.json` per input and one `affinity.csv` for the run, plus a `processed/`
directory holding the parsed structures, the RDKit conformers and the ESM-2 embeddings so a
re-run skips them.

## What it takes as input

One or more protein chains plus one or more ligands, and a `properties.affinity.binder` naming
which ligand to score.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAA
  - ligand:
      id: B
      smiles: N[C@@H](Cc1ccc(O)cc1)C(=O)O
properties:
  - affinity:
      binder: B
```

Supported:

| | |
|---|---|
| protein chains | one or more; identical copies via `id: [A, B]` |
| ligands | one or more, as `smiles:`, `ccd:`, `sdf:`, or a pickled RDKit conformer via `conformer:` |
| ESM embedding | computed for you, or supply a precomputed one with `esm:` |
| output | an affinity value (mean of two ensemble members, plus each member), a binary binder logit and probability, and six distogram entropies |

Not supported, and these are all upstream limits rather than porting gaps:

- **No nucleic acids.** The schema admits protein and ligand entities only.
- **No covalent ligands or inter-chain bonds.** Bonds only ever come from within an RDKit
  molecule; there is no `constraints:` or `bonds:` block.
- **No modified or non-standard residues.**
- **No pocket or hotspot conditioning.** The key parses, but the shipped checkpoint has no
  weights for it, so it would be read and discarded.
- **No batching.** One prediction at a time, structurally.

`version: 1` is required, as is exactly one of `smiles`/`ccd`/`sdf` per ligand. Bad SMILES, a
missing SDF, a binder that names a protein chain, an unknown entity kind and an unrecognized
residue code all raise rather than degrade.

## Three things that will bite you when comparing numbers

**Featurization samples.** `center_random_augmentation` applies a random roto-translation to
every ligand conformer, drawn from the global torch RNG. Two runs of the same input on the same
machine give affinity values up to ~0.06 apart, and that is upstream behaviour, not a defect.
Any comparison between two implementations has to share the draw — seed immediately before
featurizing — or it is measuring the draw rather than the difference.

**RDKit version changes the input.** ETKDG conformer coordinates moved by up to 1.85 A between
RDKit 2025.09.6 and 2026.03.5 for the same ligand with the same atom order, which moved the
affinity value by 0.0007. Pickled molecules are not forward-compatible either: 2025.09.6 cannot
read a mol pickled by 2026.03.5. For a reproducible comparison, feed a committed conformer
(`conformer:` or `sdf:`) rather than regenerating it.

**`--num_workers` changes the ligand.** RDKit's ETKDG takes its embedding seed from the process
RNG state, so parsing a SMILES ligand in a worker process and parsing it inline give the same atoms
in the same order with different coordinates. Keep `--num_workers` fixed across runs you intend to
compare, or feed a committed conformer.

`tt-bio affinity` pins the featurization seed by default so a screen is repeatable, which upstream
is not. `--seed -1` restores upstream behaviour.

## Checking the port

Three gates, all runnable from a clean checkout. The first two need neither a Tenstorrent card
nor an upstream install.

```bash
# host featurization, bit-exact against a committed upstream capture
python scripts/nesso1_port/parity_gate.py

# the torch model: per-module activations and every reported scalar
python scripts/nesso1_port/model_parity.py

# on device, with the run-to-run noise floor measured first
TT_VISIBLE_DEVICES=1 python scripts/nesso1_port/device_parity.py
```

`model_parity.py` scores against `ref_scalars.json`, taken from upstream's own `predict_step`
under the seed the gate reseeds to. It also reports a non-blocking `cli_draw` leg against the
scalars the upstream CLI wrote; that leg is *expected* to differ, because the CLI featurized
under its own RNG state. Refresh the capture with `scripts/nesso1_port/capture_ref.py` in a venv
that has the upstream package installed.
