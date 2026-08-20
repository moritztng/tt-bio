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

The torch reference is bit-exact against upstream. On device the model is deterministic run to run
and, with the fp32 trunk, 0.0089 off the torch reference at 148 tokens against upstream's own
0.058 run-to-run spread. `tt-bio affinity` is the CLI.

Speed is the open problem, not accuracy: 18 s per prediction at 148 tokens on one Blackhole p150a,
against 0.7 s of H200 device time at 256 aa. The model issues ~93,000 unfused ops per prediction
across 320 cheap pairformer layers, so it is dispatch-bound rather than FLOP-bound.

## Running it

```bash
tt-bio affinity complex.yaml                       # one input
tt-bio affinity ligands/ --out_dir screen          # a directory: model stays resident
tt-bio affinity complex.yaml --trunk bf16          # ~4x faster, measurably less accurate
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
