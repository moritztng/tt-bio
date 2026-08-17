# OpenFold3's own inference tests, run against this port

`aqlaboratory/openfold-3` ships an end-to-end accuracy suite at `openfold3/tests/inference`. This page
reports what happened when we pointed it at the TT-Bio OpenFold3 port: what passed, what failed, what the
port does not implement, and where we had to adapt something.

Pinned at commit `72fc3a9534d37291b1ca7f02f11a8a0b12cd80c9` (2026-08-17). The `tests/inference` directory
itself is unchanged since `9ca6bb23` (2026-08-11), so the pin is not a moving target. 19 collected items.

## The suite cannot run our model, and unmodified it skips

Two facts to state before any number.

Every test builds an upstream `InferenceExperimentConfig` and calls `InferenceExperimentRunner.run()`,
which instantiates upstream's own torch model. There is no backend seam: no flag, env var or entry point
substitutes a different implementation. Running the suite in an environment where this port exists still
runs upstream's torch model.

And on a Tenstorrent host it skips. Every test carries `@skip_unless_accelerator_available()`, and
upstream's `ACCELERATORS` is `("cuda", "rocm", "mps")`. A TT card is none of those:

```
$ pytest openfold3/tests/inference/ -v -rs
SKIPPED [16] test_inference_full.py:542: Requires cuda or rocm or mps; found cpu
SKIPPED [1]  test_pocket_constraints.py:137: Requires cuda or rocm or mps; found cpu
SKIPPED [2]  test_templates.py:206: Requires cuda or rocm or mps; found cpu
19 skipped in 1.52s
```

We report that verbatim. Patching the guard would mean editing an upstream test, which we did not do.

What is portable is each test's **claim**: a named input, a named reference structure, upstream's own
metric, and a numeric bound its authors calibrated and documented inline. So every accuracy row below is
that claim re-evaluated with upstream's metric code (`openfold3.core.metrics.alignment.best_ca_rmsd`,
imported not reimplemented), upstream's reference cif, upstream's threshold and upstream's protocol
(200 steps, 3 recycles, 8 samples, seed 42, mean over samples). The only substitution is the model.
Those rows are marked ADAPTED.

## Results

Bounds and the parenthesised numbers are upstream's own, measured on MPS and written inline in the test
files. "CPU reference" is upstream's implementation run on CPU in fp32 on the same checkpoint
(`of3-p2-155k.pt`), same protocol, same input as our device leg; it lands within 0.12 Å of upstream's MPS
numbers on both ubiquitin modes, which is what makes it usable as a control.

Seven items pass, one fails, eleven are NOT APPLICABLE. Each of the 19 collected items is classified once,
taking the adapted variant where one exists.

| upstream item | verdict | our device | CPU reference, same input |
|---|---|---|---|
| `protein_only` × {no_msa, msa}, no templates | PASS (adapted) | structure + confidence record per sample, 68 residues over 2 chains | — |
| `protein_only` × 2 template modes | NOT APPLICABLE | no template search in the port | — |
| `protein_and_ligand` × 4 modes | NOT APPLICABLE | ligands not ported, refused loudly | — |
| `ubiquitin` no_msa, ceiling 1.8 Å (1.197 ± 0.273) | **FAIL** | **2.339 Å** with `--single_sequence`; **0.836 Å** on a one-row alignment | 1.079 Å PASS |
| `ubiquitin` msa, ceiling 1.9 Å (1.224 ± 0.456) | PASS | **0.813 Å** | 1.108 Å PASS |
| `ubiquitin` × 2 template modes | NOT APPLICABLE | no template search | — |
| `query_single_protein_single_ligand` × 4 modes | NOT APPLICABLE as written | ligand chain not ported | — |
| ⤷ adapted, protein only, no_msa, floor 8.0 Å (16.002 ± 0.388) | PASS | **14.803 Å** | 14.895 Å PASS |
| ⤷ adapted, protein only, msa, ceiling 0.6 Å (0.367 ± 0.049) | see note | **0.550 Å** | 0.950 Å, over the ceiling |
| `test_template_lowers_rmsd[1a8q]`, off > 8.0 / on < 2.0 / gap > 5.0 (16.58 ± 0.68 / 0.26 ± 0.02) | PASS (adapted) | off **16.600 Å**, on **0.244 Å**, gap **16.356 Å** | — |
| `test_template_lowers_rmsd[1y57]`, off > 18.0 / on < 16.0 / gap > 6.0 (23.77 ± 1.25 / 12.07 ± 2.83) | PASS (adapted) | off **24.482 Å**, on **13.173 Å**, gap **11.309 Å** | — |
| `test_pocket_constraint_localizes_ligand` | NOT APPLICABLE | ligand, pocket constraint and pocket-guided sampling all unported | — |

Per-leg evidence, including every per-sample value and gdt_ts, is committed under
`implementation-parity-data/openfold3-upstream-suite/`.

## What the NOT APPLICABLE rows mean

Each is a documented limit of the port, not an inconvenience. The full table is in
[openfold3-port.md](openfold3-port.md).

- **Ligands.** Nine of the nineteen items carry a SMILES or CCD chain. The port is polymer-only and
  refuses: `--model openfold3 is polymer-only for now (chain(s) ['C'] are ligands)`. No structure is
  written. Verified by running both ligand-bearing queries.
- **Pocket constraints.** `test_pocket_constraint_localizes_ligand` needs three unported things at once:
  the ligand, a `pocket_constraint` block, and pocket-guided proposal sampling. Its metric is a ligand
  centre-of-mass distance, and there is no ligand in our output to measure. On that input the ligand
  guard fires first, so the constraint path is never reached.
- **Template search.** The port loads a precomputed per-chain alignment npz. It does not align a
  user-supplied CIF to the query, which is what the template-enabled items ask for. Upstream's own
  comment notes that without the MSA server, template search has nothing to draw on, so the two `no_msa`
  template modes are the same computation as their non-template siblings upstream too.
- **PAE.** No suite item touches `--write_pae`, so this limit costs nothing here.

## Adaptations, stated

Four, and each one is itself a finding.

**The cif reader.** Upstream's `Structure.from_cif` goes through `parse_mmcif`, which needs a `chem_comp`
category this port's writer does not emit; it raises `KeyError: 'chem_comp'` on our output. So
`scripts/of3_upstream_score.py` parses our cifs with biotite and hands the resulting `AtomArray` to
upstream's own `Structure`. `best_ca_rmsd`, the chain-bijection search and the Kabsch superposition run
untouched, and reference structures still go through upstream's own parser. The reader is adapted; the
metric is not.

**Ligand-dropped 7L39.** Two items score T4 lysozyme L99A with toluene bound. Dropping the ligand is the
only way to run them here, and it changes the answer: on the ligand-dropped input upstream's own
implementation moves from 16.00 to 14.89 Å in no-MSA mode, and in MSA mode it goes from 0.367 Å to
0.950 Å, i.e. *over* its own 0.6 Å ceiling, with two of its eight samples at 1.95 and 2.23 Å. Ours reads
0.550 Å with all eight samples between 0.49 and 0.65 Å. So the ceiling does not apply to the modified
input, and on the modified input the port is not worse than the reference. That is not a claim to be
more accurate: two wide samples out of eight is a tail draw at n=8.

**Template alignments.** The template tests hand the implementation a CIF and ask it to align. We ran
upstream's own `TemplatePreprocessor` (`mode="predict"`) on the identical query to produce the alignment
npz, then folded with it. Upstream does the alignment it owns, this port does the fold. Nothing about the
query is changed, so upstream's calibrated bounds apply directly, and both cases land inside upstream's
own measured spread: 1a8q off 16.600 Å against their 16.58 ± 0.68 and on 0.244 Å against their
0.26 ± 0.02; 1y57 off 24.482 Å against 23.77 ± 1.25 and on 13.173 Å against 12.07 ± 2.83. The 1y57
alignment is the case with real indels at 61.7% identity, so it exercises the aligner rather than an
identity map. Both no-MSA arms pass all three assertions: the table reports the one-row-alignment arm,
and `--single_sequence` gives 1a8q off 17.856 / on 0.270 and 1y57 off 25.499 / on 10.522, so the ubiquitin
defect below does not reach these targets.

**Output-file assertions.** `test_inference_writes_outputs` names upstream's own layout
(`msas/<run>/{main,dummy}`, `timing.json`, a per-sample confidences trio). This port writes
`openfold3_results_<query>/structures/` plus a `results.json` confidence record, so the portable form of
the claim is checked: one structure per diffusion sample and a confidence record, in both modes.

## The one FAIL, and its cause

`ubiquitin` in no-MSA mode reads 2.339 Å against a 1.8 Å ceiling. It reproduces: three runs across two
hosts and two cards span 0.006 Å, and two runs on the same card are identical to every printed digit.

The cause is an input difference, not the model. Upstream does not switch the MSA stack off when a chain
has no MSA file: it writes a one-row alignment holding the query sequence and leaves `use_msas` on.
`--single_sequence` in this port sets `use_msas=False` instead. Folding the same ubiquitin through a
one-row alignment gives:

| | MSA stack | ptm | CA-RMSD, mean of 8 |
|---|---|---|---|
| `--single_sequence` | off | 0.628 | 2.339 Å |
| one-row alignment | on | 0.677 | 0.836 Å |
| real 19310-row MSA | on | 0.678 | 0.813 Å |

1.50 Å, from an input difference. Upstream's own inline note for this case records ptm 0.67 for
single-sequence ubiquitin, which is the one-row number. Every `--single_sequence` fold in this repo is on
the affected path; MSA-mode folds are not. The fix is not landed yet.

The suite also found that a query carrying `template_cif_paths` used to fold template-free in silence,
with all-zero dummy template features. That contradicted this port's own rule that unsupported input
raises, and it is now a named error.

## Reproduce

Two environments, because ttnn and upstream `openfold3` do not share one. Fold on device, score in the
upstream CPU venv.

```bash
# the upstream suite, unmodified
git clone --filter=blob:none --no-checkout https://github.com/aqlaboratory/openfold-3.git
git -C openfold-3 checkout 72fc3a9534d37291b1ca7f02f11a8a0b12cd80c9
python3.11 -m venv of3-venv
of3-venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
of3-venv/bin/pip install -e openfold-3 pytest
of3-venv/bin/python -m pytest openfold-3/openfold3/tests/inference/ -v -rs

# one adapted leg: fold on device, then score with upstream's metric
TT_VISIBLE_DEVICES=0 python3 -m tt_bio.main predict examples/of3_upstream/ubiquitin.yaml \
  --model openfold3 --out_dir out/ubq --single_sequence \
  --diffusion_samples 8 --sampling_steps 200 --recycling_steps 3 --seed 42

of3-venv/bin/python scripts/of3_upstream_score.py \
  --pred-dir out/ubq/openfold3_results_ubiquitin/structures \
  --ref 1ubq --ref-chains A --expected-samples 8 --ceiling 1.8 \
  --label ubiquitin-no_msa --out ubiquitin-no_msa.json
```

The CPU reference arm is the same `run_openfold predict` upstream ships, with `accelerator: cpu` and the
triton / DeepSpeed / cuEquivariance triangle kernels off since they require CUDA. An MSA file passed to
upstream must be named so its stem is one of upstream's `max_seq_counts` keys (`colabfold_main.a3m`); any
other name is skipped silently and the run then dies with a bare `IndexError`.

The case YAMLs, the committed one-row alignments and the template alignment npz files are all under
`examples/of3_upstream/`.

## What this evidence does and does not establish

**Does.** On the inputs the upstream suite names, for the subset of it this port implements, the port
lands within the accuracy bounds upstream's authors calibrated, measured with upstream's own metric code
against upstream's own reference structures. Where it does not, we say so and name the cause. And the
subset the port does not implement is refused loudly rather than silently degraded.

**Does not.**

- This is not bitwise identity and we do not claim it. Diffusion noise is drawn on device, so the same
  seed replays a different draw on every backend.
- It does not run the upstream suite unmodified against the port. It cannot: the suite drives upstream's
  own runner and has no backend seam. Every scored number comes from a harness that reuses upstream's
  metric, reference and thresholds and substitutes only the model.
- It says nothing about ligands, covalent bonds, pocket constraints, paired MSAs, template search or PAE
  output. Eleven of the nineteen items are NOT APPLICABLE for those reasons. That is a statement about the
  port's scope, not its accuracy.
- The accuracy measured is that of the p2-preview checkpoint at 155k steps, not of AF3. Whether that
  checkpoint is good enough for a given target is a separate question the confidence outputs answer.
- Single-target results at 34-447 aa are not a benchmark. This is not CASP.

Our own parity evidence, which compares against the reference implementation's run-to-run spread rather
than against a single reference draw, is in [implementation-parity.md](implementation-parity.md).
