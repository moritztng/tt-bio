# OpenFold3 on Tenstorrent

[OpenFold3](https://github.com/aqlaboratory/openfold-3) is the OpenFold Consortium's
open AlphaFold3 reproduction. tt-bio runs it as `--model openfold3`: an independent
ttnn implementation of the model, with the consortium's own host-side data pipeline
vendored under `tt_bio/_vendor/openfold3/` for featurization. It folds proteins, RNA
and DNA, uses an MSA by default, and takes optional per-chain templates.

It is parity-gated against the official CPU reference on seven legs — see
[`implementation-parity.md`](implementation-parity.md) for the numbers, the noise
floors, and how to reproduce them.

## Weights

OpenFold3 is the one model tt-bio does not download for you. Fetch the consortium's
preview2 checkpoint and point `OF3_CKPT` at it, or put it where tt-bio looks by
default:

```bash
curl -o ~/.boltz/of3-p2-155k.pt \
  https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-155k.pt
```

The file is 2.29 GB and ungated (no login, no license click-through). Upstream states
the project is free for academic and commercial use under Apache-2.0 and publishes no
separate parameter license.

## Running it

```bash
tt-bio predict examples/prot.fasta --model openfold3
tt-bio predict examples/7xi5_tmpl.yaml --model openfold3    # per-chain MSA + templates
tt-bio predict proteins/ --model openfold3 --devices 0,1,2,3
```

It shares Protenix-v2's scheduler, worker, multi-card fan-out and MSA cache, so
`--devices`, `--out_dir`, `--diffusion_samples`, `--single_sequence`, `--msa_db_path`,
`--msa_endpoint` and `--controller` all behave the same way.

MSAs and templates attach per chain in the YAML. An MSA path is a ColabFold `.a3m` or
a benchmark MSA directory; a template is a precomputed alignment `.npz` (the format the
upstream benchmark cache ships). There is no template search — you supply the
alignment, and the structures it names are fetched from RCSB.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MSSATPDPAEILT...
      msa: ./msa_A
      templates: ./templates.npz
```

Ranking follows the OF3 rule: `0.8 ipTM + 0.2 pTM + 0.5 disorder - 100 clash`. Every
sample is written; the top-ranked one is `<name>.cif`. Note that this differs from the
`confidence_score` the other tt-bio fold models report, so the values are not
comparable across models.

## What is and isn't supported

Enumerated against the upstream input schema
(`of3_all_atom/config/inference_query_format.py`). Every unsupported case is a named
error, verified by running it — nothing silently degrades. Upstream's own inference suite
run against this port is reported in
[openfold3-upstream-suite.md](openfold3-upstream-suite.md), including the one item that
fails.

| capability | status |
|---|---|
| protein chains | ported, parity-gated |
| multi-chain complexes | ported, parity-gated (9BK6 heterodimer) |
| RNA, DNA | ported, folds end to end |
| templates | ported, per-chain alignment npz; no template search |
| MSA | ported: per-chain file or directory, shared hash-cache search, `--single_sequence`. An MSA that was requested but cannot be resolved raises rather than folding single-sequence |
| `--single_sequence` accuracy | **known gap.** It switches the MSA stack off, where upstream folds a one-row alignment with the stack on. On ubiquitin that costs 1.50 Å CA-RMSD (2.34 Å vs 0.84 Å). Pass a one-row a3m as the chain's `msa:` to get upstream's behaviour today. See [openfold3-upstream-suite.md](openfold3-upstream-suite.md) |
| recycling | ported; `--recycling_steps` default 3, i.e. 4 trunk cycles (the upstream default) |
| sample ranking | ported; confidence-selected best of N, all samples kept |
| multi-card `--devices` | ported, same fan-out as Protenix-v2 |
| ligands (SMILES/CCD) | **not supported** — polymer chains only, loud error |
| covalent bonds / `constraints:` | **not supported** — loud error; the fold would otherwise ignore them |
| paired MSA | **not ported** — complexes fold on per-chain unpaired MSAs |
| `--write_pae` | **not supported** — the confidence head computes PAE logits but the fold does not return the matrices |
| `--fast` | not gated for OpenFold3; it is a Boltz-2/ESMFold2 lever and no OF3 parity leg runs with it |

## Precision

Two numeric boundaries are load-bearing. Both are on by default and you should leave
them alone unless you are measuring.

**fp32 diffusion module** (`OF3_DIFFUSION_FP32_DEVICE`, default `1`). The sampler runs
on device in fp32, matching the reference rollout's own fp32 boundary — the same lever
Protenix-v2 uses for HSA. On 9BK6 the bf16 sampler misses the reference noise floor
(all-atom Kabsch 1.889 Å vs a 1.821 Å threshold) and fp32 clears it at 1.663 Å. The
cost is roughly 1.5x wall-clock on that target. Set it to `0` to opt out.

**Unfused trunk attention.** The fused
`ttnn.transformer.scaled_dot_product_attention` systematically flattens near-degenerate
attention distributions, which decorrelates the pairformer single track on targets that
sit on that knife-edge (7XI5: s-track PCC 0.44, 8.6 Å RMSD). The non-atom-level
`AttentionPairBias` therefore uses an explicit matmul + `ttnn.softmax` + matmul path,
and `fp32_softmax=True` at the OF3 trunk/template/MSA construction sites is required,
not an optimization.

The MSA track carries a residual bf16 gap (trunk z-PCC ~0.97) from the large activation
magnitudes inside its pair stack. It is inside the reference's own seed noise on every
gated leg; it is a bf16 conditioning limit, not a port defect.

## Tuning

| variable | default | what it does |
|---|---|---|
| `OF3_CKPT` | `~/.boltz/of3-p2-155k.pt` | checkpoint path |
| `OF3_TEMPLATE_STRUCTURES` | `~/.boltz/of3_template_structures` | RCSB template CIF cache |
| `OF3_DIFFUSION_FP32_DEVICE` | `1` | fp32 diffusion sampler (see above) |
| `OF3_MAX_MSA_SEQS` | `16384` | MSA row cap. Lower it only under memory pressure: the CPU reference folds the full featurized MSA, so a smaller cap is an input divergence (on 9BK6 a 1024-row subsample cost chain A 11.1 Å vs 7.6 Å) |

## Limits worth knowing

The shipped checkpoint is preview2 at 155k training steps, a fraction of the full AF3
schedule, and the full-PDB retrain is still in progress upstream. Accuracy tracks that
checkpoint, not AF3. tt-bio's job is to reproduce it faithfully, which the parity gate
measures; whether the checkpoint is good enough for your target is a separate question,
and the confidence outputs are the way to answer it.

Complexes fold on unpaired per-chain MSAs. Ligands, covalent bonds and PAE output are
not there yet.

## Reproducing the parity legs

The reference side is the official `openfold3` pip package on CPU in fp32:

```bash
python3.12 -m venv /tmp/of3-venv && /tmp/of3-venv/bin/pip install openfold3
```

`scripts/of3_ref_fixture.py` drives it for seeds 0..N on a fixed query JSON and
harvests the committed fixture layout; `scripts/full_parity_gate.py` runs the device
side and scores it. The fixtures live under
`docs/implementation-parity-data/ref-fixtures/openfold3/`. Full instructions are in
[`implementation-parity.md`](implementation-parity.md#reproduce).
