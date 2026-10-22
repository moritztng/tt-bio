# RFD4-Proteina (RFD4)

[RFD4-Proteina](https://huggingface.co/nvidia/RFD4-Proteina) (NVIDIA) designs all-atom proteins
through a latent flow: a trunk integrates a flow in an autoencoder's latent space, and the decoder
turns the final latents into coordinates and a sequence in one step. `tt-bio` runs the model as an
independent ttnn reimplementation and uses RFD4's own host code for input preparation and output
writing, vendored under `tt_bio/_vendor/rfd4_prep/`, so a spec means here exactly what it means
upstream.

## Install

```bash
pip install 'tt-bio[tenstorrent]'
```

One wheel, no container: RFD4's host package is vendored, so the only extra is the Tenstorrent
runtime. Upstream's `cuda12`/`cuda13` extras and its Dockerfile package NVIDIA hardware and have no
counterpart here. The weights download on first use.

## Basic usage

```bash
tt-bio design binder.yaml --model rfd4 --num_designs 4 --out_dir ./designs
```

`binder.yaml` is an RFD4 condition spec: one key per design, naming the input structure, a contig
and the conditions that pin parts of it.

```yaml
binder-1:
  input: target.cif                      # the target structure
  contig: "60-60, /0, A17-131"           # a 60-residue designed chain, then the target
  conditions:
    C_CRDGRP:                            # hold the target's backbone where it is
      1:
        - select: "sel('A/*/*/[N,CA,C,O]')"
    C_SEQ:                               # keep the target's sequence
      true:
        - select: "sel('A')"
    C_HOT:                               # atoms the designed chain must contact
      true:
        - select: "sel('A/*/56/[CG,OH]')"
```

A condition block is the condition's name, then the value to set, then the selections it applies
to.

The contig reads left to right: `A10-20` takes residues 10-20 of chain A from the input verbatim,
a bare number is a designed region of that length, a range (`50-120`) randomizes the designed
length per design, and `/0` is a chain break. `global_args:` sets defaults every design inherits;
a design's own keys win.

Each design is first built into an annotated CIF under `<out_dir>/inputs/`. That file is the
model's actual input, and it is kept rather than hidden in a temp directory because a design that
comes out wrong is almost always a selection that matched the wrong atoms, which is visible there
and nowhere else. Designs land beside it as `<design-id>_generated.cif`, in the input structure's
own frame, so they open alongside your target.

An annotated CIF you already have (one from an earlier run, or one written by upstream) is an input
on its own:

```bash
tt-bio design prepared.cif --model rfd4 --annotated_cif
```

Building each input runs checks on the structure before it is saved, and they are what catches a
selection that matched nothing. On a large spec they are also the slow part of preparation;
`--no_validate` skips them once a spec is known to be right.

## From Python

The same call, same defaults:

```python
from tt_bio import rfd4

paths = rfd4.design("binder.yaml", "designs/", num_designs=4, seed=0)
```

Sweeping seeds or specs pays the weight load once if you keep the engine:

```python
engine = rfd4.DesignEngine.from_pretrained()
for seed in range(8):
    rfd4.design("binder.yaml", f"designs/seed{seed}", seed=seed, engine=engine)
```

`rfd4.load_spec`, `rfd4.prepare_examples`, `rfd4.featurize` and `rfd4.write_design` are the same
steps separately, for when you want the features or the atom array in hand.

## The thinking budget

RFD4 buys accuracy with trunk passes rather than with integration steps. A recipe is a recycle
schedule: how often the trunk runs at each step and where those passes are spent. The released
recipe, `thinking_medium`, is 60 steps and **180** trunk passes, and it is what a run uses unless
you say otherwise.

| `--recipe` | trunk passes |
|---|---:|
| `thinking_low`, `low/flat`, `low/front_loaded`, `low/midpeak` | 120 |
| `medium/midpeak` | 150 |
| `thinking_medium` (= `medium/front_loaded`), `medium/flat`, `high/midpeak` | 180 |
| `thinking_high` (= `high/front_loaded`), `high/flat` | 300 |

All of them run 60 steps; they differ in how many passes each step gets and where the expensive
steps are. `front_loaded` spends them early, `midpeak` in the middle, `flat` evenly.

Upstream picks per design family: binders at `medium/front_loaded`, antibody and nanobody CDRs at
`high/front_loaded`, enzymes and motif scaffolds at `medium/midpeak`.

`--nsteps` changes the number of integration steps instead. Lowering it is a different, worse model
rather than a faster one, so tt-bio says which count ran and takes the pass schedule from the
recipe, refusing a schedule that does not match the step count.

## What a spec can condition on

| condition | what it pins |
|---|---|
| `C_CRDGRP` | the exact coordinates of a numbered group of atoms: a target, a ligand, a motif |
| `C_CRD` | coordinates, per atom, without grouping |
| `C_SEQ` | the residue type at the selected positions; everything else is designed |
| `C_HOT` | atoms the designed chain must contact |
| `C_RAS` | solvent accessibility per residue: `BURIED`, `INTERMEDIATE`, `EXPOSED` |
| `C_SEC` | secondary structure: `HELIX`, `SHEET`, `LOOP` |
| `C_HB`, `C_SALT`, `C_PISTACK`, `C_PICATION`, `C_HPHOB`, `C_MCOORD` | an interaction the position should make: hydrogen bond, salt bridge, pi-stack, pi-cation, hydrophobic contact, metal coordination |
| `C_IDX`, `C_IDXGRP` | whether a motif's residue numbering is given to the model or left for it to place |
| `C_CATH`, `C_CENTROID` | a CATH fold label, or where a chain's centre of mass should sit |

Conditions are one-body: they annotate atoms. A pairwise condition (`C_BP`, a distance) is refused
by the spec, and nucleic-acid secondary structure reaches the model through an annotated CIF
instead (`--annotated_cif`).

Selections use RFD4's `sel()` mini-language over chain, residue and atom (`sel('A/*/[54,56]')`).
A selection that matches nothing is refused when the input is built, not after the design runs.
Ligands, DNA, RNA and glycans reach the model with their elements, formal charges and bond orders;
a sugar tree arrives with its glycosidic bonds intact.

## Design families and what they cost

Measured on one Blackhole p300 chip at 1325-1350 MHz AICLK (sampled during the design), at the full
recipe: every step, every recycle. Each run made two designs with the kernels already compiled;
s/design is the run's wall time over two, so model load is included.

| family | tokens | recipe | s/design |
|---|---:|---|---:|
| motif scaffolding | 82 | `medium/midpeak` | 9.2 |
| unconditional monomer | 100 | `thinking_medium` | 11.7 |
| structure prediction | 138 | `thinking_medium` | 14.0 |
| enzyme (catalytic site + ligands) | 223 | `medium/midpeak` | 16.3 |
| binder against a protein target | 225 | `medium/front_loaded` | 21.7 |
| nanobody CDRs | 224 | `high/front_loaded` | 21.8 |
| antibody CDRs (Fab) | 329 | `high/front_loaded` | 63.7 |

Cost grows with the token count and with the recipe, not with `--num_designs`: designs are
independent trajectories.

Partial diffusion and inverse folding, which start from an input structure instead of noise, are
not on the CLI yet.

A design is reproducible from its seed, featurization included. Note that featurization is part of
that: a contig with a variable-length segment samples its length per seed, so the same spec at two
seeds is two different designs with two different token counts.

## Precision

The trunk's pair transition and the triangle multiplication's input projection run with block-fp8
weights and a bf16 matmul kernel by default, and the pair transition and the triangle multiplication's
output each run as one program with their residual add into the fp32 pair representation. Together that
takes the trunk of a 512-token design from 64.3 to 44.1 s on a Blackhole chip. Forming each block's pair
biases and writing the pair update as one program each is a further 1.10x at 512 tokens, graded on 31
designs with radius of gyration, bond outliers and clashes unchanged. Running the triangle multiplication's
input kernel as one launch, with block-fp8 weights at HiFi2, takes the 512-token trunk from 40.6 to 37.2 s
(1.09x, 1.08-1.13x from 160 to 640 tokens). It is not bit-exact against
fp32 numerics: on 30 seed-matched 512-token binder designs the designed chain moved 7.2 A median (23.9 A at
most) while a different seed moves it 37.2 A, and radius of gyration, bond outliers and clashes were
unchanged. `--fast` adds nothing further for RFD4. `TT_BIO_LEVERS=trimul_t1` restores fp32 numerics
([`tuning-flags.md`](tuning-flags.md#tt_bio_levers)).

## Many designs at once

A spec can hold any number of designs and `--num_designs` multiplies them. Locally they run one
after another on one card. Across a fleet, each becomes its own job:

```bash
tt-bio design specs.yaml --model rfd4 --num_designs 32 --controller http://127.0.0.1:8765
```

The inputs are built once on the submitting host and each prepared CIF rides with its job, so a
worker needs no view of your filesystem and a spec that selects nothing fails here rather than on
every worker. Design *k* gets seed `seed + k`, so a single design out of a 32-design run
reproduces on its own. `<out_dir>/metrics.json` lists every design that landed with its token
count and runtime.

From Python: `rfd4.design_via_controller(spec, out_dir, controller_url=...)`.

## Backends

Upstream has two: `native`, the model, and `bioir`, which runs the whole workflow through NVIDIA's
BioNeMo-IR with the trunk captured as a CUDA graph. tt-bio runs `native` on Tenstorrent.
`--backend bioir` is declined with that reason rather than quietly ignored; it produces the same
designs on hardware tt-bio does not target.

## Fine-tuned weights

A LoRA fine-tune loads on top of the released weights and is merged once at load, so it costs
nothing per design:

```bash
tt-bio design binder.yaml --model rfd4 --lora my-lora/
```

```python
rfd4.design("binder.yaml", "designs/", adapter="my-lora/")
```

`adapter` is a PEFT directory (`adapter_config.json` plus `adapter_model.safetensors`); its rank
and alpha come from that config. An adapter naming a layer the checkpoint does not have is
refused rather than silently skipped, and a path that is not a directory is refused before the
weights load. An adapter is not shipped to fleet workers, so `--lora` and `--controller` together
are refused rather than quietly dropping the fine-tune. Training an adapter is not something
tt-bio does; it runs one you already have.

## Checkpoint

The `nvidia/RFD4-Proteina` bundle downloads from HuggingFace on first use and carries both the flow
model and the autoencoder.

## License

RFD4-Proteina's code and weights are BSD-3-Clause. tt-bio vendors upstream's host side
(`tt_bio/_vendor/rfd4_prep/`, BSD-3-Clause, Institute for Protein Design) and reimplements the
model itself for ttnn; no upstream compute code is vendored. See [`NOTICE`](../NOTICE).
