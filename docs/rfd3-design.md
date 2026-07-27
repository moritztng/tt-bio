# RFdiffusion3 (RFD3)

[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967) (from
the Institute for Protein Design) is an all-atom generative model for de novo
biomolecular design: instead of folding a given sequence, it generates new
structures — and the sequences/scaffolds that support them — from a design
specification. `tt-bio` runs it as an independent ttnn reimplementation (no
upstream RosettaCommons code is vendored).

## Design modes

Every mode shares the same `contig` mini-language (below); the mode is
determined by what the spec asks for.

| Mode | What it does | Real (`--from_pdb`) input support |
|---|---|---|
| Protein binder design | Design a protein that binds a target protein | Yes |
| Motif scaffolding | Design a scaffold around a fixed structural motif | Yes |
| Nucleic-acid binder design | Design a protein binder against a fixed DNA/RNA target | Yes |
| Small-molecule binder design | Design a protein binder against a ligand | Not yet (`NotImplementedError`) |
| Enzyme design | Design catalytic residue placement around one or more ligands | Not yet (`NotImplementedError`) |
| Symmetric oligomer design | Design a cyclic/dihedral symmetric assembly | Not yet (`NotImplementedError`) |

The last three modes run on-device and are value-parity-verified against a
captured reference, but the host featurizer (the step that turns a real PDB +
contig into device input) doesn't build their input yet — only `--from_pdb`
runs against a real input for the first three.

## Basic usage

```bash
tt-bio design specs.json --from_pdb --out_dir ./designs
```

`specs.json` maps design ids to a contig-based specification, one design per
key:

```json
{
  "binder-1": {"input": "target.pdb", "contig": "A1-100,70", "length": "70"},
  "scaffold-1": {"input": "motif.pdb", "contig": "A10-20,40,A30-40"}
}
```

The contig string reads left to right: `A1-100` takes residues 1-100 of chain
A from the input structure verbatim (fixed coordinates and sequence); a bare
number (`70`) is a designed region of that exact length; a range (`60-80`)
randomizes the designed length per design. `/0` marks a chain break. See
`tt-bio design --help` for the full grammar (indexed/unindexed motifs,
per-atom fixing, symmetry, and the rest of the InputSelection mini-language).

Each design writes one `<id>.cif` to `--out_dir`. `--num_timesteps` controls
the diffusion sampling steps (default 4, a fast smoke setting; the upstream
default is 200 for production-quality designs).

### Generating multiple designs per spec

`--num_designs N` produces N independent designs per spec (each with a different
noise seed, `--seed + i`), writing `<id>_<i>.cif` (when N>1; `<id>.cif` when
N=1). Designs from the same spec share device forwards in batches of up to 8 by
default. Set `--batch_size` to tune that limit; the runtime reduces it
automatically for larger atom counts.

Batching costs nothing in accuracy: the device forward is bit-identical across
batch size, so a batched design reproduces its standalone run exactly (min
trajectory PCC 1.000000, maxabs 0, at 200 timesteps and batch 8). Pick
`--batch_size` on throughput alone.

Throughput is where it's worth being careful, because batching pays off on
smaller designs and stops paying on large ones. Measured on one Blackhole p150a
from 20-step runs projected to 200 timesteps, with the decoder-trace lever
`RFD3_TRACE_DECODER=1` set. That lever is opt-in and `tt-bio design` does not set
it, so a plain run reads lower than the table; the ratios between rows are what
the batch-size guidance below rests on:

| design | atoms | batch 1 | batch 8 | batch 8 vs 1 |
|---|---:|---:|---:|---:|
| 40 residues | 419 | 0.0807 designs/sec | 0.1352 | 1.68x |
| 80 residues | 979 | 0.0539 | 0.0611 | 1.13x |
| 150 residues | 1959 | 0.0271 | 0.0257 | 0.95x |
| Mpro + nirmatrelvir | 2702 | 0.0163 | 0.0146 | 0.90x |
| 250 residues | 3359 | 0.0129 | 0.0120 | 0.93x |

Expect a few percent of run-to-run spread on these; a warm card reads faster than
a cold one.

Batch 8 wins clearly up to about 80 residues and is within about 10% of batch
1 above that, so the default suits every size and `--batch_size` is worth changing
only if you are chasing the last few percent on a single large spec. Raising it to
16 does not help at either end (0.1352 vs 0.1299 designs/sec at 419 atoms, and no
change at 3359). The
size-dependence is a memory-traffic one: batching shares the work that does not
depend on the design, and the per-design attention tensors it cannot share grow
with atom count. `--devices` is the parallelism that matters at large design sizes
either way.

`--devices 0,1,2,3` fans the (spec × `--num_designs`) jobs across the listed
physical TT cards, one pinned subprocess per card (data-parallel — the same
pattern `tt-bio embed`/`predict` use). Use in-forward batching on each card and
`--devices` together when generating a larger set.

```bash
# 32 designs per spec fanned across 4 cards:
tt-bio design specs.json --from_pdb --out_dir ./designs \
  --num_designs 32 --devices 0,1,2,3
```

## Checkpoint

The RFD3 checkpoint downloads automatically on first use, straight from the
[Institute for Protein Design's file server](https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt)
— the same URL RosettaCommons' own `foundry install rfd3` fetches — so no
`rc-foundry`/`foundry` install is needed. The ~2.5 GiB checkpoint downloads to a
scratch path under `--cache` (default `~/.boltz/rfd3`), gets split into the
~0.65 GiB of weights `tt-bio design` actually loads, and is then deleted —
~0.65 GiB kept on disk after the first run.

## License

RFD3 is BSD-3-Clause (Institute for Protein Design, University of Washington).
`tt-bio`'s implementation is an independent ttnn reimplementation; only the
checkpoint is fetched from IPD.
