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
tt-bio design specs.json --model rfd3 --from_pdb --out_dir ./designs
```

`specs.json` maps design ids to a contig-based specification, one design per
key:

```json
{
  "binder-1": {"input": "target.pdb", "contig": "A1-100,70", "length": "70"},
  "scaffold-1": {"input": "motif.pdb", "contig": "A10-20,40,A30-40"}
}
```

`examples/rfd3_binder.json` is a working spec you can run as-is: a 70-residue
binder against a 50-residue motif taken from `examples/ground_truth_structures/prot.cif`.
`input` is resolved relative to the directory you run the command from, not to the
spec file.

The contig string reads left to right: `A1-100` takes residues 1-100 of chain
A from the input structure verbatim (fixed coordinates and sequence); a bare
number (`70`) is a designed region of that exact length; a range (`60-80`)
randomizes the designed length per design. `/0` marks a chain break. See
`tt-bio design --help` for the full grammar (indexed/unindexed motifs,
per-atom fixing, symmetry, and the rest of the InputSelection mini-language).

Each design writes one `<id>.cif` to `--out_dir`. `--num_timesteps` controls
the diffusion sampling steps (default 4, a fast smoke setting; the upstream
default is 200 for production-quality designs). A design sets up per-step device
state once before sampling, which on the largest designs costs about a second —
so the 4-step smoke setting spends most of its time on setup, and only a real
run reflects the per-step rate the table below quotes.

Designed positions come back named: RFD3's own sequence head predicts a residue
identity per designed token, and the CIF carries the prediction from the final
diffusion step, the one matching the written coordinates. Motif and target
residues keep their input identities. The built-in sequence is a starting point,
not a finished design sequence: upstream recommends redesigning it with a
sequence-design tool such as ProteinMPNN before ordering.

### Generating multiple designs per spec

`--num_designs N` produces N independent designs per spec (each with a different
noise seed, `--seed + i`), writing `<id>_<i>.cif` (when N>1; `<id>.cif` when
N=1). Designs from the same spec share device forwards in batches of up to 8 by
default. Set `--batch_size` to tune that limit; the runtime reduces it
automatically for larger atom counts.

Batching costs nothing in accuracy: the device forward is bit-identical across
batch size, so a batched design reproduces its standalone run exactly (min
trajectory PCC 1.000000, maxabs 0, at 200 timesteps and batch 8), so
`--batch_size` is a throughput and memory knob only.

Throughput: batching pays off most on small designs and still pays on large ones.
Each cell is one real 200-timestep design on one Blackhole p150a in the default
configuration, timed end to end, and is the faster of two interleaved rounds:

| design | atoms | batch 1 | batch 8 | batch 8 vs 1 |
|---|---:|---:|---:|---:|
| 40 residues | 419 | 0.0632 designs/sec | 0.1968 | 3.11x |
| 80 residues | 979 | 0.0581 | 0.1171 | 2.02x |
| 150 residues | 1959 | 0.0452 | 0.0569 | 1.26x |
| Mpro + nirmatrelvir | 2702 | 0.0317 | 0.0334 | 1.05x |
| 250 residues | 3359 | 0.0275 | 0.0316 | 1.15x |

Expect several percent of run-to-run spread on these. A warmer card reads slower,
so comparing two settings back to back needs an even number of runs with the order
alternating, or the second one is penalised.

An earlier version of this table composed its batch-1 cells from a one-time build
cost plus a per-step rate. That composition runs about 1.45x optimistic at batch 1
on a small design, so those cells read higher than these and their batch-8-vs-1
ratios read lower. The CLI agrees with the timed numbers: `tt-bio design
--num_timesteps 200` at 419 atoms and `--batch_size 1` costs 15.7 s per design,
which is the 0.0632 in the first row.

`--batch_size` is an upper bound, not the batch you get. Two limits reduce it. A
memory limit scales the batch down by atom count so a batch cannot exhaust device
memory: 8 is reachable up to 3359 atoms, 4 up to 4750, 2 up to 6718. A speed limit
then pins the batch to 1 above 2952 atoms, because past about 3000 atoms batching
stops paying: at 6051 atoms the largest batch that fits is 2 and it measures no
faster than one design at a time, and at 3844 atoms a batch of 6 is worth
about 5%, too little to be worth a size-dependent default. So every row
above is what the CLI actually does, and a target larger than 2952 atoms runs one
design at a time whatever `--batch_size` says.
Batch 8 at 3359 atoms, the largest batch the CLI will run, peaks at 11.1 GiB of
the card's 31.9.
Lower `--batch_size` only to cut memory further; raising it above 8 does not help.
`--devices` is still the parallelism that matters at large design sizes.

`--devices 0,1,2,3` fans the (spec × `--num_designs`) jobs across the listed
physical TT cards, one pinned subprocess per card (data-parallel — the same
pattern `tt-bio embed`/`predict` use). Use in-forward batching on each card and
`--devices` together when generating a larger set.

```bash
# 32 designs per spec fanned across 4 cards:
tt-bio design specs.json --model rfd3 --from_pdb --out_dir ./designs \
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
