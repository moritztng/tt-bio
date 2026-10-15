# RFdiffusion3 (RFD3)

[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967) (from
the Institute for Protein Design) is an all-atom generative model for de novo
biomolecular design: instead of folding a given sequence, it generates new
structures, and the sequences/scaffolds that support them, from a design
specification. `tt-bio` runs it as an independent ttnn reimplementation (no
upstream RosettaCommons code is vendored).

## Design modes

Every mode shares the same `contig` mini-language (below); the mode is
determined by what the spec asks for.

| Mode | What it does |
|---|---|
| Protein binder design | Design a protein that binds a target protein |
| Motif scaffolding | Design a scaffold around a fixed structural motif |
| Nucleic-acid binder design | Design a protein binder against a fixed DNA/RNA target |
| Small-molecule binder design | Design a protein binder against a ligand |
| Enzyme design | Design catalytic residue placement around one or more ligands |
| Symmetric oligomer design | Design a cyclic (`C<n>`) or dihedral (`D<n>`) symmetric assembly |

A symmetric design reads its subunit frames from the input structure, so the
input needs one protein entity whose chain count matches the symmetry order
(two identical chains for `C2`, three for `C3`). A heteromeric assembly, or
subunits of unequal size, is refused by name. `C<n>` and `D<n>` are also the
only groups upstream supports.

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
randomizes the designed length per design. `/0` marks a chain break. The full
grammar (indexed and unindexed motifs, per-atom fixing, the InputSelection
mini-language) is in the module docstring of `tt_bio/rfd3/input.py`; the fields a
spec can actually ask for are below.

### What a spec can ask for

Honoured: `input`, `contig`, `length`, `unindex` (string and dict form),
`select_fixed_atoms` (per-residue atom lists, and the `BKBN` shorthand for
N/CA/C/O), `ligand`, `symmetry`, `is_non_loopy`, `partial_t`, plus
`select_buried` and `select_exposed` on a spec that names a `ligand`.
`allow_ligand_on_existing_chain: true` is accepted because that is what this
port does anyway: every instance of a requested ligand code is picked up
wherever it sits in the input structure. `false` is refused.

Refused, because the feature they would set is not built yet and a spec that
asked for them used to run and quietly return an unconditioned design:
`select_hotspots`, `select_partially_buried`, `select_hbond_donor`,
`select_hbond_acceptor`, `select_unfixed_sequence`, `redesign_motif_sidechains`,
`ori_token`, `infer_ori_strategy`, `plddt_enhanced`, `dialect` other than 2,
`cif_parser_args`, `extra`, `select_buried`/`select_exposed` without a `ligand`,
the `TIP` atom shorthand, and any key not in the upstream schema. Hotspot
conditioning is the one most people want; there is no way to steer an RFD3 design
toward an epitope in this port today. `tt-bio design --model boltzgen` does have
it, as `binding_types`.

`symmetry` takes upstream's own block, e.g.
`{"id": "C3", "is_symmetric_motif": true, "is_unsym_motif": "Y1-11,Z16-25"}`.
Components named in `is_unsym_motif` stay as one copy instead of being
replicated per subunit. Naming a ligand code there is not supported (a ligand is
never replicated in the first place).

Each design writes one `<id>.cif` to `--out_dir`. `--num_timesteps` controls
the diffusion sampling steps (default 4, a fast smoke setting; the upstream
default is 200 for production-quality designs). A design sets up per-step device
state once before sampling, which on the largest designs costs about a second,
so the 4-step smoke setting spends most of its time on setup, and only a real
run reflects the per-step rate the table below quotes.

Designed positions come back named: RFD3's own sequence head predicts a residue
identity per designed token, and the CIF carries the prediction from the final
diffusion step, the one matching the written coordinates. Motif and target
residues keep their input identities. The built-in sequence is a starting point,
not a finished design sequence: upstream recommends redesigning it with a
sequence-design tool such as ProteinMPNN before ordering.

Throughput, batching and scaling across cards: [design-throughput.md](design-throughput.md).

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
physical TT cards, one pinned subprocess per card (data-parallel, the same
pattern `tt-bio embed`/`predict` use). Use in-forward batching on each card and
`--devices` together when generating a larger set.

```bash
# 32 designs per spec fanned across 4 cards:
tt-bio design specs.json --model rfd3 --from_pdb --out_dir ./designs \
  --num_designs 32 --devices 0,1,2,3
```

Two more options: `--spec a,b` runs only those spec ids from the inputs file, and `--fp32_residual` keeps the
residual stream of the DiT, encoder and token encoder in fp32 instead of bf16 (off by default).

The atom attention runs as one fused attention kernel that never materialises the full score matrix. It
is closer to a float64 reference than the older kernel chain, and a 685-residue design takes about 15 %
less time with it. At the same seed the design lands somewhere else, as it does after any rounding change
because the diffusion is chaotic: 5.5 to 9.3 A over four seeds, about the 7.3 A by which the older chain's
own Blackhole and Wormhole designs differ, and half of what changing the seed does. Design quality is
unchanged: CA-CA geometry, chain breaks and amino-acid diversity match, with fewer clashes. Above about
8,600 atoms the older chain runs instead, because the fused kernel's cached mask would take more than
600 MB per atom block. `RFD3_ATOM_SDPA=0` restores it.

Each diffusion step stays on the card from the noisy coordinates in to the update out. The modules used to
hand their outputs back to the host for a few additions and upload them again, about 15 round trips a step,
each of which left the card idle. A 685-residue design takes about 14 % less time on a Blackhole p150a
(89.0 to 76.1 s). The additions now round in bf16 on the card, so a design moves at the same seed: 3.6 to
13.5 A on Blackhole and 4.5 to 11.2 A on Wormhole at 685 residues, against 13 to 20 A for a seed change,
with CA-CA geometry, chain breaks, clashes and amino-acid diversity unchanged. `RFD3_RESIDENT_STEP=0`
restores the round trips.

The token transformer's attention also runs as one fused kernel, with each head's 48 channels padded to 64
in the weights so the kernel can take them. A 685-residue design takes about 15 % less time (72.7 to 61.4 s
on a Blackhole p300c). Graded over four seeds at 685 and 418 residues on Blackhole and on Wormhole: a design
moves 2.9 to 15.2 A at the same seed against 13 to 20 A for a seed change, and CA-CA geometry, chain breaks,
clashes and amino-acid diversity match the exact chain. `RFD3_DIT_SDPA=0` restores the older attention.

`--fast` additionally keeps the pair transitions' hidden activations in 8-bit block floating point and
collapses the pair-feature projection that does not change between steps into a table lookup. It runs the
same diffusion steps and recycles; only the kernels change. A 685-residue design takes about 13 % less time
than the default (60.6 to 52.5 s on a Blackhole p300c). Graded over four seeds at 685 and 418 residues on
Blackhole and Wormhole against the exact chain: a design moves 3.8 to 15.4 A at the same seed, inside what
changing the seed does (13 to 20 A), and CA-CA geometry, chain breaks, clashes and amino-acid diversity
match.

The token pair transition runs as one weight-stationary kernel that keeps its weights on chip and never
writes the hidden activations out. On a Blackhole p150a at 1350 MHz a 685-residue fast design drops from
61.3 to 56.3 s and a Wormhole one from 114.7 to 100.1 s; a default design drops from 62.9 to 51.9 s on
Blackhole and from about 125 to 91 s on Wormhole at 1000 MHz. Over 28 paired default seeds on both chips
clashes rise by about 5 per design with a standard error of about 4, chain breaks go from 14 to 15, and the
same-seed drift (2.9 to 14.1 A) stays inside the 13.8 to 21.7 A a seed change moves a design; the fast
grade (ten paired seeds) moves clashes by -1.4 and leaves chain breaks identical seed for seed.
`RFD3_PAIR_WS=0` restores the chunked transition.

The token transformer's attention masks (pair-bias projection, head-first layout, scale and mask) are built
by one kernel instead of a linear, a permute and three elementwise passes. It is closer to a float64
reference than the chain it replaces, and a 685-residue default design on Wormhole drops from 100.1 to
91.4 s. Over four seeds the design moves 3.8 to 10.1 A at the same seed against 14 to 18 A for a seed
change, with chain breaks, clashes and CA-CA geometry unchanged. It is on by default on Wormhole; Blackhole
keeps the chain until its own check reads (`RFD3_DPB=1` opts in). `RFD3_DPB=0` restores the chain.

## Checkpoint

The RFD3 checkpoint downloads automatically on first use, straight from the
[Institute for Protein Design's file server](https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt),
the same URL RosettaCommons' own `foundry install rfd3` fetches, so no
`rc-foundry`/`foundry` install is needed. The ~2.5 GiB checkpoint downloads to a
scratch path under `--cache` (default `~/.boltz/rfd3`), gets split into the
~0.65 GiB of weights `tt-bio design` actually loads, and is then deleted:
~0.65 GiB kept on disk after the first run.

## License

RFD3 is BSD-3-Clause (Institute for Protein Design, University of Washington).
`tt-bio`'s implementation is an independent ttnn reimplementation; only the
checkpoint is fetched from IPD.
