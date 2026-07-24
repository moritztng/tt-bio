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

## Checkpoint

The RFD3 checkpoint downloads automatically on first use, straight from the
[Institute for Protein Design's file server](https://files.ipd.uw.edu/pub/rfd3/)
— the same URL RosettaCommons' own `foundry install rfd3` fetches — so no
`rc-foundry`/`foundry` install is needed. The ~2.5 GiB checkpoint downloads to a
scratch path under `--cache` (default `~/.boltz/rfd3`), gets split into the
~0.65 GiB of weights `tt-bio design` actually loads, and is then deleted —
~0.65 GiB kept on disk after the first run.

## License

RFD3 is BSD-3-Clause (Institute for Protein Design, University of Washington).
`tt-bio`'s implementation is an independent ttnn reimplementation; only the
checkpoint is fetched from IPD.
