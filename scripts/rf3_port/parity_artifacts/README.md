# RF3 featurizer parity fixtures

Each directory is one capability class, captured from the upstream RF3 featurizer
(RosettaCommons/foundry `models/rf3` at `d6c07df`, checkpoint
`rf3_foundry_01_24_latest_remapped.ckpt`) by `../capture_ref_f.py`, then packed by
`../pack_fixture.py`.

| Dir | Input | Covers | I (tokens) | L (atoms) |
|---|---|---|---|---|
| `glke` | protein monomer | folding, baseline | 4 | 30 |
| `two_protein_chains` | two chains | multimer | 8 | 60 |
| `protein_dna` | protein + DNA | nucleic acids | 8 | 112 |
| `ligands` | protein + CCD + SDF + SMILES | all three ligand input forms | 53 | 79 |
| `covalent_glycan` | peptide + glycan bond | covalent modification | 25 | 34 |
| `monomer_msa` | protein + `.a3m` | MSA path (depth 3) | 12 | 100 |

Contents per directory:

- `ref_f.pt` — every tensor in the pipeline output, keys flattened with `/`
  (`feats/restype`, `ground_truth/coord_atom_lvl`, ...). 49 tensors.
- `ref_f.meta.json` — shape and dtype per key, plus the capture settings
  (`__seed__`, `__n_recycles__`, `__diffusion_batch_size__`) and
  `__zero_stub_keys__`.
- `input.json` plus any referenced ligand or MSA file, with paths rewritten
  relative so the fixture is self-contained.

Two keys are stored as shape-only stubs rather than tensors:
`feats/atom_level_embedding` and `feats/mean_atom_level_embedding`. They are the
MLFF (MACE) conformer-embedding track, whose cache lives at an IPD-internal path
that is not distributed. They are exactly zero on every public run, verified at
pack time. The contract the gate checks is that the ported featurizer emits
all-zeros of the same shape.

The reference is committed so the parity gate needs neither a foundry install nor
a device, the same arrangement as `scripts/rfd3_port/parity_artifacts/`.
