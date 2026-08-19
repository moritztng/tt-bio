# RF3 featurizer parity fixtures

Ten capability classes, captured from the upstream RF3 featurizer
(RosettaCommons/foundry `models/rf3` at `d6c07df`, checkpoint
`rf3_foundry_01_24_latest_remapped.ckpt`) by `../capture_ref_f.py`, then packed by
`../pack_fixture.py`. 4.8 MB total, self-contained: the parity gate needs neither a
foundry install nor a device.

Every column below is measured from the committed capture, not asserted. A fixture
earns its place only if it actually turns its track on.

| Fixture | I | L | MSA | prot | DNA | RNA | lig | atomized | chiral | template | cyclic | MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `glke` | 4 | 30 | 1 | 4 | 0 | 0 | 0 | 0 | 9 | 0 | `[]` | 0.12 |
| `two_protein_chains` | 8 | 60 | 1 | 8 | 0 | 0 | 0 | 0 | 18 | 0 | `[]` | 0.24 |
| `protein_dna` | 8 | 112 | 1 | 4 | 4 | 0 | 0 | 0 | 45 | 0 | `[]` | 0.41 |
| `rna` | 8 | 115 | 1 | 4 | 0 | 4 | 0 | 0 | 57 | 0 | `[]` | 0.42 |
| `ligands` | 53 | 79 | 1 | 4 | 0 | 0 | 49 | 49 | 21 | 0 | `[]` | 1.10 |
| `covalent_glycan` | 25 | 34 | 1 | 11 | 0 | 0 | 14 | 22 | 15 | 0 | `[]` | 0.34 |
| `ncaa_small` | 24 | 64 | 1 | 24 | 0 | 0 | 0 | 18 | 24 | 0 | `[]` | 0.42 |
| `monomer_msa` | 12 | 100 | 3 | 12 | 0 | 0 | 0 | 0 | 39 | 0 | `[]` | 0.44 |
| `cyclic` | 8 | 67 | 1 | 8 | 0 | 0 | 0 | 0 | 24 | 0 | `[0]` | 0.27 |
| `template` | 41 | 138 | 1 | 14 | 0 | 0 | 27 | 27 | 63 | 196 | `[]` | 0.98 |

`I` tokens, `L` atoms, `MSA` depth, `template` = non-zero entries in
`has_distogram_condition`, `cyclic` = the `cyclic_asym_ids` value.

What each one is for:

- `glke` — 4-residue protein, the smallest thing that exercises the whole path.
- `two_protein_chains` — multimer, distinct `asym_id` / `entity_id`.
- `protein_dna`, `rna` — nucleic acids, via `chain_type` `POLYDEOXYRIBONUCLEOTIDE` /
  `POLYRIBONUCLEOTIDE`.
- `ligands` — CCD code, SDF file and SMILES in one input, all three ligand forms.
- `covalent_glycan` — a peptide/glycan covalent bond.
- `ncaa_small` — `GLK(MSE)ET(SEP)K`: two non-canonical residues, which the pipeline
  atomizes (18 of 24 tokens are atom-level).
- `monomer_msa` — an `.a3m` path, MSA depth 3 rather than query-only.
- `cyclic` — `--cyclic_chains A`; the only fixture where `cyclic_asym_ids` is non-empty.
- `template` — 14 backbone-resolved residues of 9DFN chain A plus the SAM ligand, with
  `template_selection A` and `ground_truth_conformer_selection C`.

Contents per directory:

- `ref_f.pt` — every tensor in the pipeline output, keys flattened with `/`
  (`feats/restype`, `ground_truth/coord_atom_lvl`, ...). 49 tensors.
- `ref_f.meta.json` — shape and dtype per key, the capture settings (`__seed__`,
  `__n_recycles__`, `__diffusion_batch_size__`), `__zero_stub_keys__`, and
  `__non_tensor__`.
- `input.json` (or `input.cif`) plus any referenced ligand or MSA file, with paths
  rewritten relative so the fixture is self-contained.

## Two things the capture has to get right

**Not every feature is a tensor.** `feats/cyclic_asym_ids` is a plain Python list and
the model reads it (`RelativePositionEncoding`, `pairformer_layers.py:485`). A
tensors-only capture drops it without a word, which is what the first version of this
harness did. It now lands in `ref_f.meta.json` under `__non_tensor__`, and the
`cyclic` fixture is the one that proves the value can be non-empty.

**Two keys are stored as shape-only zero stubs**: `feats/atom_level_embedding` and
`feats/mean_atom_level_embedding`. They are the MLFF (MACE) conformer-embedding track,
whose cache lives at an IPD-internal path that is not distributed. They are exactly
zero on every public run, asserted at pack time. The contract the gate checks is that
the ported featurizer emits all-zeros of the same shape.

## `ground_truth_conformer_selection` moves `ref_pos_ground_truth`, not the flag

Selecting a chain sets the atomworks policy to `ADD`, not `REPLACE`. Measured by
capturing the `template` input twice, once with `--ground_truth_conformer_selection C`
and once without: exactly the 27 SAM atom rows of `ref_pos_ground_truth` change, while
`ref_pos` and `ref_pos_is_ground_truth` are bit-identical. So `ref_pos_is_ground_truth`
stays all-False on anything the public inference API can express, and a port that
scored only that flag would conclude the track was unsupported.
