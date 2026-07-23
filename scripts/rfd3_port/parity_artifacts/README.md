# RFD3 featurizer value-parity artifacts (IAI protein motif-scaffold)

This directory holds a **real reference `f` capture** used to value-gate the ported
`tt_bio/rfd3_featurize.py` against the upstream RosettaCommons/foundry featurizer —
**without vast.ai**. The reference featurizer runs locally on CPU once the foundry
package is installed, so the value gate no longer needs a rented GPU.

## Input

- `IAI_protein.pdb` — chain A of PDB 8AOM's IAI ligand stripped to protein-only
  (1071 ATOM, 150 residues), used as the motif source.
- contig: `A1-10,20,A31-40` (motif A1-10 + 20-residue designed scaffold + motif A31-40).

## Reference capture

- `ref_f.pt` — the 43-tensor `f` dict produced by
  `rfd3.transforms.pipelines.build_atom14_base_pipeline(is_inference=True, ...)`
  run through `rfd3.inference.datasets.ContigJsonDataset.__getitem__` (no
  model/checkpoint needed; the featurizer pipeline is standalone).
- `ref_f.meta.json` — shapes + dtypes of every key.
- I=40 tokens, **L=419 atoms** (variable per token — see below).

## Reproduce

```bash
# one-time: install foundry on python 3.12 (uv makes this trivial, no vast.ai)
uv venv --python 3.12 /tmp/fndry_venv
uv pip install --python /tmp/fndry_venv/bin/python "torch==2.6.0" \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/fndry_venv/bin/python "rc-foundry[rfd3]"

# capture (CPU, ~5 s)
/tmp/fndry_venv/bin/python scripts/rfd3_port/capture_ref_f.py \
    --pdb scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb \
    --contig "A1-10,20,A31-40" \
    --out_dir /tmp/ref_iai_capture

# compare ported vs reference
python scripts/rfd3_port/parity_artifacts/parity_iai.py
```

## Value-gate result (p12 — 43/43 keys bit-exact, both token- and atom-level)

**Token-level: 19/19 keys bit-exact** (landed p11): `restype, ref_motif_token_type,
ref_plddt, is_non_loopy, is_motif_token_unindexed, is_motif_token_with_fully_fixed_coord,
is_protein, is_rna, is_dna, is_ligand, is_polar, terminus_type, asym_id, entity_id,
sym_id, residue_index, token_index, token_bonds, unindexing_pair_mask`.

**Atom-level: 24/24 keys bit-exact** (landed p12). The reference does NOT pad
every token to 14 atoms — it uses a **variable** atom count per token via the
`rfd3.constants.association_schemes["dense"]` scheme:
- MOTIF (fixed-seq) tokens emit ONLY their real heavy atoms, looked up per-slot
  via the "dense" scheme (with symmetry-reserved gaps, e.g. GLU's OE2 lands at
  slot 9, not 8) — e.g. SER=6 atoms (N,CA,C,O,CB,OG).
- DESIGNED (sequence-unknown) tokens emit the full 14-atom template
  (N,CA,C,O,CB + V0..V8 virtuals).
- Beyond backbone, atom NAMES are relabeled to generic `V0..V8` for BOTH motif
  and designed atoms (hides side-chain chemical identity from the name channel;
  real geometry still flows through `motif_pos`).

Protein-specific reference-feature semantics (`CreateDesignReferenceFeatures.forward`,
where `has_sequence` excludes protein under `generate_conformers_for_non_protein_only`):
- `ref_pos`, `ref_mask`, `ref_pos_is_ground_truth`, `ref_charge` are all-zero/False
  for EVERY protein atom (motif or designed) — real motif coordinates flow only
  through `motif_pos`.
- `ref_element`'s one-hot is the constant index-0 row for every atom (its scalar
  source is never filled for protein, not real chemical identity).
- `motif_pos` is centered: the whole design is translated so the center of mass
  of the real (motif) atoms sits at the origin.
- `ref_atomwise_rasa`, `active_donor`, `active_acceptor` are all-zero in the
  default inference config (not computed).

See `tt_bio/rfd3_featurize.py` module docstring for the full implementation and
state §2l for the p12 writeup.
