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

## Value-gate result (p11)

**Token-level: 19/19 keys bit-exact** vs the reference capture, after the p11
token-level fixes (ref_plddt direction, asym_id/entity_id/residue_index 0-based,
token_bonds all-False, restype designed->class 31):
`restype, ref_motif_token_type, ref_plddt, is_non_loopy, is_motif_token_unindexed,
is_motif_token_with_fully_fixed_coord, is_protein, is_rna, is_dna, is_ligand,
is_polar, terminus_type, asym_id, entity_id, sym_id, residue_index, token_index,
token_bonds, unindexing_pair_mask`.

**Atom-level: STRUCTURAL MISMATCH (owed rework).** The ported featurizer pads every
token to 14 atoms (L=560); the reference emits a **variable** atom count per token:
- motif (fixed-coord) tokens emit ONLY their real heavy atoms (no virtuals)
  — e.g. SER=6 (N,CA,C,O,CB,OG);
- designed (sequence-unknown) tokens emit the full 14-atom template
  (N,CA,C,O,CB + V1..V9 virtuals).

Plus the protein-specific semantics the ported layer gets wrong (all derived from
`CreateDesignReferenceFeatures.forward`, where `has_sequence` excludes protein):
- `ref_pos` is **all-zero** for protein (motif coords go via `motif_pos`, not `ref_pos`);
- `ref_mask`, `ref_pos_is_ground_truth` are **all-False** for protein;
- `ref_charge` is all-zero; `ref_element` is set by a later transform (verify values);
- `motif_pos` = `coord * is_motif_atom` **centered at the COM of the fixed motif**
  (origin subtracted); ported version is uncentered;
- `ref_atomwise_rasa`, `active_donor`, `active_acceptor` are **all-zero** in the
  default inference config (not computed); ported version must zero them;
- `is_virtual` is True only for the designed-token virtual slots, False for all
  motif atoms (ported version sets virtuals for every token).

The atom-level rework is the p11 major item; see state §2k.
