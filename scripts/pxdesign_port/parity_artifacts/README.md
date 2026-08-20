# PXDesign featurizer parity artifacts

`pdl1/` is a committed capture of the **upstream** PXDesign design featurizer on
`examples/PDL1_quick_start.yaml` as shipped: 196 tokens (116 cropped PD-L1 target + an
80-residue binder), 1250 atoms, 50 feature keys. `scripts/pxdesign_port/parity_gate.py` scores
`tt_bio.pxdesign.featurize` against it bit-exact, with no upstream install, no network and no
card.

| file | what |
|---|---|
| `ref_design_f.pt` | the 17 design-relevant tensors, full values |
| `ref_condition_inputs.pt` | the verbatim inputs to upstream's `get_condition_template_feature` -- distogram-atom coords, `res_name`, `mol_type`, `is_resolved` |
| `ref_design_f.meta.json` | shape, dtype and a 16-hex sha256 for all 50 keys upstream produced, so an unexpected key set is caught too |

## Regenerating

Needs qb1's `~/protenix_ref_venv` and the pinned source at `~/pxdesign_src`:

```
~/protenix_ref_venv/bin/python scripts/pxdesign_port/capture_ref_design_f.py \
    --pxdesign_src ~/pxdesign_src --yaml examples/PDL1_quick_start.yaml \
    --out_dir scripts/pxdesign_port/parity_artifacts/pdl1
```

PXDesign pins a protenix from the 0.5 era and the box has 2.0. `upstream_shim.py` re-points the
five moved modules, relaxes one `ListValue` guard and stubs the absent `pxdbench` with an object
that raises on any access, so a capture that actually depends on `pxdbench` fails loudly. None of
that touches an arithmetic the featurizer performs.

Two things the capture path needed that are not obvious. `parse_target` leaves
`condition.structure_file` pointing at the CIF it was handed, but `InferenceDataset` only accepts
the bioassembly `.pkl.gz` it wrote beside it, so the capture rewrites the path. And
`InferenceDataset.__getitem__` swallows exceptions into a placeholder dict with two keys, so the
capture drives `process_sample_dict` + `process_one` directly -- otherwise a failed capture looks
like a successful one that produced nothing.

## What the gate covers, and what it does not

Covered, bit-exact: `conditional_templ`, `conditional_templ_mask`, the 65-row lookup index, the
`xpb` exclusion (with a sensitivity arm that fails if leaking the placeholder would not change
the feature), and the 36-way `restype`. Both `conditional_templ` arms fail on a deliberately
introduced 65th bin edge and on a placeholder that is not excluded, so the gate is known to be
able to fail.

Not covered: the atom-array construction upstream of those functions -- CIF parse, tokenization,
crop, hotspot annotation. `hotspot` and the token-level identity keys are captured and
shape-checked but not yet recomputed by a tt-bio path. Do not read this gate as end-to-end
featurizer parity.
