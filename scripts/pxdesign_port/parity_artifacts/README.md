# PXDesign featurizer parity artifacts

Each directory is a committed capture of the **upstream** PXDesign design featurizer on one
target. `scripts/pxdesign_port/parity_gate.py --art <dir>` scores `tt_bio.pxdesign.featurize`
against one bit-exact, with no upstream install, no network and no card.

| capture | target | tokens | atoms | conditioned | at origin |
|---|---|--:|--:|--:|--:|
| `pdl1_protenix05_noH` | PD-L1, hydrogens stripped — **the default** | 196 | 1250 | 116 | 0 |
| `rbd_6m0j` | SARS-CoV-2 RBD, 6M0J chain B — second anchor | 274 | 1857 | 194 | 0 |
| `pdl1` | PD-L1 from the shipped CIF, protenix 2.0 — **kept as the defect** | 196 | 1250 | 116 | **61** |
| `pdl1_protenix05` | PD-L1 from the shipped CIF, protenix 0.5.5 | 135 | 784 | 55 | 0 |

The last two are wrong on purpose and kept so the defect stays diffable. The shipped
`examples/5o45.cif` carries 1248 explicit hydrogens, and PXDesign's CIF path does not filter
them, so 61 of the target's 116 cropped residues are parsed as fully unresolved and conditioned
on at the origin — see `strip_cif_hydrogens.py` for the mechanism and the fix. protenix 2.0 keeps
those 61 as tokens at the origin; 0.5.5 deletes them and leaves a 55-residue target. Both read
the same 55 real coordinates, and neither is the target the YAML asks for.

| file | what |
|---|---|
| `ref_design_f.pt` | the 17 design-relevant tensors, full values |
| `ref_design_inputs.pt` | the model-ready input dict for a generation-only run |
| `ref_condition_inputs.pt` | the verbatim inputs to upstream's `get_condition_template_feature` — distogram-atom coords, `res_name`, `mol_type`, `is_resolved` |
| `ref_design_f.meta.json` | shape, dtype and a 16-hex sha256 for all 50 keys upstream produced, so an unexpected key set is caught too, plus which protenix produced the capture |

## Regenerating

Needs qb1's `~/protenix_ref_venv`, the pinned source at `~/pxdesign_src`, the unpacked protenix
0.5.5 at `~/protenix05/pkg` and biotite 1.0.1 on `PYTHONPATH` (0.5.5 reads
`pdbx.convert.PDBX_COVALENT_TYPES`, which biotite 1.4.0 removed). `LAYERNORM_TYPE` matters or
protenix JIT-compiles a CUDA extension at import.

```
python3 scripts/pxdesign_port/strip_cif_hydrogens.py \
    ~/pxdesign_src/examples/5o45.cif ~/pxdesign_src/examples/5o45_noH.cif
# then point a copy of the YAML at the stripped CIF

PYTHONPATH=$HOME/pxd_pinned_deps/pkg LAYERNORM_TYPE=torch_layernorm \
~/protenix_ref_venv/bin/python scripts/pxdesign_port/capture_ref_design_f.py \
    --pxdesign_src ~/pxdesign_src --yaml examples/PDL1_noH.yaml \
    --protenix05 ~/protenix05/pkg \
    --out_dir scripts/pxdesign_port/parity_artifacts/pdl1_protenix05_noH
```

The RBD anchor is `anchors/RBD_anchor.yaml` in this tree; drop it and
`curl -O https://files.rcsb.org/download/6M0J.cif` into `~/pxdesign_src/examples/`. It needs no
hydrogen stripping (6M0J has none) and no MSA — PXDesign's YAML makes the MSA optional, and
PXDesign-d has no trunk to feed one to.

Capturing on 2.0 instead goes through `upstream_shim.py`, which re-points the five moved modules,
relaxes one `ListValue` guard and stubs the absent `pxdbench` with an object that raises on any
access. An earlier version of this file claimed none of that touches an arithmetic the featurizer
performs; that was the load-bearing assumption under three passes of a docking failure and it is
withdrawn. The version difference is real and it changes the token count.

Two things the capture path needed that are not obvious. `parse_target` leaves
`condition.structure_file` pointing at the CIF it was handed, but `InferenceDataset` only accepts
the bioassembly `.pkl.gz` it wrote beside it, so the capture rewrites the path. And
`InferenceDataset.__getitem__` swallows exceptions into a placeholder dict with two keys, so the
capture drives `process_sample_dict` + `process_one` directly — otherwise a failed capture looks
like a successful one that produced nothing.

## What the gate covers, and what it does not

Covered, bit-exact: `conditional_templ`, `conditional_templ_mask`, the 65-row lookup index, the
`xpb` exclusion (with a sensitivity arm that fails if leaking the placeholder would not change
the feature), and the 36-way `restype`. Those arms fail on a deliberately introduced 65th bin
edge and on a placeholder that is not excluded, so they are known to be able to fail.

The fourth arm exists because the first three cannot fail on a bad capture: they recompute the
design features from the captured inputs, so the port and upstream go wrong together and the
comparison still passes. All three pass on `pdl1`, whose target is half fictional. **A gate that
scores a port against a capture cannot score the capture.** So the origin arm scores the capture
instead — no conditioned token may sit at the origin — and it fails on `pdl1` with 61 and passes
on the other three. `tests/test_pxdesign_featurizer.py` pins that asymmetry in both directions.

Still not covered: the atom-array construction upstream of these functions — CIF parse,
tokenization, crop, hotspot annotation. `hotspot` and the token-level identity keys are captured
and shape-checked but not recomputed by a tt-bio path. Do not read this gate as end-to-end
featurizer parity.
