# of3t-bondcov — §6's last uncovered loss term fires, and it moves the gradient

Row: `of3t-bondcov`. Branch `wk/of3t-bondcov`, based on `origin/wk/of3t`. CPU only, no card,
no shipped default moved, nothing merged.

VERDICT: GO

TARGET: **4G5J** — afatinib (CCD `0WN`) covalently bound to EGFR Cys797, one `covale`
`_struct_conn` row whose partners sit in entities of different kind. Chosen from
`of3t-orchestrator/bondcov`'s five candidates as the cleanest single-bond case: one covalent
inhibitor, one link, so the mask entry and the gradient delta have one cause and not 48. Source:
the structure is entry `4g5j` of **OpenFold3's own `training_cache_with_templates.json`**
(180,975 structures), fetched preprocessed from the public unsigned `s3://openfold3-data` bucket
by `scripts/of3_port/build_of3_subset.py --ids 4g5j,4byh`. It is upstream's data, not ours, and
nothing is redistributed into the repo — `perf/of3t_bondcov/.gitignore` keeps the npz, sdf and
cache out and the corpus is identified by digest
`e20b564af303d16ecf155cbc283ae773f7a5df2c3e02d655d3c80d8bdf505975`. **4BYH** (ASN–NAG, two
links) was fetched alongside as the glycoprotein arm, because the brief pre-registered that
ASN–NAG and CYS–ligand may travel differently through the featuriser and a single shape would
not have told us. No bond is synthesised anywhere; a zero would have been the finding.

MASK: the featuriser **does** express the predicate. `finetune_1` / `weighted-pdb`, crop
{{MASK_CROP}}, seed {{MASK_SEED}}, 19 datapoints walked, 0 silent sample substitutions:
**{{N_FIRED}} of
{{N_WALKED}}** carry a non-zero `bond_mask`, computed with the loss's own expression
(`core/loss/diffusion.py:205-210`, `token_bonds * (is_polymer[..., None, :] * is_ligand[...,
None])`). 4G5J fires on **{{N_4G5J_FIRED}} of {{N_4G5J}}** of its datapoints, 4BYH on
**{{N_4BYH_FIRED}} of {{N_4BYH}}**. On the headline target {{HEAD_TARGET}} the single entry is
(i = 321, `is_ligand` True, `is_polymer` False; j = 92, `is_polymer` True, `is_ligand` False) —
one orientation, because the mask keeps `[ligand, polymer]` only — against `token_bonds` nnz
{{HEAD_TB}} and `loss_weights.bond` 4.0. Full table, `is_polymer`/`is_ligand` token counts beside
the bond counts:

| target | datapoint | index | polymer tokens | ligand tokens | token_bonds nnz | polymer–ligand pairs | bond_mask nnz |
|---|---|---|---|---|---|---|---|
{{MASK_TABLE}}

The 4BYH datapoints that read 0 are the useful half: they carry hundreds of `token_bonds` each
and **no** polymer–ligand pair, which is the shape all 8 corpus targets have and is exactly why
`of3t-auxheads` measured a zero. They are the negative control at mask level, on the same corpus,
the same instrument and the same crop as the positive.

**The crop is a draw and the mask count is a property of it.** A second sweep at crop
{{MASK2_CROP}}, seed {{MASK2_SEED}}, reads {{MASK2_FIRED}} of {{MASK2_WALKED}}, with 4G5J firing
on {{MASK2_4G5J_FIRED}} of {{MASK2_4G5J}} instead of {{N_4G5J_FIRED}} of {{N_4G5J}}: at
{{MASK_CROP}} tokens the single covalent bond survives every draw, at {{MASK2_CROP}} it does not.
That is why the gradient run records the `bond_mask` of its own batch rather than inheriting a
count from this table, and why the two are read at different crops (see GAP).
Artifacts `perf/of3t_bondcov/bond_mask_4g5j_4byh.json` and `..._crop256.json`, instrument
`perf/of3t_bondcov/bond_mask_probe.py`.

GRADIENT: **{{HEAD_DELTA}}**, non-zero. One forward, two losses differing only in
`loss_weights.bond`, two backwards, read the way `of3t-auxheads` read the zero one — the same
script, `perf/of3t_auxheads/bond_coverage.py`, unchanged in what it computes. On {{HEAD_TARGET}},
{{HEAD_STAGE}} / {{HEAD_DATASET}}, crop {{HEAD_CROP}}, {{HEAD_DTYPE}}, seed {{HEAD_SEED}}:
`bond_loss` = {{HEAD_BOND_LOSS}}, `‖g(bond=4) − g(bond=0)‖²` = **{{HEAD_DELTA}}** against a total
`‖g(bond=4)‖²` of {{HEAD_TOTAL}}, a share of the squared gradient norm of **{{HEAD_SHARE}}**, and
**{{HEAD_MOVED}} of {{HEAD_PARAMS}}** tensors moved. Total loss {{HEAD_LOSS4}} with the term and
{{HEAD_LOSS0}} without.

| target | bond_mask nnz | bond_loss | ‖Δg‖² | ‖g‖² | share of squared norm | tensors moved |
|---|---|---|---|---|---|---|
{{GRAD_TABLE}}

The first row is `of3t-auxheads`' reading on 5nw3, kept beside the new one because it is the
control that matters: same instrument, same stage, same crop, same seed, weight 4.0 on both, and
it reads **0.0 exactly** with {{BASE_MOVED}} of {{BASE_PARAMS}} tensors moved. The difference
between the two rows is the corpus and nothing else, which is what makes the non-zero attributable
to the bond rather than to the arm.

Where the delta lands, by top-level section:

| section | tensors | ‖Δg‖² in it | ‖g‖² in it | share of its own |
|---|---|---|---|---|
{{SEC_TABLE}}

COVERAGE: **8 of 8**, carried by **(`finetune_1`, `weighted-pdb`, `4g5j` chain 1)**. That triple
gives `bond` weight 4.0, a `bond_mask` with {{HEAD_MASK}} non-zero entry, a `bond_loss` of
{{HEAD_BOND_LOSS}} and a gradient contribution of {{HEAD_SHARE}} of the squared norm over
{{HEAD_MOVED}} tensors, which is §6's full test and not just its first half. The `bond` NOT
COVERED entry is retired: its reason was *"0 of 8 corpus targets carry a polymer–ligand bond"*,
which was true of the corpus and never true of the featuriser, and the distinction is the whole
result here.

`scripts/of3_port/stage_loss_coverage.py` now reports two numbers instead of one. It counted a
term covered when some (stage, dataset) pair gave it a non-zero weight, and by that rule `bond`
read 8 of 8 from the day the yamls were parsed while contributing exactly nothing — ten of the
sixteen pairs handing it 4.0 for no effect. The script separates **WEIGHTED** (8 of 8, off the
yamls) from **DEMONSTRATED** (read off measurement records naming the term, the triple, the
quantity measured and its value). A record with value 0 is refused with a note rather than
counted, because §6's own rule is that a term firing with a zero gradient contribution has been
skipped with extra steps; a malformed record raises. Checked against all three cases before the
real record existed: value 0 gives 0 of 8 with the note, a positive value gives 1 of 8 naming
the triple, a record missing fields exits listing them.

The other seven terms are not asserted here either. `census_evidence.py` converts
`of3t-gradients`' §6 runtime census into the same records, and the table then reads **7 of 8 with
`bond` as the only hole** — the campaign's own recorded figure, re-derived from
`perf/of3t_gradients/coverage_census.json` rather than quoted. Adding this row's record makes it
**{{DEMONSTRATED}} of 8**. The record carries the quantity's name because the two rows measure
different things: that census reads the norm of the gradient each term seeds into the model
outputs on the frozen 5nw3 batch, this row reads a share of the squared parameter-gradient norm.
Both are evidence that the term contributes; neither is the other, so the table prints the
quantity instead of adding them together. `make_evidence.py` derives this row's record from the
measurement json, so no value is typed into the table anywhere.

PROVES: that OpenFold3's featuriser turns a polymer–ligand `covale` into a `token_bonds` entry
whose partners carry `is_ligand` and `is_polymer`, that the resulting `bond_mask` is non-zero,
and that the `bond` term computed over it makes a measurable non-zero contribution to the
parameter gradient at `finetune_1` / `weighted-pdb` on a structure from upstream's own training
corpus. §6's last loss-term hole is closed on evidence.

DOESNOT: prove anything about long-run training. This is one step's gradient on one batch: it
shows the update rule's `bond` term is live and contributes, and it says nothing about stability
or drift over a full 100k-step run, which no single-step measurement can. It is CPU float32, not
a device comparison, so it is a statement about the term, not about our port's accuracy on it —
the device-side equivalence of this term at this triple is a separate measurement and belongs to
the rows that own model-scope gradients. It does not establish that the 8-structure corpus can
cover `bond`: it cannot, which is why the corpus was extended with structures upstream already
ships in its own cache.

GAP: the gradient is read at crop {{HEAD_CROP}} and the mask table at {{MASK_CROP}}, and the
reason is worth stating because it stopped being true during the pass. The first attempt at crop
{{MASK_CROP}} finished its forward in 269 s and was OOM-killed during the backward at 22 GB RSS
on a 30 GB box. Dropping to {{HEAD_CROP}} barely moved the peak, which is the tell: the memory was
not the activations. `of3-p2-155k` is 2.29 GB, so 570 M parameters in fp32, and the probe held four
copies of that scale — the checkpoint and its recast copy for the whole run, and two float64
gradient snapshots at 4.6 GB each. Freeing the checkpoint after `load_state_dict` and cloning at
the parameter's own dtype (the comparison upcasts per tensor, so every accumulated sum is float64
either way) takes about 14 GB off the peak, and crop {{MASK_CROP}} now fits on this host. The
reading below is the {{HEAD_CROP}} one because that is what ran; re-reading it at {{MASK_CROP}} is
now an ordinary run rather than a bigger host, and it is the obvious next thing to do with this
instrument.

4BYH's gradient is read at mask level only. `of3t-orchestrator/bondcov`'s three high-count targets
(6VXX 48 links, 7KJ2 38, 5T3X 19) are deliberately excluded: they are the same predicate at higher
count, and 4G5J answers the question with one cause instead of 48.

**A defect found on the way, fixed here.** `collate1` in `bond_coverage.py` recursed into
`ref_space_uid_to_perm`, which is a per-sample mapping `{ref-space uid -> [n_perm, n_atom]}` whose
batch axis is a plain python list, not a tensor dimension. Recursing unsqueezed every permutation
tensor and handed `single_batch["ref_space_uid_to_perm"]` the entry for uid 0 rather than the
mapping, so the first uid above 0 raised `IndexError`, upstream's `safe_multi_chain_permutation_
alignment` caught it, and the run continued on **naive alignment** with one warning line.
Measured: `ds[12]` on 4G5J emits a dict of 199 ref spaces, and the rank template that upstream's
own collator produced is a list of length 1 holding that dict. The fix is one branch; with it the
run logs zero alignment fallbacks. It needs a crop with more than one ref space to fire at all,
which is why it survived `of3t-auxheads`.

## Reproducing

    python scripts/of3_port/build_of3_subset.py --target-dir <D> --split train --ids 4g5j,4byh
    python perf/of3t_bondcov/bond_mask_probe.py --data-dir <D> \
        --cache-file <D>/training_cache_with_templates_subset_2.json \
        --stage finetune_1 --crop {{HEAD_CROP}} --out bond_mask.json
    python perf/of3t_auxheads/bond_coverage.py --package openfold3 --data-dir <D> \
        --cache-file <D>/training_cache_with_templates_subset_2.json \
        --checkpoint of3-p2-155k.pt --stage finetune_1 --crop {{HEAD_CROP}} --index 12 \
        --rank-template batch_step003.pt --out bond_gradient.json
    python perf/of3t_bondcov/census_evidence.py \
        --census perf/of3t_gradients/coverage_census.json --out-dir perf/of3t_bondcov/evidence
    python perf/of3t_bondcov/make_evidence.py --report bond_gradient.json \
        --out perf/of3t_bondcov/evidence/bond.json
    python scripts/of3_port/stage_loss_coverage.py --yamls <training_yamls> \
        --demonstrated perf/of3t_bondcov/evidence

openfold3 0.4.3, torch 2.13.0+cpu, host pc. `--index 12` is 4G5J chain 1; the index-to-target map
is the dataset's own `datapoint_cache` and both scripts print it.

This doc is rendered from the artifacts by `perf/of3t_bondcov/render_state.py`; no figure in it
is retyped.
