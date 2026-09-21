# of3t-covpaths: what I expect, registered before the first run

Five conditional paths read NOT COVERED in `perf/of3t_gradients/coverage_census.json`:
`templates`, `bond`, `nucleotide`, `disabled_parameters`, `multichain_permutation`. This file
states the outcome I expect for each and the evidence the expectation rests on, all of it read
off upstream source and off the training cache before any measurement exists.

A31 applies to each one: the thing I am asking for has to be satisfiable by the thing being
asked. For every path below I name the source line that decides whether it can fire at all,
and I say what would have to be true for the answer to be "cannot fire".

## Where the census's five reasons actually come from

Four of the five reasons are properties of ONE structure, 5nw3, which is the only target in the
frozen bundle. The census says so itself: "the four template slots are all mask-zero on 5nw3",
"no inter-token bond on 5nw3", "5nw3 is one protein chain plus Fe and Na". A property of one
target is not a property of the path. The fifth, `disabled_parameters`, is attributed to the
DATASET, on the grounds that `weighted-pdb` does not zero the confidence weights, and that
attribution is what I expect to be wrong: the gate is per-target, not per-dataset.

## 1. templates, FIRES

`initial_training.yml` gives `weighted-pdb` `template: {n_templates: 4, take_top_k: false}`, so
the path is configured on in the stage the census names. Whether it carries anything is a
property of the target's cache entry: `training_cache_with_templates.json` stores a
`template_ids` list per chain, and the featuriser can only fill a template slot from that list.
`1kvu` chain 1 carries six (`1udc_A 1nah_A 1nai_A 1uda_A 1udb_A 1xel_A`) and the corpus already
on this host holds 154 template structure arrays, so the slots have something to read.

Expect: `template_pseudo_beta_mask` non-zero on `1kvu`, and the template embedder's parameters
carrying a gradient that goes away at `n_templates=0`.

Cannot-fire would look like: the mask staying zero with six template ids present, which would
mean the template arrays never reach the feature tensor.

## 2. bond, FIRES, and the census is already out of date

`of3t-bondcov` measured it: `finetune_1` / `weighted-pdb` / 4G5J chain 1, crop 256, `bond_mask`
nnz 1, `bond_loss` 0.0012424831511452794, and 3,924 of 4,170 parameter tensors moving for a
0.146902 share of the squared gradient norm. The census still reads `"bond": {"covered": false}`.

Expect: I confirm that row by citation and by re-reading its artifact, contradict the census,
and do not re-run the gradient. If its artifact does not carry the numbers its state doc quotes,
that is the finding instead.

## 3. nucleotide, FIRES

The census's reason is that 5nw3 has no nucleic chain. The training cache has 19,572 DNA chains
and 16,000 RNA chains over 13,196 structures, all of them reachable by `WeightedPDBDataset`,
which is the class `initial_training` / `weighted-pdb` instantiates. Nothing about the dataset
excludes them.

Target: `4hj5`, one protein chain plus one DNA chain, 2.04 Ang, one templated chain. Resolution
is in `[0.1, 4.0]` on purpose, so the confidence weights stay on and this run varies only the
nucleotide axis.

Expect: `is_dna` non-zero, and upstream's `mse` entity weighting (`w_dna` 5.0 against 1.0 for
protein, `loss.py:1063-1207`) changing the gradient on this batch while the same edit changes
nothing on a protein-only batch. That pair is the break control.

Cannot-fire would look like: the featuriser mapping DNA tokens onto `is_protein`, so no
nucleotide-conditioned branch is ever selected.

## 4. disabled_parameters, FIRES, and the census's reason is the wrong gate

This is the one I expect to contradict on substance. The census says the path has no batch here
because "the frozen batch is weighted-pdb, the one dataset that does NOT zero them". The dataset
default is only half the gate. `set_loss_weights`
(`core/data/pipelines/featurization/loss_weights.py:44-49`) zeroes EVERY confidence loss when

    resolution is None  or  resolution < min_resolution  or  resolution > max_resolution

and `dataset_config_components.py:174-175` ships `min_resolution 0.1`, `max_resolution 4.0`.
So a `weighted-pdb` target whose resolution is outside that window arrives with all four
confidence weights at zero no matter what the yaml says, and `_get_sample_disabled_param_names`
(`runner.py:364-386`) then returns the confidence head's parameter names because their summed
weight is 0.

Satisfiability, checked before any number exists: 378 structures in the training cache are
single-chain, templated, protein-only and carry a NUMERIC resolution above 4.0. Target: `5oid`
at 4.6 Ang. Control: `1kvu` at 1.9 Ang, same stage, same dataset, same code path.

Expect: `batch["loss_weights"]` reading 0.0 for `experimentally_resolved`, `plddt`, `pae` and
`pde` on 5oid and non-zero on 1kvu; `_get_sample_disabled_param_names` returning a non-empty set
on 5oid and `None` on 1kvu; and after a backward, those parameters holding `grad is None` on
5oid rather than a zero gradient, which is the distinction PROTOCOL 3b makes load-bearing.

One thing I am deliberately NOT relying on: a `NaN` resolution. `NaN < 0.1` and `NaN > 4.0` are
both False, so a NaN resolution keeps the confidence losses ON. 5oid is a real number above the
bound, so the branch is taken for the reason I claim.

## 5. multichain_permutation, ALREADY COVERED; I expect to reproduce "completes", not the KeyError

`of3t-permalign` (D117) reports that upstream's `safe_multi_chain_permutation_alignment`
completes inside the real checkpointed forward on both batches this campaign holds, and that the
two ways the campaign drove it into the naive fallback are harness defects. Its
`census_correction.json` names the edit the census needs and says why that row did not make it.

Read at source, the mechanism it describes is there. `model.py:670` pops `ref_space_uid_to_perm`
off the CALLER's batch dict and writes it back onto a NEW dict built by `tensor_tree_map`, so a
second forward over the same dict finds the key gone; `pop(..., None)` then stores `None`, and
`reshape_per_sample_inputs` (`permutation_alignment.py:1693-1706`) only re-adds the key when it
is not None, so `single_batch["ref_space_uid_to_perm"]` at line 1412 raises `KeyError`. That is
the census's exact exception, produced by calling a forward twice on one dict rather than by
anything upstream cannot construct.

Expect: I re-run that row's probe myself rather than taking its word, and get "completes" on the
arms it says complete and the fallback on the three arms it configures to break. If instead I
get the KeyError on an arm that row records as completing, its result is the one that moves.

## What I am not claiming

`diffusion_rollout` and `model_forward` are outside this row. They need the OF3 training forward
wired in `tt_bio/train/`, which no batch selection reaches.
