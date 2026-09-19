# BUNDLE-MIN — the validated step-1 gradient reference

What instrument A is measured against: one fixed batch, the weights it was taken at, the
per-parameter float64 gradient, the random draws that step made, and a finite-difference validation
of the gradient itself.

The gradient agrees with float64 central finite differences to a **median 4.17e-04 and a worst
2.18e-03** over 12 entries, ten of them inside the pairformer trunk. The bar it will be compared at
is 5.0e-02, so the reference sits 23x inside it.

**Read-only once published.** If something needs to change, publish a new bundle with a new
manifest; a row that verified a hash must be able to tell.

## Files

| file | what it is |
|---|---|
| `MANIFEST.json` | THE PUBLICATION. Hashes, versions, the upstream commit, the draws, the validation. In git, read this one. |
| `hsweep.json` | the step-size sweep showing why a recycled trunk gradient cannot be finite-difference validated. In git. |
| `grad_presence.json` | per parameter, whether it had a gradient at all. In git. |
| `run_record.json` | one `bundle_min.py` invocation. In git. If it disagrees with MANIFEST.json, MANIFEST.json wins. |
| `grads_f64_recycles0.pt` | **the validated gradient.** Use this one. |
| `draws_recycles0.pt` | its draws: recycle count, diffusion noise, every `torch.randn` and `random.random` value, full RNG state |
| `grads_f64_recycles3.pt` | the gradient at the recycle count their config actually draws. Not finite-difference validatable. |
| `draws_recycles3.pt` | its draws |
| `batch_step003.pt` | the frozen batch |

Artifacts live on `tt-quietbox2` at `/home/ttuser/of3t/bundle_min/`, each with a sha256 in the
manifest. Git carries the manifest, the hashes and the generator, never a `.pt`.

`w_0` is not shipped: it is `of3-p2-155k.pt`, downloadable from
`https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-155k.pt`. The three keys it
does not supply are named in the manifest; a consumer must end up with the same two `layer_norm_z`
weights unset or the two stacks do not start from the same `w_0`.

## The batch

`5nw3`, 56 tokens: one protein chain plus an Fe and an Na ligand, cropped by their own
`WeightedPDBDataset` at `token_budget` 384. It is the smallest entry in the corpus on which all
eleven loss terms are non-zero.

## Four things a consumer must not get wrong

**Load the checkpoint, not a random init.** At a random init OF3 zero-initialises 2,271 of 4,890
weight tensors, a zero gate passes zero gradient upstream, and the step-1 gradient reaches 6 of
4,147 tensors. A comparison there is zero against zero and passes with your trunk deleted.

**Pin `num_recycles` to 0 on your side too.** Their trunk runs `num_recycles + 1` passes over the
same weights and tapes only the last, so the recycle count changes what the gradient *means*. A
reference at 0 compared against yours at 3 disagrees with no bug anywhere.

**Match the draws before comparing anything.** The recycle count and the diffusion noise are inputs
to the update rule. The recycle draw in particular comes from `OpenFold3.synced_generator`, a
private `np.random.default_rng` held as a model attribute — `torch.manual_seed` does not reach it.

**`None` is not zero, and the gradient is before clipping.** Compare the presence pattern as a set
first. At world size 1 with `accumulate_grad_batches` 1 the optimizer-facing gradient is exactly
`grad * clip_coef`, and the manifest records the coefficient.

## Fetch the bytes, do not regenerate

Rebuilding the batches twice on one machine gives 20/20 byte-identical files. Rebuilding on a
different machine from the same seed, corpus and config gives 0/20 — while producing the identical
entry order and token counts, so nothing announces it.

Full method and results: `~/.coworker/state/of3t-reference.md`.
