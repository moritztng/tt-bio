# BUNDLE-MIN — the frozen step-1 gradient reference

What instrument A is measured against: one fixed batch, the weights it was taken at, the
per-parameter gradient in float64, the random draws that step made, and a finite-difference
validation of the gradient itself.

**Read-only once published.** If something here needs to change, publish a new bundle with a new
manifest rather than editing this one; a downstream row that verified a hash must be able to tell.

## What is in it

| file | what it is |
|---|---|
| `manifest.json` | the record: hashes, versions, the upstream commit, the draws, the finite-difference result. This file is in git. |
| `grad_presence.json` | per parameter, whether it had a gradient at all. In git. |
| `batch_step003.pt` | the frozen batch, hashed in the manifest |
| `w0.pt` | the weights the gradient was taken at |
| `grads_f64.pt` | per-parameter gradient, float64, `None` kept as `None` |
| `draws.pt` | recycle count, diffusion noise levels, every `torch.randn` draw, every `random.random` draw, and the full RNG state including the model's private generator |

The four large files live on `tt-quietbox2` at `/home/moritz/of3t_bundle/bundle_min/`; the manifest
in git carries their sizes and sha256 so a consumer can verify what it fetched. They are not in git
because they are 7.5 GB.

## The batch

`5nw3`, 56 tokens: one protein chain plus an Fe and an Na ligand, cropped by their own
`WeightedPDBDataset` at `token_budget` 384. It is the smallest entry in the corpus on which all
eleven loss terms are non-zero, which is why it was chosen — a gradient checked on a batch that
never fired the confidence heads proves nothing about them.

## Things a consumer must not get wrong

**Match the draws before comparing anything.** The trunk recycle count is drawn per step from
U{0..3} and the diffusion head noises 48 structures. A comparison whose two stacks drew differently
is comparing different amounts of work, and it looks exactly like a weight divergence. The drawn
values are in `draws.pt` and repeated in the manifest.

**The recycle draw comes from a generator that `torch.manual_seed` does not reach.**
`OpenFold3.synced_generator` is a private `np.random.default_rng` held as a model attribute. Seed
everything you can think of and a replayed forward will still pick a different recycle count unless
you restore that object's state too.

**`None` is not zero.** Their runner disables the confidence-head parameters on samples whose
confidence loss weight is zero, and those parameters then have no gradient rather than a zero one.
Comparing presence as a set comes before comparing any magnitude.

**The gradient here is before clipping.** The manifest records the clip coefficient their
`PerSampleGradManager` would apply. At world size 1 with `accumulate_grad_batches` 1 the
optimizer-facing gradient is exactly `grad * clip_coef`.

**Their model is not float64-clean, and the reference works around it.** `run_trunk` ends with an
unconditional `.float()` and several modules wrap work in `autocast(dtype=torch.float32)`. Both are
precision floors for their bf16/fp32 paths and both downcast a float64 graph, so the reference
neutralises them for the duration of one forward. Every change only ever removes a downcast. The
finite-difference check is run against this same forward, not the unpatched one.

Full method and results: `~/.coworker/state/of3t-reference.md`.
