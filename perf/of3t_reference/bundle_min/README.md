# BUNDLE-MIN

**Read this first, 2026-09-19.** The published gradient is now **`grads_f64_r0.pt`** on qb2 at
`/home/ttuser/of3t/bundle_min/`, sha256
`89457d8977327699c84fc90741a013bc369f835dea6008676492c786fb87f113`, recomputed on the host after
the transfer. `grads_f64_recycles0.pt` and `grads_f64_recycles3.pt` are **withdrawn**: they were
taped with the Pairformer's rate-0.25 Dropout live, so each published one draw.

Three things changed, and only the first was asked for:

1. **Dropout in eval and at rate 0.** The mask is drawn from the device's default generator, and
   an RNG-state snapshot restores a state, not a value, so a train-mode tape is not reproducible
   across devices.
2. **Deterministic kernels pinned.** With dropout already off, two fresh processes still
   disagreed: worst 1.985 relative L2 over 55 of 4,147 tensors, every one a `layer_norm_z.bias`
   whose gradient is a sum that cancels, so a 1e-16 change of reduction order reads O(1)
   relative. With `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`:
   4,147 of 4,147 bit-identical, worst 0.0, same file hash.
3. **The draws are replayed, not sampled.** Putting Dropout in eval removes 61 consumers of the
   CUDA generator, so the diffusion noise changed (`23.705, 5.853, 10.730, ...` became
   `2.612, 1.467, 5.936, ...`). PROTOCOL 4a makes the draws inputs to the update rule, so the
   published run consumes `draws_recycles0.pt`, the draws every consumer already holds.

Numbers: loss 1.6422035029890711, gradient global norm 3.707776369277739, 4,138 of 4,147 tensors
non-zero, 0 absent, clip coefficient 1.0. Finite differences at h = 1e-4 over 16 stratified
entries: worst 1.1678e-03, median 2.2650e-04, against PROTOCOL 3d's 5.0e-02 bar. The earlier
2.1817e-03 described the withdrawn artifact and does not carry over.

It does **not** land on `of3t-gradients`' pre-registered loss 1.631143239324149 and norm
3.727845454375: the non-zero count matches exactly, the two numbers are 6.8e-03 and 5.4e-03 away.
Replay closed most of the gap (before it, loss 1.60736306238088 and norm 4.305010199537414) and
what remains is open, between an A100 float64 run and a qb2 CPU float64 run of the same graph on
the same draws.

Regenerate with `perf/of3t_reference/rebuild_r0_replay.sh`; ship with `ship_to_qb2.sh`.

---

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
