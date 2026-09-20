# BUNDLE-MIN — the step-1 gradient reference

One fixed batch, the weights it was taken at, the per-parameter float64 gradient, the random draws
that step made, and a finite-difference validation of the gradient itself. This is what instrument
A is measured against.

## Which file is the reference

**`grads_f64_r0.pt`**, on qb2 at `/home/ttuser/of3t/bundle_min/`, sha256
`89457d8977327699c84fc90741a013bc369f835dea6008676492c786fb87f113`, recomputed on the host after
the transfer.

**Check that digest before you read a byte.** Three gradients have been published under this
bundle and two of them are dead:

| digest | file | status |
|---|---|---|
| `89457d89…fb87f113` | `grads_f64_r0.pt` | **the reference** |
| `1a6af8bb…8b4b4b74` | `grads_f64_recycles0.pt` | withdrawn: taped in train mode, so the rate-0.25 Pairformer Dropout was live and it published one draw |
| `2a19e011…2c49ee657` | `grads_f64_recycles3.pt` | withdrawn for the same reason |
| `b30c0938…880cc1f0` | `grads_f64_r0_selfdrawn_SUPERSEDED.pt` | superseded: r = 0 and dropout off, but it drew its own 45 `randn` values instead of replaying `draws_recycles0.pt` |

The last one is the easy mistake. It sat on qb2 for a while named `grads_f64_r0_rebuild.pt`, it is
r = 0, and its dropout is correctly off — the only thing wrong with it is that it did not replay
the draws, and that alone moves the loss by 3.48e-02 and the global norm by 0.597 on the same
weights, same batch, same recycle count. It is kept, under a name that says what it is, because
that difference is the measured cost of not matching the draws.

## The numbers

Loss 1.6422035029890711, gradient global norm 3.707776369277739, 4,138 of 4,147 tensors non-zero,
0 absent, clip coefficient 1.0. Finite differences at h = 1e-4 over 16 stratified entries: worst
1.1678e-03, median 2.2650e-04, against PROTOCOL 3d's 5.0e-02 bar, so the reference sits 43x inside
the bar it will be compared at. The earlier 2.1817e-03 described the withdrawn train-mode artifact
and does not carry over.

Reproduced: two fresh processes, runs C and D, gave 4,147 of 4,147 bit-identical tensors, worst
0.0, median 0.0, and the same file hash (`evidence/det_selfdrawn/reproduction_A13.json`).

It does **not** land on `of3t-gradients`' pre-registered loss 1.631143239324149 and norm
3.727845454375. The non-zero count matches exactly; the two numbers are 6.8e-03 and 5.4e-03 away.
Replay closed most of the gap and what remains is open — an A100 float64 run against a qb2 CPU
float64 run of the same graph on the same draws.

## Three things changed from the withdrawn artifact, and only the first was asked for

1. **Dropout in eval and at rate 0.** The mask is drawn from the device's default generator, and
   an RNG-state snapshot restores a state, not a value, so a train-mode tape is not reproducible
   across devices.
2. **Deterministic kernels pinned.** With dropout already off, two fresh processes still
   disagreed: worst 1.985 relative L2 over 55 of 4,147 tensors, every one a `layer_norm_z.bias`
   whose gradient is a sum that cancels, so a 1e-16 change of reduction order reads O(1)
   relative. With `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`:
   4,147 of 4,147 bit-identical.
3. **The draws are replayed, not sampled.** Putting Dropout in eval removes 61 consumers of the
   CUDA generator, so a self-drawn run gets different diffusion noise. PROTOCOL 4a makes the draws
   inputs to the update rule, so the published run consumes `draws_recycles0.pt`, the draws every
   consumer already holds. Run E is what skipping this costs.

## Files

| file | what it is |
|---|---|
| `MANIFEST.json` | THE PUBLICATION. Hashes, versions, the upstream commit, the draws, the validation, and every superseded digest. In git, read this one. |
| `grads_f64_r0.pt` | **the reference gradient.** On qb2. |
| `draws_recycles0.pt` | its draws: recycle count, diffusion noise, every `torch.randn` and `random.random` value, full RNG state. On qb2. |
| `grad_presence_r0.json` | per parameter, whether it had a gradient at all. On qb2 and in git as `grad_presence.json`. |
| `batch_step003.pt` | the frozen batch. On qb2. |
| `w0_r0_rebuild.pt` | the float64 weights the gradient was taken at. Identical across runs C, D and E. On qb2; needed for the trajectory, not for instrument A. |
| `run_record_r0_replay.json` | run C's own record, every finite-difference check included. In git. |
| `evidence/` | A13's reproduction, with and without deterministic kernels. In git. |
| `hsweep.json` | the step-size sweep showing why a recycled trunk gradient cannot be finite-difference validated. In git. |

Git carries the manifest, the hashes and the generator, never a `.pt`.

**Read-only once published.** If something needs to change, publish a new bundle with a new
manifest; a row that verified a hash must be able to tell.

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

Regenerate the gradient with `perf/of3t_reference/rebuild_r0_replay.sh` on a CUDA box, with
`batch_step003.pt` and `draws_recycles0.pt` carried in by hash. Ship with `ship_to_qb2.sh`.

Full method and results: `~/.coworker/state/of3t-reference.md`.
