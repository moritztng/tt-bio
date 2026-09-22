# of3t-trainfwd — addendum to PREDICTION.md (2026-09-22, source reading, no result exists)

This is a reading of upstream 0.4.3's source made after `PREDICTION.md` was committed and
before any scoring run. It is not a result and PREDICTION.md is not edited; P3 stands as
written and will be reported against as written.

## The census's stated reason for `diffusion_rollout` is wrong

`perf/of3t_gradients/coverage_census.json` gives the reason:

> the rollout is a differentiated path through the diffusion module and needs a model forward

It is not differentiated. Upstream runs it under `torch.no_grad()`:

    of3pkg043/openfold3/projects/of3_all_atom/model.py:381-393
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="cuda", dtype=torch.float32),
        ):
            noise_schedule = create_noise_schedule(...)
            atom_positions_predicted = self.sample_diffusion(...)

and it is the MINI rollout in training, not the full one (`model.py:360-371`,
`no_mini_rollout_steps` / `no_mini_rollout_samples` when `self.training`). Its output feeds
`self.aux_heads`, which is where the gradient starts.

So the diffusion module is trained by the one-step denoise objective, not through the rollout
recursion, and the rollout's contribution to a parameter gradient runs through the confidence
heads' inputs. An arm that detaches the rollout output is therefore what upstream ALREADY does,
which means P3's OFF arm and upstream's ON arm are the same computation through the denoiser.
P3 as phrased predicts a difference that upstream's own arrangement reads as zero there.

That is the finding, and it lands on the census rather than on this row: `diffusion_rollout`
cannot be covered in the sense the census names, because that sense does not exist in the model.
What can be covered is the sense the model has — the rollout fires, produces the coordinates the
confidence heads score, and a parameter gradient moves against an arm that scores the same heads
on a prediction the rollout did not produce.

## The shipped sampler severs the tape every step

`tt_bio/openfold3_sample_diffusion.py:171` reads the denoised coordinates back to host inside
the rollout loop (`ttnn.to_torch(xl_denoised_dev)`), and lines 124-125 do the same for the atom
mask and the initial positions. The EDM update is host arithmetic between device denoiser calls.
So `OF3SampleDiffusion.__call__` cannot carry a tape through the recursion whatever the tape
does, and `tt_bio.taped_ttnn.tape()` does not change that: it rebinds `ttnn` inside tt-bio's
modules, and `to_torch` leaves the device.

This is consistent with upstream rather than a defect — the rollout is `no_grad` there too — but
it is the reason a training forward cannot be `OpenFold3.fold()` with a tape opened around it,
and it is what the adapter has to be shaped around.
