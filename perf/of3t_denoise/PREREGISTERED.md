# of3t-denoise: pre-registration

Written 2026-09-23 ~13:50Z on branch head 2d34d6fa4 (wk/of3t-inproj's head, which carries the
D253 trunk masks and the D255 resolved-term mask), before any gradient of the denoise arm was
computed on this code. Bars below do not move after numbers exist.

Instrument: `trainfwd_run.py --arm full --denoise` through `perf/of3t_denoise/devstep.py`
(fullstep64's devstep with `--denoise` passed through), fullstep64's batch
`batch_step003.pt` (sha256 3c32597a...6285f, 56 real of 384), its rollout draws `draws.pt`
replayed, seed 20260922, qb2 p300c card 2. The denoise arm's sigma and noise come from
`np.random.default_rng(20260922)` inside the adapter; the float64 reference draws them with
the same generator and the same calls, and both record sigma and the noise sha256.

## Predictions

P1 finiteness. The 3 non-finite gradients (`sampler.dc.w_lin_z/w_lin_s/w_lin_n`) were read on
the trunk that ran unmasked over 328 pad tokens (D253), which carried 5-7x float64's gradient
energy in `msa_module`. PREDICT: every gradient finite at 64 and at 384 on this head.
Confidence moderate (~60 %). Falsifier: any non-finite tensor at either width. If it fires,
the op that produces the first non-finite value is found before anything is changed.

P2 magnitude. PREDICT: max|g| per top-level module well under the pre-D253 1.7e+05 (under
1e+03), and the device total |g|^2 within 3x of the float64 denoise reference's.

P3 D256. PREDICT: on the pre-fix tree, exactly 84 device weights exist after the taped forward
that the registering walk did not see, all `sampler.dm.enc_at._wc` and `sampler.dm.dec.at._wc`
(3 blocks x 14 lazily uploaded tensors x 2 atom transformers). The guard raises on that tree.
After the fix: 0 unregistered, and all 84 carry a non-zero gradient.

P4 diffusion module vs float64, 384 wide (rel = ||g_dev - g_f64|| / ||g_f64||, concatenated
per section, A43 residual published). PREDICT: device diffusion-module rel in [0.03, 0.30];
upstream bf16's own in [0.01, 0.15]; device no more than 3x bf16 on the module. Ordering:
conditioning (`diffusion_conditioning`, which reads the bf16 trunk pair) highest of the four
sections, the atom encoder/decoder lowest.

P5 terms. PREDICT: six of eight terms fire (mse, smooth_lddt, distogram, resolved, pae, pde).
`bond` does not: its weight is 0 in upstream's initial_training stage. `plddt` does not: its
label is a function of the prediction and the batch cannot carry it (adapter docstring).

P6 placed-but-empty: 0 inside the diffusion module after the D256 fix.

P7 A/A: two device runs of the fixed code bit-identical on every tensor.

P8 trunk. With the denoise cotangent entering s_trunk / z_trunk the trunk's rel moves; PREDICT
it stays under 0.30 (fullstep64 ON read 0.1305 without it).

## Bars (fixed now)

- FINITE: every gradient finite at 64 and 384. A miss is a STOP on the default flip.
- A40: the float64 denoise reference replays the rollout draws with 0 mismatches, refreshes
  its loss to <= 1e-12, and a central finite difference along g/||g|| (rollout frozen, the
  denoise sigma and noise fixed) agrees to rel <= 1e-6 at its best h. No ratio before it passes.
- REGISTERED: guard fires on the pre-fix tree; 0 unregistered device weights on the fixed tree.
- GO requires FINITE, A40, REGISTERED, diffusion-module rel <= 0.50 with cos >= 0.90, and every
  diffusion section at rel <= max(0.50, 3 x bf16's rel on that section). The 3x-bf16 reading is
  reported per section as HIT/MISS whatever the verdict.
- Tie: two rels within 5 % of each other are a tie (fullstep64's convention).
