# of3t-paedraws: predictions, committed before draw 1 runs

PROTOCOL A44 grades every section of the 384-wide step on `mean_k rel(ours, k) / mean_k rel(bf16, k)`
over six denoise draws, each side against float64 at the same draw, bar 3x. This file fixes the
draws, how each is set, and what I expect, before any of draws 1 to 5 exists.

## The draws

| k | noise seed | sigma | EDM scale | source |
|---|---|---|---|---|
| 0 | 20260922 | 0.30396 | 10.828 | CF384 as banked: `perf/of3t_confpfe/SCORE_CF384.json`, `ref384c`, `grad_CF384.pt`; not re-run |
| 1 | 1 | 8.0927 | 0.019175 | this row |
| 2 | 2 | 6.3992 | 0.028327 | this row |
| 3 | 3 | 102.92 | 0.0040007 | this row |
| 4 | 4 | 1.8129 | 0.30819 | this row |
| 5 | 5 | 1.4473 | 0.48131 | this row |

sigma and the per-atom noise are `tt_bio.train.openfold3.denoise_draw(seed, 422)`, a function of the
seed alone (numpy `default_rng`); the column above is that function evaluated, not a run. Batch
(`batch_step003.pt`, sha256 3c32597a...), checkpoint (`of3-p2-155k.pt`), build seed, crop and the
rollout draws (`/home/ttuser/of3t_fullstep64/draws.pt`, replayed by every arm) are CF384's. Tree:
`tt_bio/` is 78687c9e4's (`git diff 78687c9e4 -- tt_bio` empty on this branch).

How the seed is set, per side, with nothing under `tt_bio/` touched:
- float64 and bf16 reference: `perf/of3t_fullstep64/ref_step.py --seed k --replay-draws draws.pt`.
  With a replay file, `--seed` reaches only `denoise_draw`; the rollout reads the replayed draws.
- device: `perf/of3t_paedraws/devstep.py --seed k`, a wrapper around confpfe's `devstep.py` that
  appends `--seed k` to `trainfwd_run`'s argv, which hands it to `OpenFold3Forward(seed=k)` and on
  to `denoise_draw`. fullstep64's `devstep.py` rebuilds that argv without a seed, which is why the
  wrapper exists. The rollout still replays `draws.pt` (`_gen_rollout`'s `torch.randn` calls go
  through `Draws`, mismatch count recorded per draw).
- per draw, the device's recorded `denoise_sigma` must equal the float64 reference's `sigma`, and
  both replay mismatch counts must be 0, or the draw is not scored.

Scored with `perf/of3t_fullstep64/score.py` unedited; per-draw sections from
`perf/of3t_orchestrator/sections/section_ratio.py` run unedited (imported with `runpy`).

## Predictions

What moves between draws is the denoise arm: `pred_xyz` feeds the pae and pde labels (binned,
`tt_bio/train/losses.py:323`) and the diffusion loss is scaled by the EDM weight above. At
sigma 0.3 the network output enters `pred_xyz` through c_out of about 0.30; at sigma 6 to 103 it is
about 15, so any arithmetic error in the diffusion module reaches the structure some 50x larger on
draws 1 to 3 than on draw 0, on both sides.

1. `aux_heads.pae` mean-over-6 ratio 1.1, 80 % interval 0.6 to 2.5; within 3x with probability
   0.8. paez's null has ours at +1.4 sd and bf16 at -1.3 sd on draw 0, and ours had the smaller
   structure error (0.035 A against 0.052 A RMS); if that ordering holds on the other draws, ours
   carries fewer label flips than bf16 on average.
2. `aux_heads.pde` mean-over-6 ratio 1.2, 80 % interval 0.6 to 2.5; within 3x with probability 0.8.
3. The draw-0 row of both is CF384's, 6.07x and 2.62x, published beside the mean.
4. `aux_heads.distogram` and `aux_heads.experimentally_resolved` read the same rel on every draw,
   ours and bf16 alike (probability 0.85): the distogram head reads only z and ground truth, and the
   resolved head's labels come from ground truth through the replayed rollout. The trunk's z
   itself does not change with the draw (the forward is the same), so the ratios stay 1.44x and
   1.04x. A difference would mean the draw leaks into the forward.
5. The five GRADIENTS fields hold on every draw (probability 0.7): unread 0, placed-but-empty 0,
   multi-placed 0, global rel at or below bf16's, mass at or better than bf16 at or above 0.95. The
   risk is the last: at EDM scale 0.004 to 0.03 the diffusion loss's share of the step falls and
   the mass moves toward the trunk and heads, where the margin is not known.
6. No section other than pae and pde past 3x on its mean (probability 0.85). The diffusion module
   sections move with sigma on both sides, and nothing about the draw favours one side.

Predicted verdict: GO (probability 0.55, the product of 1, 2 and 5 with some correlation). If a
mean exceeds 3x, D267 (or pde) stays open as a defect under the unchanged clause. None of these
bars move after the draws.
