# of3t-go384: predictions, committed before the arm runs

Tree: `wk/of3t` a9dd90e9e (pass-426 composition). `git diff --stat 06b91c5ce HEAD -- tt_bio/` is
empty, so `tt_bio/` is of3t-ieatom's scored tree. Arm GO384: one denoise training step on
devstep.py's default 384-wide batch (`bundle_min/batch_step003.pt`, the batch DN384R ran), exact
on, fullstep64's draws (byte-identical to `ref384/draws.pt`), qb2 card 3. Scored with
`perf/of3t_fullstep64/score.py` against `ref384/grads_f64.pt` with `ref384/grads_bf16.pt` as the
bar. Comparison arm: DN384R (`perf/of3t_denoise/SCORE_DN384R.json`, tree 9e6de9d0a).

## What is known before the arm

- DN384R: global rel 0.1558 against bf16 0.6025, mass at or better than bf16 0.971, unread 105
  (the input-embedder atom encoder, D263), placed-but-empty 9 (msa_att_row, D264).
- PF64F (64 tokens, this `tt_bio/`): unread 0, placed-but-empty 0, multi-placed 0, global rel
  0.1348, mass 0.990.
- On both, two confidence sections sit past 3x their own bf16 rel:
  `aux_heads.pairformer_embedding` 11.44 against bf16 0.312 (36.7x on DN384R, 35.0x on PF64F) and
  `aux_heads.experimentally_resolved` 0.130 against 0.0228 (5.7x on both). Every commit under
  `tt_bio/` since DN384R is D263, D264 or a merge, none touching the confidence heads. The
  confidence head carries 4e-10 of the float64 mass, so it cannot move the global figures.

## Predictions

1. `scored.unread_n` 0 (D263 places the 105).
2. `placed_but_carried_nothing.n` 0 (D264 fills the 9).
3. `multi_placed` 0.
4. Global rel 0.15, 80 % interval 0.13 to 0.18, against BF16 0.6025. At or below bf16 with
   probability 0.99.
5. `mass_at_or_better_than_bf16` 0.98, interval 0.96 to 0.995. At or above 0.95 with probability
   0.95.
6. `aux_heads.pairformer_embedding` stays past 3x its bf16 (probability 0.95) and so does
   `aux_heads.experimentally_resolved` (0.9). Every other section within 1x its bf16, and
   `input_embedder` and `msa_module` move by more than 0.02 from DN384R because D263/D264 changed
   them.
7. Loss within 1e-2 relative of DN384R's 1.7811428 (D263 moves the atom encoder to fp32).

Predicted verdict: the five pre-registered GRADIENTS conditions hold, and the row reads STOP on
the 3x clause, first failing section `aux_heads.pairformer_embedding`. A miss on any of 1 to 7 is
reported as measured. None of these tolerances are moved after the arm.
