# of3t-ieatom: predictions, committed before the fix arm runs

Tree: `wk/of3t-ieatom` 06f26f434 (guard 60c4c5470 + routing). Arm: 64-token denoise step,
`batch_step003_t64.pt`, exact on, fullstep64's draws, qb2 card 3. Scored with
`perf/of3t_fullstep64/score.py` against `ref384/grads_f64.pt` (sha 667e5d37) with
`ref384/grads_bf16.pt` as the bar, exactly as SCORE_PW64.json. The comparison arm is PW64F.

## What is known before the arm

Read from the two references only, no arm involved:

- The 93 carry 2.08e-3 of float64 squared mass, 8.24e-6 of the step's total.
- Upstream bf16 against float64 on the 93, concatenated: rel **1.43**. Per tensor: median 1.75,
  min 0.28, max 714. bf16 does badly here.
- `linear_q.0.weight` is 76.3 % of the 93's mass (bf16 rel 1.50); `linear_ref_pos` 11.4 %,
  `linear_ref_atom_chars` 5.6 %.
- The five glue linears downstream of `s_input` read 0.351 on PW64F against bf16 0.551.

## Predictions

1. Unread: 0 of the scored 4170 (the 93 and the 12 template tri-att tensors now placed).
2. Placed-but-empty: 0.
3. The 93, concatenated rel vs float64: **0.6**, 80 % interval 0.3 to 1.2. Better than bf16
   (1.43) with probability 0.85. Within 3x bf16 (4.3) with probability 0.98.
4. `linear_q.0.weight` alone: 0.5 (interval 0.25 to 1.0).
5. Per tensor: at least 60 % of the 93 at or better than their own bf16 rel.
6. Loss: moves, because `s_input` is now computed in bf16 on the card instead of host fp32 then
   bf16. |Δloss| / loss < 1e-3 against PW64F's 1.7933790552496434.
7. Global rel stays at 0.1348 +- 0.003; the trunk head stays within 0.02 of 0.1947.
8. No section gets worse than PW64F by more than 3x its bf16 rel. Sections downstream of
   `s_input` (input_embedder glue, msa_module_embedder, diffusion_conditioning) move; the
   diffusion stacks move less than 0.01.

A miss on 3 to 8 is a finding, reported as measured. The tolerances above are not moved.
