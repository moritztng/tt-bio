# of3t-go384: result

One 384-wide denoise training step on `wk/of3t` a9dd90e9e (`tt_bio/` identical to 06b91c5ce),
qb2 card 3 (p300c), exact on, fullstep64 draws (61 of 61 consumed, 0 mismatches), scored by
`perf/of3t_fullstep64/score.py` against `ref384/grads_f64.pt` with `grads_bf16.pt` as the bar.
Artifact: `SCORE_GO384.json`. AICLK median 1350 MHz; 1 of 511 samples below 1200, the 800 at
22:17:24 before the forward started.

## Pre-registered GRADIENTS conditions (all hold)

| field | value | bar |
|---|---|---|
| scored.unread_n | 0 | 0 |
| placed_but_carried_nothing.n | 0 | 0 |
| multi_placed | 0 | 0 |
| global.rel | 0.1558 | <= BF16 0.6025 |
| mass_at_or_better_than_bf16 | 0.9711 | >= 0.95 |

Loss 1.7811428176578317, bit-identical to DN384R. Scored 4152 of 4152 reference tensors.

## Per section, rel against float64

| section | GO384 | DN384R | bf16 | GO384 / bf16 |
|---|---|---|---|---|
| aux_heads.distogram | 0.1478 | 0.1478 | 0.1055 | 1.40 |
| aux_heads.experimentally_resolved | 0.1302 | 0.1302 | 0.0228 | **5.70** |
| aux_heads.pairformer_embedding | 11.4389 | 11.4389 | 0.3121 | **36.66** |
| diffusion_module.atom_attn_dec | 0.1867 | 0.1867 | 0.6018 | 0.31 |
| diffusion_module.atom_attn_enc | 0.1899 | 0.1899 | 0.5110 | 0.37 |
| diffusion_module.diffusion_conditioning | 0.1057 | 0.1057 | 0.3317 | 0.32 |
| diffusion_module.diffusion_transformer | 0.1642 | 0.1642 | 0.7079 | 0.23 |
| diffusion_module.layer_norm_a | 0.2198 | 0.2198 | 0.5585 | 0.39 |
| diffusion_module.layer_norm_s | 0.0411 | 0.0411 | 0.3210 | 0.13 |
| diffusion_module.linear_s | 0.1593 | 0.1593 | 0.4720 | 0.34 |
| input_embedder | 0.8246 | 0.4521 | 0.9266 | 0.89 |
| layer_norm_s | 0.2517 | 0.2517 | 0.4404 | 0.57 |
| layer_norm_z | 0.1860 | 0.1860 | 0.4457 | 0.42 |
| linear_s | 0.1589 | 0.1589 | 0.3884 | 0.41 |
| linear_z | 0.1753 | 0.1753 | 0.3960 | 0.44 |
| msa_module | 0.2748 | 0.2787 | 0.5593 | 0.49 |
| msa_module_embedder | 0.2910 | 0.2910 | 0.4925 | 0.59 |
| pairformer_stack | 0.2091 | 0.2091 | 0.3815 | 0.55 |
| template_embedder | 0.2140 | 0.2137 | 0.4041 | 0.53 |

DN384R left 105 input-embedder tensors unread. They are read now, which is why input_embedder
rises from 0.452 to 0.825 (still under its bf16 0.927).

## Defects (sections past 3x their own bf16)

Per tensor in `CONF_TENSORS_GO384.txt` (script `conf_tensors.py`, offline, reuses score.py).

- `aux_heads.pairformer_embedding`, 36.7x. The gradient is wrong, not just scaled. Across the
  section's 231 tensors the device/reference norm ratio has median 12.8 (p10 0.87, p90 26),
  per-tensor cosine median 0.26 (p10 0.04), and the best single rescale (2.01) still leaves
  rel 5.66. The error concentrates in the pair-stack layer norms of the two confidence
  pairformer blocks (`blocks.0.pair_stack.pair_transition.layer_norm.bias` carries 16 % of the
  section error, rel 27.5 against bf16 0.41), with `linear_i` and `linear_j` at rel 19. The
  section holds 4e-10 of the float64 mass, so it cannot move the global figures, but a
  confidence head trained on this gradient would be trained on noise.
- `aux_heads.experimentally_resolved`, 5.7x. `linear.weight` carries 95 % of the section error:
  rel 0.218 against bf16 0.039, norm ratio 0.88.

Both sections match DN384R to four digits, so no commit since DN384R touched them.

## Predictions (PREREGISTERED.md, a1d213e63)

Predictions 1 to 3 hold (0, 0, 0). 4 holds (0.1558, interval 0.13 to 0.18). 5 holds (0.971,
interval 0.96 to 0.995; the 0.98 point estimate was high). 6: both confidence sections are past
3x, as predicted, with two misses: `aux_heads.distogram` sits at 1.40x its bf16 rather than
within 1x, and `msa_module` moved 0.004 from DN384R, not the predicted > 0.02. 7 holds
trivially because the loss is bit-identical. D263 changed what the scorer reads, not the
forward, so the stated reason (the fp32 atom encoder moves the loss) was wrong.

## Verdict

STOP. All five pre-registered GRADIENTS conditions hold. The first failing condition is the 3x
section clause, on `aux_heads.pairformer_embedding` (36.7x), and
`aux_heads.experimentally_resolved` (5.7x) is past it too.
