# of3t-inproj: pre-registration

Written before any gradient from the fixed code existed. The only gradients already taken on this
row are the two root-cause diagnostics of the UNFIXED code at 64 tokens (`diag_f3.py`,
`DIAG_T64_before.json`), which established the cause below. Tolerances and bars are fullstep64's
and are not moved.

## Root cause (measured before this file, on the unfixed code)

`OpenFold3Forward.__call__` registers parameters with a walk run BEFORE the taped forward, and
registers only once. Every TriangleMultiplication's fused in-projection is built from host torch
on first use inside that forward, so none of it exists when the walk runs. At 64 tokens: first walk
3636 tensors, walk after backward 3952; all 232 trimul cache entries (116 trimuls x one `_gp_cache`
chunk + one `_gp_gout_cache`) were minted after registration, 0 are leaves, 0 hold a gradient.
Under the tape `_gp_gout_cache` is built and never read (the split matmul declines while taping),
so the tensor the forward multiplies is the `_gp_cache` chunk, as an untracked constant. The other
84 late tensors are the diffusion atom transformer's `_wc` cache, minted by the untaped rollout.

## Fix under test

Training only: each trimul holds `g_in` and `p_in` as two device leaves (`train_in_proj`), set on
the built model before registration, and every fused chunk is cut from them per call with
`ttnn.slice`/`ttnn.concat`, so the tape sums every chunk into its leaf and nothing is cached. F2:
the resolved term averages over `atom_mask` (token scope: real tokens), as upstream does.

## Predictions (384 wide, 56 real, ON = exact training ops, against the regenerated float64)

- P1 coverage: placed-but-empty goes from 473 to 0 (p 0.8). If not 0, the remainder is the
  diffusion `_wc` class, which carries no float64 mass on this step.
- P2 in-projection rel: each of `linear_a_p`, `linear_b_p`, `linear_a_g`, `linear_b_g`, pooled
  over its blocks per stack, lands in 0.05-0.30 for ON and within 3x of upstream bf16's own figure
  for the same tensor (p 0.7). The in-projection is ordinary trunk arithmetic now; there is no
  reason for it to sit further from float64 than the trunk around it.
- P3 global rel moves little: ON 0.1455 -> 0.135-0.150 (p 0.75). The in-projection held 9.7e-4 of
  float64 mass scored as zero, worth about 0.004 of global rel.
- P4 trunk rel: ON 0.1305 -> 0.115-0.135 (p 0.7).
- P5 confidence rel falls from 0.948 toward the bf16 bar: ON 0.15-0.60 (p 0.6). fullstep64 put the
  confidence head's gap to bf16 on the pad-averaged resolved term.
- P6 pad invariance (the F2 control): ON under PADOFF (+3 on pad restype/profile/deletion_mean)
  gives a confidence rel within 1e-3 of unperturbed ON (p 0.85), and every gradient tensor
  bit-identical (p 0.5).
- P7 forward unchanged by F3: the distogram term's value is bit-identical to fullstep64's FIXON
  (slicing and concatenating bf16 is exact), p 0.9. The loss changes only through the resolved term.
- P8 dx through the in-projection was never dropped: the trimul `norm_in` weights, whose gradient
  reaches them only through the in-projection's input cotangent, carry a gradient at every trimul
  and score within 2x of the trunk rel (p 0.8).
- P9 ON A/A: 2 ON runs bit-identical on every tensor (p 0.95).

## Gating (A40), before any ratio

Regenerated float64 reference: draw replay 0 mismatches, loss refresh diff <= 1e-12, central FD
along g/||g|| at h 1e-3/1e-4/1e-5 with rel err <= 1e-6 at the best h.
