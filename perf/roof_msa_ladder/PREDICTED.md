# roof-msa-ladder — the depth-axis audit, and the prediction, both before the device runs

Branch `wk/roof-msa-ladder`, tip of `main` at `3df8e8cd4`. Nothing here was measured on hardware;
the numbers below are predictions and the audit is a read of the source with line numbers.

## PART 1 — the audit the brief demanded, on all four units, not one

The question: does any reduction / softmax / mean / normalisation on the MSA depth axis divide by,
iterate over or normalise against the PADDED extent rather than the true row count `n_msa`? If any
does, shrinking the pad changes the model's answer and this row is a STOP.

Answer: **no. There is exactly one reduction over the depth axis in the whole MSA track, and it
divides by `n_msa`.**

| unit | every depth-axis operation it performs | divides by | line |
|---|---|---|---|
| `OuterProductMean.__call__` | `z = sum_s a_is b_js`, one matmul contracting S | `n_msa if n_msa is not None else S` | `tenstorrent.py:9743` |
| `OuterProductMean._small_depth` | the same contraction, one pass per row | `n_msa if n_msa is not None else S` | `:9472` |
| `PairWeightedAveraging.__call__` | NONE. Its only softmax reduces the TOKEN axis | n/a | `:9246` |
| `Transition.__call__` (msa_transition) | NONE. layer_norm over channels, 3 linears over channels | n/a | `:7789` |
| `MSALayer.__call__` | NONE of its own; two `add_` per row, then OPM | passes `n_msa` down | `:9907`, `:9914` |
| `MSA.__call__` | NONE. one per-row `linear` + a per-row `add_` of the `emb` projection | passes `n_msa` down | `:9984` |

Detail, unit by unit.

**OuterProductMean.** `scale = 1 / (n_msa if n_msa is not None else S)` at `:9743`, applied to `z`
before `proj_o`. The padded rows reach `z` through `a` only, and `a` is masked to zero inside
`project_ab`: `ac = ttnn.multiply_(ac, maskc)` at `:9544`, with `maskc` the depth mask built at
`:11067-11071` as `msa_mask[:n_msa] = 1.0` over a `zeros(padded_msa, 1, 1)`. So a padded row
contributes `0.0 * b_jd` to the sum and is excluded from the divisor. **Shrinking the pad shortens
a reduction whose extra terms are exactly 0.0.** That is the claim the brief asked to be verified,
and it holds.

**`_small_depth`.** Same divisor, same variable, `:9472`. It is also OFF by default:
`_OPM_SMALL_DEPTH = env_flag("TT_BIO_OPM_SMALL_DEPTH", False)` at `:123` and
`OPM_SMALL_DEPTH_MAX` defaults to 8 at `:124`, so no rung of a 32..1024 ladder can route into it
unless someone sets both env vars. Noted rather than ignored: it is the one path in the track that
is deliberately not bit-exact, and the ladder does not open it.

**PairWeightedAveraging.** The unit the brief flagged as the risk ("PairWeightedAveraging's softmax
over rows") does not have a softmax over rows. `_softmax_over_tokens` at `:9246` takes
`proj_z(z)`, which is `[tokens, tokens, n_heads]` — `z` is the PAIR tensor and carries no depth
axis at all — permutes `(2, 0, 1)` to `[n_heads, tokens, tokens]` and reduces `dim=-1`, the token
axis. The attention weights are a function of `z` alone, which the code already states at `:9276`
("A function of `z` alone, so it does not depend on the MSA depth"). Everything else in the unit is
per MSA row: `norm_m` reduces channels (`:9229`), `proj_m`/`proj_g` are channel linears
(`:9296`, `:9314`), the weighted average at `:9302` contracts the TOKEN axis, and the depth
blocking at `:9386` splits rows into independent blocks with no cross-block accumulation.

**Transition (msa_transition).** `swiglu` at `:7788`: `layer_norm` over the channel axis, `fc1`,
`fc2`, `fc3` over the channel axis. No depth reduction, no depth-dependent constant.

**`n_msa` threading, both callers.** Non-resident path: `n_msa = m.shape[1]` at `:11039`, cached at
`:11072` and handed to `MSAModule`'s MSA call at `:11097`. Resident trunk: `n_msa_arg` at `:11567`,
stored at `:11636`, passed at `:11724`. Both set it to `None` when there is no pad, which makes the
divisor `S` — and with no pad `S == n_msa`, so the two branches agree.

**Consequence for bit-exactness.** Because the real rows are rows `0..n_msa-1` and the padded rows
are all above them, the matmul's K blocking puts the real rows in the same leading 32-wide K blocks
at every rung. Extra K blocks contain only zeros. So the arithmetic over the real terms is not just
mathematically identical, it is plausibly bit-identical, and md5 is a free check rather than a
requirement. Predicted below.

## PREDICTED (written before any hardware ran)

1. **Bit-exact.** The structure md5 at 512 aa is IDENTICAL between `MSA_PAD_MULTIPLE = 1024` and a
   64-rung ladder. Reason above. Confidence moderate, not high: a different padded extent can pick a
   different matmul program config, and a different `in0_block_w` reassociates the real terms among
   themselves. If it is not bit-exact the deviation should be at the last bf16 bit and the Angstrom
   number should round to 0.00, well inside the ~0.02 A this row is allowed to spend of the
   campaign's 0.105 A remaining budget.
2. **Warm win at the published 512 aa cell: 1.03x to 1.05x, i.e. 0.5 to 0.8 fold-seconds of the
   17.340 s cell.** Derivation, not a guess: `b2z2-msa-layer-census` fits one `MSALayer` at
   `139.47 ms + 0.0995 ms/padded row` (Wormhole), so at 1024 rows the row-proportional term is
   101.9 ms of 241.4 ms = 42.2 % of the layer, and 16 calls carry 1.63 s of that fold's 3.83 s MSA
   track. Scaling that 42.2 % onto the Blackhole cell's 1.900 s MSA track gives 0.80 s of
   row-proportional cost, of which a 1024 -> 64 rung removes 93.75 % = **0.75 s**, i.e. 1.045x.
   I predict the measured number lands BELOW that arithmetic, at 0.5-0.65 s, because the campaign
   has twice measured that deleted traffic does not return time at the roof rate.
3. **Not all of the brief's 1.305 s.** The four units carry 1.305 s above roof at the cell, and the
   depth axis is one of three axes in their shapes. Anything at or above 1.0 s would mean the fit
   above is wrong in my favour and I would go looking for the error before reporting it.
4. **Compile cost: 10-40 s for the first fold at a new rung, once per process, and it is the MSA
   track's programs only.** The pairformer, diffusion and atom programs do not see the depth axis
   and are not recompiled. This is the number that decides the ladder's rung count.
5. **The win does not scale to a deep MSA, and the published fixture is the best case.**
   `cdk2x2_512.a3m` has **35 rows**, `build_fold` caps at `max_msa_seqs=8192`
   (`scripts/gpu_vs_tt/tt_baseline.py:374`), and a ColabFold search on a real target routinely
   returns thousands. A ladder that tops out at 1024 and falls back to multiples of 1024 above it
   is worth 0.00 s on any target with more than 1024 rows: 1100 rows pads to 2048 with and without
   it. The win is confined to the shallow regime, and I expect to have to say in the state doc that
   the published cell over-reports what a user sees.
6. **Kill criterion I expect to be closest to failing:** none of the four. I expect the row to pass
   its own bar and to be worth less than the brief's 1.305 s framing implies.

## Kill criteria, restated so the verdict is checkable

- Any depth-axis reduction found to normalise against the padded extent -> STOP, which is a pass.
  **Not triggered: the audit above clears all four units.**
- Warm win under 1.01x at 512 aa on two interleaved sessions -> kill.
- Added compile cost above the warm win for a single-fold service request -> kill.
- Structure past the 0.60 A bar at 512 aa (seed floor 1.84 A) -> kill.
