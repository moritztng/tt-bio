# Pairformer block 47: the two levers are orthogonal, and the median ranks the arms backwards

Orchestrator independent recompute, from the row's own four block-47 instrument-A bundles
(`instrument_a_bundle_043{,spb}_block47_crop64_tb{shipped,off}.json`, all four against the
**same** frozen reference, `bundle.sha256 = 1d4ea922...`, verified identical across arms and
bit-identical `ref_norm` on every shared tensor). Recomputed under PROTOCOL **A14** (zero-reference
tensors excluded) and **A15** (reach reported as squared-norm share, not tensor count), on the
51 tensors common to all four arms.

| arm | median rel | over 5.0e-02 | reach, share of compared | reach, share of FULL block | worst |
|---|---|---|---|---|---|
| `scale_pair_bias` ON, `transpose_bias` shipped | 0.9439 | 45/51 | 12.336 % | **9.114 %** | 10.84 |
| `scale_pair_bias` ON, `transpose_bias` off | 0.9684 | 45/51 | 12.336 % | **9.114 %** | 13.12 |
| `scale_pair_bias` OFF, `transpose_bias` shipped | 0.2001 | **51/51** | 0.000 % | **0.000 %** | 33.08 |
| `scale_pair_bias` OFF, `transpose_bias` off | 0.2369 | **51/51** | 0.000 % | **0.000 %** | 33.27 |

## 1. The median and the reach rank the two arms in opposite directions

Turning `scale_pair_bias` off improves the median **4.7x** (0.944 -> 0.200) and takes the reach
from 9.1 % to **zero**: every one of the 51 tensors goes over the per-tensor bar. The arm that
looks better on the headline number is the arm under which nothing at all passes. This is A15
doing exactly the job it was written for, and it is a live trap here because the median is the
number that appears in summaries.

Nor is the 9.1 % a partial success. It is **one tensor** — `attn_pair_bias.layer_norm_z.weight`,
12.02 % of the compared mass, rel 0.029 — plus three crumbs totalling 0.31 %. No sub-module of
block 47 passes.

## 2. `transpose_bias` rescues blocks 0 and 23 and does nothing at block 47

Same recompute, same rules, at the three measured blocks (reach as share of compared mass):

| block | `transpose_bias` off | `transpose_bias` shipped | median off -> shipped |
|---|---|---|---|
| 0 | 75.900 % | **94.886 %** | 0.0736 -> 0.0121 |
| 23 | 30.047 % | **74.956 %** | 0.0998 -> 0.0192 |
| 47 | 12.336 % | 12.336 % | 0.9684 -> 0.9439 |

`transpose_bias` moves block 47's reach by **exactly zero** and its median by 2.5 %, while moving
block 23's reach by 45 points. Conversely `scale_pair_bias` moves block 47's median 4.7x and
blocks 0 and 23 by under 2 % (0.0121 -> 0.0121, 0.0192 -> 0.0192). **Two levers, two disjoint
populations of blocks.** Whatever block 47 has is not what the shipped `transpose_bias` fix
addresses, so no amount of tuning that lever will reach it.

## 3. The shipped arm hides the largest tensor it is being judged on

`attn_pair_bias.linear_z.weight` carries **26.12 %** of block 47's reference gradient mass
(ref_norm 1.684e-01; the full block is 1.0861e-01 squared-norm, the compared set only
8.0244e-02). Under `scale_pair_bias` **OFF** it is compared, and fails at **0.811**. Under
`scale_pair_bias` **ON** — the row's shipped default — it lands in `absent`: the value-bijection
cannot place it, presumably because the device folds the scale into the weight and the value no
longer matches.

So the shipped default **removes its single largest attn_pair_bias tensor from the instrument**,
and the removal is silent: `absent` tensors are outside the reach denominator, which is why the
row's own file reports 12.336 % rather than the honest **9.114 %**. The same +1 live-tensor
asymmetry appears at blocks 0 and 23, so across 48 blocks this is **48 tensors invisible to
instrument A whenever the shipped default is on**.

A further 4 tensors (`attn_pair_bias.mha.linear_{q,k,v}`, and `linear_q.bias`) are absent in
**both** arms — consistent with a fused QKV on the device side, unquantified here because the
bundles do not carry their reference norms. Until that is resolved the compared set is
**73.88 %** of block 47 in the best case.

## 4. A18: the gradient at block 47 is taken at a forward that disagrees

`forward_rel.z = 8.26e-02` at block 47, **above the 5.0e-02 per-tensor bar**; blocks 0 and 23 sit
under it at 2.86e-02 and 3.79e-02. Masked to the 56 real tokens of the 64-token crop it is
8.82e-03. All of the excess lives in the **pad**. Block 0 is the extreme case: `forward_rel.s` is
**3.39e-01** unmasked against 8.1e-03 masked, a 42x reduction — the device writes garbage into pad
rows and the mask hides it in the forward.

That is benign for the forward and **not established benign for the gradient**. A near-passing
block-0 median of 0.0121 is taken at the worst forward disagreement in the table. Whether the
backward is equally mask-clean is not measured by anything here, and it is cheap to measure:
re-run one block with the pad rows set to NaN and see whether any parameter gradient becomes NaN.

## 5. One module family is worst everywhere

The worst tensor in **all four** block-47 arms, and at blocks 0 and 23 in every arm, is
`attn_pair_bias.layer_norm_*`:

| block | arm | worst tensor | rel |
|---|---|---|---|
| 47 | spb off | `attn_pair_bias.layer_norm_a.weight` | 33.08 |
| 47 | spb on | `attn_pair_bias.layer_norm_a.weight` | 10.84 |
| 23 | spb off | `attn_pair_bias.layer_norm_*` | 5.74 |
| 0 | spb on | `attn_pair_bias.layer_norm_*` | 1.25 |

and the campaign's standing worst offender in the *other* stack is
`diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight` (87.82 under
D21, 1.291 at 1-of-48, 18.50 in the all-blocks device gradient). **The same module family is the
worst case in the pairformer and in the diffusion transformer, across two references, four arms
and three scopes** — and it is the module `scale_pair_bias` acts on. That is one suspect, not two,
and it is where a bisection should start.

## Reproduce

`block47_arms.json` carries all four arms: the common-set medians, both reach denominators
(compared set and full block), the `absent` lists and the forward residuals, together with the
shared `bundle.sha256` that establishes all four were scored against one reference.

Inputs, copied verbatim from the row's tree at
`tt-quietbox2:/home/ttuser/of3t_rebase/wt/perf/of3t_rebase/`, are alongside.

## 6. Depth is not the mechanism, and the 26.68x probe is sub-linear

`device_gradient_043all.json` carries two 48-long arrays, one per pairformer block, measured at
full scope (NP 448, no 64-token crop).

`accumulation_probe` runs 1.2510e-04 -> 3.3378e-03, Spearman **+0.998** against depth, 7
inversions in 47 steps — the "probe growing 26.68x" the campaign published at pass 116. Fitted in
log-log it is **k^0.765 at R^2 0.974**:

| model | growth over 48 blocks |
|---|---|
| random walk, `k^0.5` | x6.93 |
| **observed** | **x26.68** (`k^0.765`) |
| coherent sum, `k^1.0` | x48.00 |

Strictly between a random walk and a coherent sum, and **strictly below** a naive per-block error
budget with no cancellation. Amplification — a geometric instability — means exceeding `k^1`.
This does not. The per-block contributions partially cancel and the stack **damps** relative to a
naive sum, so 26.68x is not the warning sign it was read as.

`forward_rel` per block is **flat**: min 2.05e-03, median 8.34e-03, max 1.57e-02, log-log
exponent **-0.071 at R^2 0.0195**, Spearman **-0.166**. Block 47 reads **1.32e-02**, lower than
block 25's 1.57e-02; blocks 0 and 23 read 1.10e-02 and 3.69e-03. There is no depth trend at all.

This settles two things. **Depth is not the mechanism** behind block 47 — consistent with §2,
where block 47 fails to respond to the lever that rescues blocks 0 and 23. And it **scopes §4**:
the masked crop-64 figure at block 47 (8.82e-03) matches this full-scope figure (1.32e-02) to
within a factor of 1.5, so the 8.26e-02 unmasked is a property of the crop-64 harness's padding
rather than of the model. The NaN-pad discriminator is still owed, because blocks 0, 23 and 47
were all measured on that harness.

Record: `depth_scaling.json`.

## 7. The precision story holds at blocks 0 and 23 and fails at block 47

If the residual were catastrophic cancellation on small-gradient layer norms, relative error
would concentrate on small-`ref_norm` tensors. Spearman(`ref_norm`, `rel_l2`) **within each arm**
(pooling across blocks is confounded by block-level offsets in `rel_l2`, so it is not the unit):

| block | rho range across its arms |
|---|---|
| 0 | -0.50 .. -0.29 |
| 23 | -0.33 .. -0.05 |
| 47 | **-0.11 .. +0.15** |

Blocks 0 and 23 carry the precision signature. **Block 47 does not** — its failure hits large-mass
and small-mass tensors alike, so "imprecise, not wrong" is not available there. Record:
`norm_vs_error.json`.

## 8. A mechanism with a shipped lever, and a registered prediction

The bundles carry the activation norms at each block's own boundary. The **single** track grows
through the stack and the **pair** track does not:

| block | `s_norm` | vs block 0 | `z_norm` | vs block 0 | gradient median |
|---|---|---|---|---|---|
| 0 | 5.8502e+03 | 1x | 1.3204e+06 | 1.000x | 0.0121 |
| 23 | 1.5190e+06 | **260x** | 1.3191e+06 | 0.999x | 0.0192 |
| 47 | 2.1493e+06 | **367x** | 1.2937e+06 | 0.980x | **0.9439** |

Mean per-block fractional growth of `s` is **27.3 %** over blocks 0-23 and **1.46 %** over blocks
23-47. bf16's relative resolution is `2^-8 = 3.9e-03`, so the per-block update is **70x** the
resolution early in the stack and **3.7x** late. That is the quantisation regime.

`tt_bio/tenstorrent.py` already implements the remedy and names it: `s_fp32_residual`, whose own
comment says it "is the difference between carrying an update and losing it: a track whose
residual is much larger than its per-block update quantises that update away, because bf16's
resolution is relative to what the accumulator already holds." It is set `True` in exactly one
place — `openfold3_confidence.py:101`, the **confidence** Pairformer. `openfold3_trunk.py:152`
does not pass it, so **the trunk runs at the default `False`**, and the trunk is the stack whose
single track grows 367x.

**Experiment, one flag:** re-run the existing crop-64 instrument-A arms at blocks 0, 23 and 47
with the trunk Pairformer constructed `s_fp32_residual=True`, everything else identical.

**Registered before the run:** block 47's median falls by more than 2x; block 0 moves by less than
20 %; and the pair track (`tri_mul_*`, `tri_att_*`) moves less than the single track
(`attn_pair_bias.*`, `single_transition.*`), because `z` does not grow through the stack and `s`
does.

**Caveats, stated because they matter.** `s_norm` is an L2 norm, not an absmax, so it cannot be
compared directly to the 2.28e5 absmax the flag's comment cites. The 27.3 % and 1.46 % are segment
means over 23 and 24 blocks, not local values. And **block 23 already sits at 70.7 % of block 47's
`s_norm` while its gradient median is 49x better**, so `s_norm` alone does not order the failures
— which is why block 23 is the control: if only 47 moves the mechanism is threshold-like, if both
move it is graded, if neither moves the hypothesis is refuted. Turning the flag on in the trunk is
**release-gated** (accuracy change, one extra `[B, L, c_s]` fp32 tensor): keep it on the branch,
flag it, do not merge it.

Record: `s_residual_hypothesis.json`.
