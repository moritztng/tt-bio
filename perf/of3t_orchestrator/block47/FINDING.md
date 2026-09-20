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
