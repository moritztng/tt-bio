# of3t-trunkg043 — pre-registration, written before any number exists

The trunk's gradient has never been read against the revision the checkpoint is bound to.
`of3t-pairformer` filed 1.061553e+01 over 2,496 of 2,736 tensors against a **0.5.0** reference at
a forward that disagreed on both tracks; A18 voids that reading. Two rows have since closed both
halves — `of3t-trunk043ref` rebuilt the reference on **0.4.3** (pair track passes as shipped at
4.947045e-02) and `of3t-trunkcliff` root-caused the single track to `scale_pair_bias=False`
(D1), a held convention whose flip reads 1.655263e-02. So an arm in which both tracks clear A18
exists, and this row takes its gradient.

## The boundary, fixed

`/home/ttuser/of3t_trunk043ref/boundary_c64.pt`, sha256
`85015b4a2622c1d3a6ae16f49c45cafd6f575dc0dab144fe08ffa87b83498ffa` — block 0's captured inputs
cropped to the first 64 token positions, 56 of them real (padding fraction single 0.125, pair
0.234375). The cotangent is the one the capture recorded at block 47's output, cropped the same
way, from `/home/ttuser/of3t_gradients/cap/block47_boundary.pt` sha256
`a55ef1c4e6c90c87984242f9d1cc8523fbd9470a5f592eac0d0aff7000d29cf5`. The stack's parameters appear
once in their graph and at `num_recycles` 0 the trunk runs once, so
`dL/dtheta = d/dtheta [<cot_s, s_out> + <cot_z, z_out>]` is exact at this boundary.

## The arms, fixed

| arm | what | role |
|---|---|---|
| FLIPPED | `scale_pair_bias=True, tri_att_scale_pair_bias=False` | the forward that passes A18; **this scope's reading** |
| SHIPPED | the configuration read off `OF3Trunk`'s own construction by spying on the `Pairformer` constructor | the held product decision's gradient cost, reported, not the reading |
| REF-F64 | upstream 0.4.3, every parameter and activation float64, checkpoint upcast once at load | the reference every `rel_l2` is against |
| REF-BF16 | upstream 0.4.3, fp32 parameters under `torch.autocast('cpu', bfloat16)` | upstream's own floor (A27: the denominator names how it was built) |
| ZERO | a gradient of exact zeros scored against REF-F64 | A16 — must read exactly 1.0 |
| BREAK | the captured cotangent permuted over the 56 real token positions, everything else untouched | must move the reading |

`of3t-trunkcliff` found the token permutation near-inert on the single track (99.00 % of its mass
in 3 of 384 channels), so the **pair track is the decisive break arm** and the break is judged on
it.

## The statistic, fixed

A23: mass-weighted, never a count and never a median-over-tensors. The weight of tensor `t` is
its share of REF-F64's squared gradient norm over the compared set:

    rel_mw  =  sqrt( sum_t w_t * rel_t^2 ),   w_t = ||g_ref_t||^2 / sum_u ||g_ref_u||^2

Per tensor, three numbers and not one (D35): `rel_l2`, `norm_ratio = ||ours||/||ref||` and
`cos`. A bare `rel_l2` cannot tell 18x too small from 2x too big. The worst tensor is named as a
full parameter path, and the compared set's share of the stack's squared gradient norm is
reported beside the count (A15/A20 — an unplaced tensor is UNREACHED, not absent, and carries its
reference norm out with it).

Both references are reported for every arm and each names which it is against (A27):
`ours_vs_float64` and `ours_vs_their_bf16`.

## The branches, fixed before the numbers

- FLIPPED `rel_mw` **inside 2.0e-02** → the trunk's 5.8282 % of the model is REPRODUCED.
- FLIPPED outside 2.0e-02 but within A26's sqrt(2) of it (**<= 2.828427e-02**) → reachable;
  report the gap.
- FLIPPED **worse than 2.5x** REF-BF16's own `rel_mw` → there is a backward defect the forward
  does not show. That is D9's shape (a 3.2x gradient change under a 12 % forward change) and it
  is the interesting outcome, to be said plainly rather than reported as a miss.

These three are not exclusive: a reading can be inside the bar and still above 2.5x the floor,
and if it is, both are stated.

## What this row will not do

No shipped default moves. `compose_verify.sh` asserts the trunk line reads
`scale_pair_bias=False, tri_att_scale_pair_bias=False` and must still pass against this branch;
the flipped arm is a measurement configuration, not a proposal. The Angstrom decision is
`of3t-confhead`'s and it concluded. `s_fp32_residual` is refuted twice and is not re-derived, and
`fp32_softmax`/`accurate_softmax` are structurally dead at the token-level branch of
`AttentionPairBias.__call__`, which never calls `self._attention`.
