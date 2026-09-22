# of3t-vjpln — pre-registration

Committed before the first device arm of this row. Nothing below moves after a number appears.

## The falsifier, taken verbatim from `of3t-blk4544` and not restated in my own words

    R44 <= 1.33   names the carrier
    R44 >= 1.90   refutes it

`R44 = rel_l2(ds, rung 44) @ padded 384 / rel_l2(ds, rung 44) @ padded 64`, masked to the 56
real tokens, each leg scored against the in-frame float64 reference at ITS OWN width, both legs
from the SAME arm. Shipped-path baseline 2.201450; `of3t-blk4544`'s ceiling arm `allref`
2.203113.

## The arm

`allref2`: every `_v_matmul`, `_v_softmax`, `triangle_attention`, `_taped_layer_norm` and
`_taped_linear` in blocks 45 and 44 pinned to its float64 VJP. The two new VJPs are this row's
build; the other three are `of3t-blk4544`'s, unchanged.

Coverage, counted off a backward census and not argued from the selector source:
`allref` reached 1296 of 11856 backward firings (10.93 %). `_taped_linear` adds 1584 (13.36 %)
and `_taped_layer_norm` 816 (6.88 %), so `allref2` reaches **3696 of 11856, 31.17 %** — 2.85x
the previous ceiling. The residue stays 8160 firings, 68.83 %, and it is data movement
(`_v_permute`, `_v_reshape`, `_sliced`, `_identity_grad`, the head packers) plus `impl` and
`silu`.

## What I expect, and what each outcome means

`of3t-trunkact` concluded NO-GO after this row's brief was written and it moves the prior
substantially, so this is stated against its findings rather than against `of3t-blk4544`'s alone.

**I expect R44 >= 1.90 — refutation.** Three independent reasons, all measured by other rows:

  1. `of3t-bwdaccum`'s `SCOPE_LEVERS_c64.json` priced every device lever for the layer-norm
     weight reduction `dW = sum_t g_t xhat_t`: NONE 24.1356x its bf16 floor, PROD_FP32 24.1523x
     (inert), DX_FP32 32.3658x (worse). The reduction's arithmetic is not the carrier.
  2. `of3t-trunkact`'s census puts 56.1174 % of our trunk error mass in the 96
     `attn_pair_bias.layer_norm_a.{weight,bias}` tensors, and the SAME 96 hold 42.8309 % of
     upstream's own full-cast bf16 error mass. Per leaf the spread over 57 leaves is 1.032x to
     3.390x, median 1.571x: nothing beats its siblings by the margin one mis-wired op would show.
  3. `of3t-blk4544` measured the step as MASS INJECTION — `norm_ratio` 1.722041 at 384 against
     0.854295 at 64, while the reference's own masked `ds` norm is bit-identical to 16 digits
     between the widths — and per-op VJP substitution neither produces nor removes mass.

**R44 <= 1.33.** The layer-norm/linear backward's arithmetic IS the carrier, `of3t-trunkact`'s
regime reading is wrong, and this is the bigger of the two results. It would be the first arm in
the campaign to move `R44`, and I would say plainly that I predicted the other way.

**R44 >= 1.90.** It is not, and combined with `of3t-trunkact` the trunk's excess is a property
of what the backward READS rather than of how it reduces. The next object is then the forward
activations arriving at `attn_pair_bias.layer_norm_a`, named with the arithmetic — not another
ladder rung.

**1.33 < R44 < 1.90** is `PARTIAL` and names nothing on its own. It would be reported as amber
with the firing counts, not rounded toward either bar.

## Before the arm is read: the metric has to be able to move

`of3t-blk4544` reported an instrument defect in its own pre-registration twice, and the second
one is the one that matters here: `R44` reads `ds`, the SINGLE-track cotangent, and `ds` came
back bit-identical (exactly 0.0 at all 49 rungs) under both pair-track pins — so the metric
could not test two of that row's three candidates. A metric that cannot move is not a test.

So the demonstration is pre-registered as a gate, taken at padded 64 before the 384 leg is read:

    `allref2` at padded 64, blocks 45 and 44, against the shipped baseline `cot_B64.pt`:
    `ds` at rung 44 must differ by a NONZERO masked rel_l2.

If it is exactly 0.0, this arm has tested nothing about `ds` and the row reports that, with the
`dz` reading beside it, instead of a verdict on the falsifier.

## The VJPs, and how they are validated

Not against another device arm and not against a conventionally-written reference. The SDPA trap
is on record: the shipped attention adds the mask BEFORE the scale, and a reference written the
usual way agrees with a wrong gradient while both disagree with the forward. So each function is
read off `tt_bio/autograd.py` and reproduced term for term, then checked three ways:

  * **central finite differences** in float64 on the modelled forward — the check the brief
    requires, and the only one that cannot share an algebra error with the VJP;
  * **`torch.autograd.grad`** on the same forward, exact to roundoff, which separates "the VJP is
    wrong" from "h was badly chosen";
  * **the modelled forward against the device's own `out_v`** at every sampled live firing, which
    is the one question finite differences cannot answer — FD and autograd both differentiate the
    forward `refvjp.py` writes.

`layer_norm`'s reference is the backward's OWN arithmetic: `_taped_layer_norm`'s backward
recomputes its statistics from `x.value` with a two-pass `E[(x-mean)^2]`, so the float64
reference recomputes the same way. Pinning to a float64 recomputation of the ttnn FORWARD would
fold the forward's error into a reading that is supposed to be about the backward, which is the
choice `_ref_softmax` already makes and for the same reason.

`linear`'s fused activation is NOT part of this node: `_taped_linear` composes it as a separate
taped verb because `ttnn.linear` folds it into the packer and silu is not invertible. A VJP that
included an activation derivative here would be differentiating a function this node does not
compute.

Roles are resolved by object identity against the shipped frame's own locals, never by counting:
`parents` is `[t for t in (x, w, bias) if t is not None]`, so a call without a bias shifts every
later index, and `dw` returned where `dbias` is expected is wrong in a way no shape check catches
when both are rank-1. An unmatched parent raises.

## `impl` is censused before anything is written against it

`impl` is 2448 firings, 20.65 %, the largest single entry in the residue, and the brief is
explicit that a VJP written against a dispatcher tests nothing. `impl` is the closure name
`taped_ttnn._unary` and `_binary` give their wrappers, so the census is keyed on the SHIPPED
callable and the definition site, not on `co_name`. Whether it is in scope is decided from that
census and not before it.

## Floors, taken before any arm is read

A/A determinism at both widths (`of3t-blk4544`'s committed `cot_B64.pt` and `cot_B384.pt` are
the A side), the `census` arm's cotangent bit-identical to the shipped one, the `identity2`
control firing as often as `allref2` and coming back bit-exact, and wall clock against
`of3t-blk4544`'s 42.2 s / 680.5 s. The pin's own floor is unchanged and carried by every arm:
the exact VJP is written back in the parent's OWN dtype, so one float32 rounding per contribution
survives, counted as `float32_writeback_lossy`.

## What this row will not claim

Nothing here is a perf claim, nothing merges, no shipped default moves. This is one taped
backward of one captured boundary at two padded widths; it says nothing about stability over a
training run. Naming or failing to name the carrier at rung 44 does not narrow that bound by one
step.
