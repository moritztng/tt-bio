# of3t-readverbs — pre-registration

Committed before the first device arm of this row. Nothing below moves after a number exists.

## What this row substitutes

`_identity_grad` (26.50 % of the taped backward at padded 384) and `_sliced` (21.47 %), the two
verb classes `of3t-blk4544` excluded on purpose when it picked *"the two largest unreachable
verbs that are real arithmetic rather than data movement"*. Together 28,848 of 60,144 firings,
47.97 %, and nothing has ever been substituted in either.

At blocks **44, 4 and 0**, which `of3t-trunkblocks` measured as 76.4502 % of the trunk's absolute
error mass (30.6209 %, 29.1573 %, 16.6720 %) against block 45's 0.5981 %. Every prior arm in this
sub-campaign ran in blocks 45 and 44, i.e. 31.2190 % of the mass.

## What I expect, and why

**Refutation on the arithmetic, and a split answer between the two verbs.** Stated before the
arm so the census can contradict it:

1. **`_sliced` has no arithmetic to be wrong.** Its backward is `ttnn.zeros(dtype=g.dtype)` on
   each cut axis and `ttnn.concat`, both pure data movement, plus one `to_layout` to TILE when
   the cotangent arrives row-major. In exact arithmetic the only thing it can get wrong is WHERE
   the gradient lands — the `starts`/`ends` index arithmetic and, for `chunk`, the accumulation
   of several adjacent slices into one parent. That is a correctness question with a binary
   answer, not an error source with a magnitude, and I expect it to come back exact.
2. **`_identity_grad` has exactly one lossy step and it is a dtype cast, not an operation.**
   `bw` is `x.add_grad(ttnn.typecast(g, src_dtype) if cast and g.dtype != src_dtype else g)`
   (`tt_bio/taped_ttnn.py:110`). The `cast` branch is reached only by `typecast`; `clone`,
   `reallocate`, `to_layout` and `to_memory_config` register with `cast=False` and hand `g` to
   `add_grad` untouched. So the class's whole error budget is the firings where `cast` is true
   AND `src_dtype` is narrower than `g.dtype` — an fp32 cotangent rounded to bf16 on its way
   into a bf16 parent.
3. **That cast is not harmless even though the parent is bf16.** `Tensor.add_grad` stores the
   FIRST contribution exactly as it arrives and only promotes to fp32 on the second
   (`tt_bio/autograd.py:345`). A cotangent rounded to bf16 before the first contribution lands is
   rounded before the fp32 accumulator exists, which is the one place `of3t-vjpln`'s
   "`add_grad` promotes on the second contribution, so a bf16 running sum is not the mechanism"
   does not reach.

So the census must report, per class per block, how many firings take a NARROWING cast, and the
arm must be read against that count. If the narrowing count is zero in blocks 44, 4 and 0, the
class is inert by construction and the row says so with the census rather than spending a 384
arm to discover it.

## The instrument's own ceiling, pre-registered

`pinvjp._run` writes the exact contribution back with `dtype=tgt.dtype`, the parent grad's own
dtype. For a first contribution that arrived bf16 that re-rounds the float64 reference to bf16
and reproduces the shipped value, so **the comparable-protocol arm is expected to be nearly inert
on `_identity_grad` by construction**. That is a property of the write-back, not a finding about
the verb, and it is why this row runs TWO arms:

    route      the float64 VJP written back in the parent's OWN dtype. Same protocol as
               `allref`, `allref2` and `all`, so the readings stay comparable.
    routef32   the same VJP written back in float32 when the parent's grad is narrower. This is
               the one that actually removes the rounding, and it is the arm the COUNTERFACTUAL
               is read from.
    identity3  every node either selects, round-tripped through host float64 and written back
               UNCHANGED. Must be bit-exact. The inertness control (D121).

`routef32` changes a storage dtype and is therefore not a pure VJP substitution. It is reported
as what it is.

## The falsifiers

### Leg A — block 44. `R44`, inherited verbatim

    R44 <= 1.33   names the carrier
    R44 >= 1.90   refutes it

`R44 = ds@384 / ds@64` at rung 44, masked to the 56 real tokens, each leg scored against the
in-frame float64 reference AT ITS OWN WIDTH. Inherited from `perf/of3t_blk4544/PREDICTION.md`
(`d6cbfdc67`) and re-registered by `perf/of3t_vjpln/PREDICTION.md` (`38a94d7e8`), word for word
and threshold for threshold, so this row's reading sits in the same table as `base` 2.201450,
`all` 2.202460, `allref` 2.203113 and `allref2` 2.206861.

### Leg B — blocks 4 and 0. `R44` IS STRUCTURALLY BLIND THERE, so a second metric is required

This is the part the brief did not have, and it is registered here rather than discovered later.
Rung k is the gradient of block k's INPUT, and the backward runs 47 -> 0, so blocks 4 and 0 fire
*after* rung 44 has already been written. **No substitution at block 4 or block 0 can move `R44`
by any amount.** An arm at those blocks read only through `R44` would return a guaranteed
refutation that tested nothing, which is D121's shape with the sign flipped.

So leg B is read on the leaf gradients, in the frame the GRADIENTS clause is evaluated in:
`perf/of3t_frame384/frame384.py` at crop 384, `MATCHED.ours_vs_REF_LOCAL_bf16_n384`, ours
against upstream's OWN bf16 with both sides built on qb1 from the same boundary.

    T_base = 1.0293953377723410     the trunk section in frame, shipped path
    A26    = 0.5268825372815341     the in-frame A26 bar for this scope
    gap    = 0.5025128004908069     T_base - A26, what an arm has to remove

    F = (T_base - T_arm) / gap      the fraction of the gap the class closes

    F >= 1.00    names the carrier -- the class closes the clause at this scope
    F <= 0.05    refutes it
    0.05 < F < 1.00  amber, and the number is quoted with the class's share of firings beside it

`T_base` is re-measured by THIS row's own harness before `F` is read from anything; the
1.0293953377723410 above is `of3t-frame384`'s banked reading on
`dev_RENORM_n384_nocaptures.pt` and is used as the pre-registered constant, not as this row's
baseline. If my own baseline disagrees with it, the disagreement is the finding and `F` is
computed against MY baseline, with both printed.

## The gates that must pass before any reading is quoted

1. **A/A determinism floor at padded 384 and padded 64**, taken before any arm is read, against
   `of3t-blk4544`'s committed `cot_B384.pt` / `cot_B64.pt`. Bit-identical or the row stops.
2. **The control FIRED.** `identity3` must select and apply the same node count `route` does,
   with `float32_writeback_lossy` 0 on the identity side. A control that cannot fire has tested
   nothing.
3. **The arm FIRED, measured.** Reach is `substitutions_total / backward_firings_total` from the
   SAME run's own counters, never from a selector's name. `of3t-blk4544` paid for that: a
   selector called `all` reached 6.88 %.
4. **The float64 VJPs are validated against float64 CENTRAL FINITE DIFFERENCES** before any
   device arm, including the composed `chunk` case where several adjacent slices accumulate into
   one parent, and against `torch.autograd.grad` on the same forward. FD and autograd both
   differentiate the forward the reference writes, so the modelled forward is separately checked
   against the device's own `out_v`.
5. **AICLK sampled DURING** every device run with its sample count, host and card in the
   artifact (D155).

## What each outcome means

* `F >= 1.00` — the read/route class is the carrier and the clause closes at this scope. The
  next row prices an fp32 cotangent path on device.
* `0.05 < F < 1.00` — a real but partial share. The number is the class's contribution and the
  residue is what the next arm has to find.
* `F <= 0.05` with a NONZERO narrowing-cast count — the class is reachable, live, measured and
  not the carrier. Combined with `of3t-trunkact`, `of3t-bwdaccum`, `of3t-blk4544` and
  `of3t-vjpln`, that leaves `impl` (10.93 %) and `_v_permute` (7.90 %) as the only unsearched
  verb shares, and the excess is then not located in any single verb class at all.
* narrowing-cast count ZERO in blocks 44, 4 and 0 — the class has nothing to make exact there,
  the census says so, and the row moves to the next class by share without spending the arm.
