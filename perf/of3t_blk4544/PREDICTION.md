# of3t-blk4544 — pre-registration

Committed before the first arm of this row. Everything measured afterwards is scored against the
clauses below, and a clause that fails is reported as failed rather than repaired.

Host for every arm and every floor: **qb1 (tt-quietbox), card 2, p150a Blackhole**, AICLK sampled
every 4 s DURING each run. Every float64 reference and every upstream-bf16 floor is built on that
same box (D189). Every figure carries its padded width (D180, D193).

## The object

`of3t-tapeattn` measured, in-frame, masked to the 56 real tokens, upstream's own bf16 as the floor
per rung:

    rung 46   ds@384 / ds@64 = 1.042
    rung 45                  = 1.858
    rung 44                  = 2.201    <- D191's model-scope 2.1795x is already here
    rung 43                  = 2.275
    rung  0                  = 1.724

Rung k is the gradient of block k's INPUT, so rung k is what block k's backward produced. The step
is therefore created by the backwards of **blocks 45 and 44** and diluted over the remaining 44.

All 48 PairformerLayers are structurally identical. So whatever creates the step is an op that
runs in every block and only injects at 45 and 44, which makes the carrier **data-dependent, not
structure-dependent**. That constrains the candidates: it has to be an op whose error depends on
the magnitude or the conditioning of what arrives, not on a shape or a dispatch key. Both
shape-keyed dispatch sites are already pinned and dead (D200, +0.00670 % and exactly 0.0).

## The metric, fixed now

Primary, at padded 384, masked, against the in-frame float64 reference:

    R44 = rel_l2(ds, rung 44) @384  /  rel_l2(ds, rung 44) @64        baseline 2.201
    R45 = the same at rung 45                                         baseline 1.858
    B46 = the same at rung 46                                         baseline 1.042

`B46` is the pre-step level. An arm **names the carrier** if it drops `R44` to **<= 1.33**, i.e.
removes at least 75 % of the step `2.201 - 1.042`. An arm **refutes** its candidate if `R44`
stays **>= 1.90**, i.e. removes under 27 %. Between the two is a partial carrier and is reported
as a share, not as an answer.

Secondary, reported for every arm and never used to decide: `R44` on the `dz` track (baseline
1.952), `over_floor_masked` at rungs 47 down to 43, and the end-to-end `R0` (baseline 1.724).

## Three candidates, named in advance

**Candidate 1 — the pair track's TriangleAttention backward (primary).** `TriangleAttention`
runs twice per block (`_start`, `_end`) on `[N, 4, N, *]`, four heads, and its softmax reduces
over the full padded token axis. The single track's softmax is the largest injector at both
widths and `of3t-tapeattn` refuted it only as the thing that GROWS with width; the pair track's
softmax was never read, its reduction is 6x longer at 384 than at 64, and it feeds `dz`, which is
the track that steps hardest at rung 44 (1.952x).
*Falsified by:* pinning every `triangle_attention` backward in the stack to its float64 VJP and
reading `R44 >= 1.90`.

**Candidate 2 — TriangleMultiplication's backward.** `TriangleMultiplication` runs twice per
block and its product contracts over the padded token axis, so its backward carries the longest
reduction in the block: N terms at bf16 input precision, N = 384 against N = 64. Its gate is a
sigmoid, so its backward is also the one place in the pair track where a near-cancellation
multiplies a cotangent that has already grown.
*Falsified by:* pinning every TriangleMultiplication backward contraction to its float64 VJP and
reading `R44 >= 1.90`.

**Candidate 3 — the z to s bias path in AttentionPairBias.** The single track reads `z` as an
additive bias on its scores, so `dz` picks up a contribution that is a reduction of the single
track's score cotangent, and `ds` picks up the pair track's growth through the same node. This is
the only edge that couples the two tracks inside a block, and it is the only way a pair-track
defect can appear in `ds` at all, which it does one rung before `dz` steps.
*Falsified by:* pinning the bias node's backward to its float64 VJP and reading `R44 >= 1.90` on
BOTH tracks.

## The null candidate, screened before any device arm

The step may not be an op at all. At padded 384 there are 328 pad tokens against 8 at padded 64,
and every reduction over the token axis runs over them. If our pad rows carry mass that the
float64 reference's do not, a masked reading can step without any arithmetic getting worse.
*Screened by:* the pad-row norm of our own cotangent per rung, read off the cotangent files
`of3t-tapeattn` already wrote, at no device cost. If the pad-row share of the norm jumps at rungs
45 and 44 in step with `R44`, the carrier is mask handling and the three candidates above are all
wrong.

## What this row expects to happen

The primary prediction is **Candidate 1**: `R44 <= 1.33` with the pair track's
`triangle_attention` backward pinned, and `R44 >= 1.90` with the single track's softmax pinned.

The single-track softmax arm the brief asks for first is expected to be **inert on the step**:
`of3t-tapeattn` already showed its own arithmetic is 6.3 % better per firing at 384 with the
weights held, so removing it should move `R44` by less than the 27 % refutation band. If instead
it drops `R44` below 1.33, `of3t-tapeattn`'s decomposition is wrong and that is the finding.

## Controls that must read before any arm is believed

1. **A/A determinism**, two shipped-path runs at the same width, cotangent files compared tensor
   by tensor. `of3t-tapeattn` read exactly 0.0 at both widths; anything above 0 here invalidates
   every ratio in this row.
2. **Wall-clock floor** over that same pair, against `of3t-tapeattn`'s 1.92 % at padded 384, with
   the co-tenant load on qb1 named.
3. **Instrument inertness**: the pinning wrapper run in `identity` mode, the same round trip
   through host float64 and back, writing the device's OWN values in the parent's own dtype and
   layout, must reproduce the shipped cotangent file byte for byte. A control that cannot fire has
   tested nothing, so `identity` is checked to have fired the same number of times as the real pin.

## What this row will not do

No fix, no shipped default moves, nothing merges. A named op is a LOCATION, not a repair: the
campaign has named three mechanisms for this object and refuted all three, so Deliverable 3 is an
arithmetic statement about what removing the step would do to the 2.2341x at padded 384 against
the in-frame A26 bar of 0.5268825373, and nothing more.

If no pin moves `R44` out of the refutation band, the honest answer is that the ladder is
saturated (O(1) relative error from rung 47 down at both widths) and cannot localise further, and
this row names the arm that would instead of reading the ladder harder.
