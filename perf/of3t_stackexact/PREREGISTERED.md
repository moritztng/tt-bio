# of3t-stackexact: pre-registration

Written and committed before any ladder arm has run. The only device work before this commit
is `lncheck.py`, the exact layer norm's float64 unit check (`LNCHECK.json`), which scores the
lever's own arithmetic and reads no gradient of the trunk.

## The ladder

Every rung is the model-frame trunk arm of `of3t-modelever` (`dev_cot.py --lever none`,
`TT_BIO_SOFTMAX_BW_RENORM=1`, arm flipped, crop 0, 8 threads, boundary `boundary_model_n384.pt`
583bcd7c, correction `cot_external.pt` 1d15a8dc), run through `stackarm.py`, so two rungs
differ only in which exactness scopes are open. Boundary version, for every gradient figure:
**upstream OpenFold3 0.4.3** (the boundary is captured from `of3pkg043`; the graded reference is
0.4.3 bf16 autocast `ff78d7bc`; the contrast reference is 0.4.3 float64 `1d4ea922`).

    rung  tag       scopes                     role
    1     SHIP_A    none                       A/A: must be bit-identical to of3t-recut's banked
                                               external arm (c0a1b121) and to modelever's SHIP_A
    2     S         softmax                    must reproduce 1.3037867474869442x exactly, and
                                               be bit-identical to modelever's dev_EXACT.pt
    -     L         layer_norm                 decomposition arm: the second lever alone
    3     SL        softmax + layer_norm       the second rung
    -     SHIP_B    none                       second A/A, interleaved after L
    4     (optional, only if the card allows after 1-3 are scored; same bands, same frame)

Run order: SHIP_A, S, L, SHIP_B, SL. If rung 1 is not bit-identical, or rung 2 does not
reproduce 1.3037867474869442x, the ladder stops there and that is the report.

## Quantities

`c` = clause (model `mass_weighted_rel_l2` vs upstream 0.4.3 bf16) over the 0.15210099830945006
bar, read through `perf/of3t_modelframe/clause.py`, levels read from `CLAUSE.json`, never
re-derived. `e = c - 1` is the excess, `e_ship = 0.4511706984958472`. A rung's closure is
`f = 1 - e/e_ship`, so `f(S) = 0.32667` if rung 2 reproduces.

    f(L)          the second lever alone
    dL = f(SL) - f(S)   what the second lever adds on top of the first
    I  = f(SL) - f(S) - f(L)   interaction; 0 is exact additivity
    theta_g, theta_f    the trunk's concatenated angle vs upstream bf16 (graded) and vs float64
                        (contrast), identity residual beside each (A43)

Resolution `T = 0.02` (two points of the excess). The A/A floor is expected to be exactly 0;
if it is not, `T` becomes max(0.02, 3 x the floor's own closure) and that is stated.

## Bands (two axes, then the verdict)

Axis A, the clause (graded space):

- **A-COMPOSES**: `f(L) >= T` and `dL >= T` and `dL >= 0.5 f(L)`. The second lever keeps at least
  half of what it gives alone when stacked on the first.
- **A-CANCELS**: `dL < -T` (the curve turns over: rung 3 is worse than rung 2), or `f(L) >= T` and
  `dL < 0.5 f(L)` (the lever gives materially less on top of the first than alone).
- **A-INERT**: `|f(L)| < T` and `|dL| < T`. Exactness at this component does not reach the clause.
- anything else (e.g. `f(L) <= -T`) is reported as observed, with the numbers, and not forced
  into a band.

Axis B, float64 direction at rung 3 relative to rung 2 (`theta_f(SL) - theta_f(S)`):
**B-TOWARD** below -0.5 deg, **B-AWAY** above +0.5 deg, **B-FLAT** between.

Verdict:

- **COMPOSES** = A-COMPOSES and not B-AWAY.
- **CANCELS** = A-CANCELS, or A-COMPOSES with B-AWAY (graded improves while float64 degrades
  further, the second sighting of rung 2's signature).
- **INERT** = A-INERT. This is not COMPOSES; the stacking plan gains nothing from this component.
- If axis A and axis B point different ways outside these cases, it is a **SPLIT**, reported with
  `dL / f(L)` as the fraction.

## What else could produce a turnover (R186)

"Not COMPOSES, therefore error cancellation" would be the R186 mistake. A turnover or a
sub-additive rung can come from at least:

1. **cancellation between our own components** (the hypothesis: exact softmax removed an error
   that had been offsetting a bf16 error elsewhere in our trunk);
2. **the graded reference's own error**. 0.4.3 runs LayerNorm with bf16 weights inside
   `autocast(enabled=False)` and the attention softmax in bf16 (`of3t-fp32islands`). An exact
   component can move us toward float64 and away from the reference's own rounding, which the
   graded clause scores as worse. Axis B separates (1) from (2): turnover with B-TOWARD points at
   (2), turnover with B-AWAY at (1);
3. **a defect in the lever**. Guarded before the ladder: `LNCHECK.json`, float64 autograd, worst
   exact rel 1.74e-3 on bf16 inputs (the bf16 storage floor) and 2.5e-8 on fp32, with the shipped
   op's own error beside it. The census (`verb`, `bw`, `raw`, and dev_cot's own LN counter
   reading 0 backward calls when this scope holds the verb) guards "fired but displaced";
4. **the frame**: 56 real tokens padded to 384, so pad rows dominate the pair LN population. An
   exact LN on pad rows may move mass that a real crop would not carry. This row cannot separate
   it and says so;
5. anything else is left open. A CANCELS verdict states which of (1)/(2) axis B selects and
   claims nothing further about mechanism.

## Predictions (mine, before measuring)

- Rung 1 bit-identical, rung 2 reproduces to all digits (the arms are deterministic: modelever's
  A/A floor was exactly 0).
- `f(L)` in [-0.05, +0.10]. The shipped bf16 LN is 2.4x the bf16 storage floor in dx
  (`LNCHECK.json`), so there is less error to remove than at the softmax.
- Most likely verdict INERT or CANCELS (together ~65 %), COMPOSES ~25 %, SPLIT ~10 %.
- Falsifier of my prediction: `f(L) >= 0.10` or `dL >= 0.10`.

These are scored afterwards in the state doc, including the ones that miss.
