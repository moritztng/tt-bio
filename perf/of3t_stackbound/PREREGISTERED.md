# of3t-stackbound: pre-registration

Written and committed before any device arm has run on this boundary, and before the boundary's
own references have produced a single gradient figure. Nothing below may move after an arm
exists (A37). A miss is the finding.

## The question

`of3t-stackexact` cleared the clause at 0.9822570327981535x with exact softmax plus exact
LayerNorm (SL), on one captured 0.4.3 boundary (`boundary_model_n384.pt` 583bcd7c, 5nw3, 56 real
tokens padded to 384, one draw set). Is that a property of the trunk, or of a boundary that is
85 % padding?

## The second boundary

- Sample: **4hhb**, step 2 of the campaign's frozen batch stream (`build_batch.py`,
  `stage_config.yml`, seed 20260920, datapoint 1), **384 real tokens of 384**, 2257 atoms, 2 MSA
  rows. Built on pc with the 0.5.0 featuriser that built `batch_step003`. File sha256
  `667b7530e421eba525fbb630d925d182e4e5abd5c1d357cfaea058b9b03e54b3`, bit-identical to the
  campaign's own pc manifest entry for step 2. The same pc build of step 3 differs from the
  canonical `3c32597a` only in `ref_pos` (the RDKit conformer), so the featurisation is
  upstream's up to conformer generation.
- Draws: sampled fresh at `--seed 20260923` by the float64 run and recorded; every other
  reference and the capture replay them. r = 0, dropout off, deterministic kernels.
- References, all on qb1 CPU with `perf/of3t_reference/bundle_min.py` unmodified (D189: one host
  for every reference): float64 (contrast), bf16 autocast over fp32 parameters (graded), fp32
  upstream casts (instrument floor).
- Capture: `perf/of3t_modelframe/capture_model_frame.py` unmodified, `--expect-loss` set to the
  float64 run's own loss, `--ref-grads` its own gradient (the witness).
- Gate, before any arm is graded (A40): **COTANGENT_COMPLETE**. The float64 trunk replay through
  `perf/of3t_trunkg043/ref_grad.py` (graph-cut-external, the D242 repair, `--policy f64
  --crop 384 --checkpoint`) driven by the captured `(cot_s, cot_z)` must reproduce the float64
  reference's own `pairformer_stack` section: mass-weighted rel_l2 over all 2,736 trunk tensors
  **<= 1e-12**, and block 47 alone <= 1e-12 (the bar fixed in 2520681ed). If it fails, the frame
  is broken and that is the report; no arm is graded.
- A42: the correction `delta = d<cot_s,s_out>/d(z_out)` from that replay is pre-subtracted once
  into `cot_external_sb.pt`; every device arm is driven by that one file and cites its sha256.

## Arms

All on qb1 card 1 (p150a), `dev_cot.py --lever none` through `stackarm.py`,
`TT_BIO_SOFTMAX_BW_RENORM=1`, arm flipped, crop 0, 8 threads, the same harness as stackexact with
the board check widened to p150a and nothing else changed.

    tag      exact scopes            pf-set
    SHIP_A   none                    -
    SL       softmax, layer_norm     -
    SHIP_B   none                    -            (A/A, read before any arm is graded)
    SLZ      softmax, layer_norm     z_fp32_residual=1   (of3t-msafwd c647cf357, merged here)
    S        softmax                 -
    L        layer_norm              -

Order as listed. The A/A floor is SHIP_A vs SHIP_B, tensor by tensor, and is read before the
first clause number. It is expected to be exactly 0. If it is not, the resolution below becomes
`T = max(0.02, 3 x the floor's own closure)` and that is stated.

## Scoring on this boundary

`perf/of3t_stackbound/score.py`, which imports `perf/of3t_trajectory/agreement.py` (the scorer
every composed artifact used) and applies `model_scope.py`'s and `clause.py`'s arithmetic.

- Union: the same 3,660 tensor names as stackexact's composed artifacts (`UNION3660.json`).
- **Bar**, by the A26 rule, from THIS boundary's references only:
  `bar_B = sqrt(2) * floor / r`, floor and r the mass-weighted rel_l2 and norm ratio of this
  boundary's bf16 reference against its float64 reference over the union. No bar from the 5nw3
  boundary is used.
- **Trunk**: each arm's 2,736 `pairformer_stack` tensors against this boundary's bf16 reference
  (graded) and float64 reference (contrast), concatenated triple with its A43 residual.
- **The other five scopes.** Their device arms exist only on 5nw3; each has its own capture and
  instrument, and re-running them is outside this row. The clause here is therefore composed
  with them held at their banked graded readings (stackexact's `MODEL_SHIP_A` per-section
  `renorm_vs_UPSTREAM_BF16` rel_l2, the same values under every stackexact rung), weighted by
  THIS boundary's own per-section reference mass. That transported term is declared on every
  clause figure (A39). Beside it, the rest-free sensitivity `R0` (non-trunk sections exact) is
  reported, which is the loosest the allowance can be.
- **Allowance** by `clause.py`'s `solve()`: the trunk reading at which the composed clause equals
  `bar_B`, with the rest held.
- Control, before any arm is read: the composition function reproduces all four of stackexact's
  published clause values (SHIP_A, S, L, SL) from their own per-section tables at
  rel_difference <= 1e-15.

Quantities, as stackexact defined them: `c` = clause / bar_B, `e = c - 1`, closure
`f = 1 - e/e_SHIP`, `I = f(SL) - f(S) - f(L)`, `dZ = f(SLZ) - f(SL)`. Resolution `T = 0.02`.
If SHIP already clears here (`c_SHIP <= 1`), closure is undefined and every lever is read as the
margin it adds in bar units, `c_SHIP - c_arm`, with the same T.

## The disjunction (R186)

On SL, exactly one of:

- **CLEARS**: `c(SL) <= 1`.
- **IMPROVES ONLY**: `c(SL) > 1` and `f(SL) >= T`.
- **DOES NOT IMPROVE**: `f(SL) < T`.

(and **SHIP CLEARS**, `c(SHIP) <= 1`, if the boundary moves the baseline across the bar, read as
above.) SL failing here is a publishable result, not a reason to re-cut the boundary.

On SLZ: **ADDS** `dZ >= T`; **INERT** `|dZ| < T`; **COSTS** `dZ <= -T`.

On the interaction: **SUPER-ADDITIVE** `I >= T` (stackexact read +0.3418); **ADDITIVE**
`|I| < T`; **SUB-ADDITIVE** `I <= -T`. It "reproduces" if SUPER-ADDITIVE, and "reproduces in
size" if also within a factor of two of +0.3418.

Pad confound (stackexact's 4): if SL clears here with no pad rows, the clearing was not a pad-row
artifact. Boundary confound (5): the interaction band above.

## My prediction, before anything ran

- Baseline: SHIP does not clear here (80 %). A 384-real-token trunk carries more of the model's
  mass than 56 real tokens did, so the trunk's excess weighs more in the clause.
- SL: CLEARS 35 %, IMPROVES ONLY 55 %, DOES NOT IMPROVE 10 %. The float64 space closed faster
  than the graded one on 5nw3, which is what a genuine precision gain does, so I expect it to
  carry over; whether it clears depends on a 1.77 % margin that a new mass split can erase.
- Interaction: SUPER-ADDITIVE 55 %, ADDITIVE 35 %, SUB-ADDITIVE 10 %. The cancellation reading
  (softmax and LN errors partly offsetting) is a property of the ops, not of the pad rows, but the
  pad rows dominated the LN population on 5nw3, so its size may shrink.
- SLZ: ADDS 55 %, INERT 35 %, COSTS 10 %. The bf16 pair residual carried 0.88 of msa_module's
  forward gap and the trunk fires the same five adds 48 layers deep; the trunk's GRADIENT has not
  been measured under it.

Falsifiers of my own reading: SL DOES NOT IMPROVE, or the interaction turns SUB-ADDITIVE, says
stackexact's clearing belonged to its boundary.
