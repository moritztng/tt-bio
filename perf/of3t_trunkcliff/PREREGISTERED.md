# of3t-trunkcliff — pre-registration

Written and committed before this row's first number exists. Every branch is named here so
none of them can be picked after the result is known.

## What is inherited, not re-derived

`of3t-trunk043ref` settled the reference revision. Against 0.4.3 the pair track passes at
4.947045e-02 (crop 64, masked) and the single track reads 1.065338e-01 with norm ratio
0.907813 and cosine 0.998430: 74.88 % of the squared error is magnitude, 25.12 % direction.
The depth ladder puts 6.22x of that in the last three of 48 blocks. `transpose_bias` is
closed. None of this is re-opened.

## The question

Where inside those blocks is the 9.22 % of norm lost, and is it ours or is it what bf16
costs in this regime?

## The instrument

One block at a time, block index 45 (the k=46 rung, the sharpest single step), with 44 and 46
beside it. Four arms per block, all fed the SAME state, so the block's own arithmetic is
separated from error carried into it:

  REF64      upstream 0.4.3, float64 parameters and activations, fed the f64 stack state
  REF64Q     upstream 0.4.3, float64 arithmetic, fed the state ROUNDED TO BF16 first.
             The cost of representing the input at all, with no bf16 arithmetic anywhere.
  REFBF16    upstream 0.4.3 under torch.autocast('cpu', bfloat16), fp32 parameters, fed the
             bf16-rounded state. Upstream's own training recipe: the floor (A27: that is the
             dtype policy, "bf16" would only be a width).
  OURS       tt-bio's shipped Pairformer, one block, on device, fed the bf16-rounded state.

Ten tensors per arm, the decomposition both sides genuinely share:

  z1..z5   the pair track after each of tri_mul_out, tri_mul_in, tri_att_start, tri_att_end,
           pair_transition, residual included
  sn       LayerNorm of the single track (ours: pre_norm_s; upstream: attn_pair_bias.layer_norm_a)
  u1       the AttentionPairBias update
  s1       s + u1
  u2       the single transition update
  s2       s1 + u2

Norm ratio and error cosine beside every relative L2, and MEDIANS PER LEAF OP, not the worst
tensor.

## Branches, named before the run

**M1 — made in the block.** Fed identical input, OURS's u1 or u2 is short in norm by more than
REFBF16 is against REF64. The deficit is our arithmetic, at a named op, and the op is named.

**M2 — carried in, not made.** Fed identical input, OURS agrees with REF64 to within REFBF16's
own distance from REF64. Then the cliff is amplification of the 1.7e-02 already present at
k=45, and the thing to name is the amplification, not an op.

**M3 — the pair track drives it.** The deficit follows z, not s: feeding ref-z with our-s (and
the reverse) moves the single-track update deficit.

**M4 — input representation alone.** REF64Q already shows most of the deficit, so bf16 cannot
hold this state and no arithmetic change reaches it.

**F1 — a shippable change recovers the magnitude.** Both tracks re-measured, masked, padding
fraction beside each; release-gated, stays on the branch.

**F2 — this is what bf16 costs in this regime.** Named here in advance as a legitimate
outcome, not a fallback: upstream's own bf16 arm moves 3.6x over the same stretch. F2 is the
verdict if OURS's per-block excess over REFBF16 stays within the A26 reachable factor sqrt(2)
and no lever recovers the norm without spending it somewhere else.

## Controls, all measured before any headline

1. **Instrument identity.** The single-block harness fed the f64 stack state at k=45 must
   reproduce the 48-block stack's k=46 output BIT FOR BIT on the REF64 arm. If it does not,
   the harness is a different function from the thing being explained and every number below
   it is void.
2. **Step identity on device.** The step-by-step device decomposition must reproduce the
   shipped `PairformerLayer.__call__` output bit for bit. Otherwise the decomposition is a
   reimplementation and not the shipped path.
3. **Break control.** The 56 real token positions permuted in the fed state, weights, masks,
   flags and kernels untouched. Every leaf reading must move. A leaf that does not move is
   saturated and carries no information about that leaf.
4. **Floor re-run in this row's process.** REFBF16 is not carried in from `of3t-trunk043ref`;
   it is re-run here.

## Discipline

- A26: the bar an independent bf16 port can reach is sqrt(2) x threshold = 7.0711e-02.
- A27: every arm used as a denominator names its dtype policy, not a width.
- D99: both tracks side by side. D95: the padding fraction beside every figure.
- A precision change that could move accuracy, OOM or perf for another model is
  release-gated: it stays on `wk/of3t-trunkcliff` and is flagged, never merged here.
