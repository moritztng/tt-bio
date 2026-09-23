# of3t-residual — the 2.04x was mostly the instrument

## What this row was sent to do

`of3t-trajectory` bounded every softmax in float64 and took the device gradient from
7.426742e+00 to 1.172914e-01 against upstream's own bf16 training step, over the 547 tensors
that hold 51.1358 % of the model's squared gradient norm. That is 63.3x of the gap, and it
leaves 2.04x the threshold a perfect port would read. This row was sent after the 2.04x, with
two pointers: `atom_attn_dec` did not move at all under the bound (5.65x its own floor), and
the worst tensor moved house to a `conditioned_transition` layer norm.

## What it is

Both pointers were the same defect in the comparison harness, not in the port.

`perf/of3t_diffusion/device_gradient.py` put the device gradient back into checkpoint
orientation with a shape test: if the shape is not the wanted one but its reverse is,
transpose. `_w_tt` stores every weight on the card as `w.t()`, so the taped dW rule returns
`dW^T` — and that test cannot fire on a square weight. **87 of the 547 compared tensors are
square, and every one of them was scored as `dW` against `dW^T`.**

A transpose is norm-preserving and, on a matrix with no particular symmetry, decorrelating.
So the signature is exact and it is the same in every arm:

| arm | n | mass | median rel | median cos | median r |
|---|---|---|---|---|---|
| shipped, square | 87 | 0.2035 % | 1.4137e+00 | +0.0023 | 1.0029 |
| shipped, everything else | 460 | 50.9323 % | 1.4062e-01 | +0.9916 | 1.0138 |
| float64-softmax bound, square | 87 | 0.2035 % | 1.4001e+00 | +0.0023 | 0.9838 |
| upstream's own bf16, square | 87 | 0.2035 % | 7.1522e-02 | +0.9976 | 1.0135 |

1.4137 is sqrt(2) to four digits. The same 87 tensors read 7.15e-02 with cos 0.9976 when
upstream's own bf16 step is scored against float64, so they are not intrinsically hard. Under
the bound they were **56.22 % of the arm's squared error** while carrying 0.2035 % of its mass.

The fingerprint that maps a device tensor back to its checkpoint name is deliberately
transpose-invariant (`tuple(sorted(x.shape))`), so it could not catch this either.

`ORIENTATION_IS_THE_RESIDUAL.json` asks the data rather than the code: score both orientations
of every compared tensor against the pinned arm4. **All 87 square tensors score better
transposed, and no non-square one does.** Transposing only the winners removes 56.03 % of the
bound arm's squared error. `mha.linear_o.weight` in DiT block 5 goes 1.3887e+00 to 8.9705e-02,
cos 0.0050 to 0.9964.

## The fix

The orientation is recorded at load, not inferred at comparison time. The tensor handed to
`ttnn.from_torch` is compared bit-exactly to the checkpoint tensor in both orientations:
336 of 547 come back `T`, 211 `N`, none unknown. A weight whose load is not a plain transpose
(a reshape, a pad, a fuse) records `?` and falls back to the old shape test, which is right
for it. The comparison then asserts the final shape, so a wrong answer throws instead of
scoring.

## What it does to the numbers

    arm                          v their step   v float64      r        cos      x threshold
    shipped                      7.426217e+00   7.568637e+00   7.7814   0.4114   129.13
    float64-softmax bound        7.777580e-02   5.930664e-02   0.9787   0.9971     1.35
    the same bound before D83    1.172914e-01   1.066810e-01   0.9787   0.9932     2.04

The shipped headline barely moves (7.426742e+00 to 7.426217e+00) because the square tensors
carry 0.2035 % of the mass and the shipped arm's error is dominated by tensors that are
genuinely wrong. It is the *bound* arm the defect was hiding, which is exactly the arm the
campaign now cares about.

Per section, against each section's own `floor / r`:

| section | mass | bound v their step | threshold | x | bound v float64 | their floor |
|---|---|---|---|---|---|---|
| diffusion_transformer | 43.6221 % | 7.8965e-02 | 5.6702e-02 | 1.39 | 6.2927e-02 | 5.7809e-02 |
| atom_attn_enc | 4.7348 % | 7.9831e-02 | 7.1373e-02 | 1.12 | 3.6248e-02 | 7.0855e-02 |
| atom_attn_dec | 1.2843 % | 4.8851e-02 | 4.8564e-02 | 1.01 | 1.1684e-02 | 5.0000e-02 |
| layer_norm_s | 0.9835 % | 4.8164e-02 | 3.0080e-02 | 1.60 | 1.8751e-02 | 3.1012e-02 |
| layer_norm_a | 0.2666 % | 3.7430e-02 | 3.5952e-02 | 1.04 | 4.9312e-03 | 3.6970e-02 |
| linear_s | 0.2445 % | 8.3418e-02 | 6.4550e-02 | 1.29 | 3.9285e-02 | 6.5442e-02 |

`atom_attn_dec` was 5.65x and is 1.01x. **On every section except the diffusion transformer,
the bounded device gradient is closer to float64 than upstream's own bf16 training step is.**

## What the 1.35x that is left actually is

A25's error cosine answers it. Over the scope:

    ours from float64      5.930664e-02
    theirs from float64    5.852018e-02        ratio 1.0134
    cos(ours - f64, theirs - f64)   +0.0977

Two bf16 error vectors of the same size pointing in nearly independent directions. The
distance between them follows:

    sqrt(1.0134^2 + 1 - 2*1.0134*0.0977) * 5.750945e-02 = 7.7776e-02

against the measured 7.777580e-02. **The 1.35x is arithmetic, not a mechanism.** If our error
were an independent bf16 error of exactly upstream's own size, the reading would be
sqrt(2) * 5.750945e-02 = 8.1331e-02; we read 7.7776e-02, which is 0.956x that.

This is also the honest correction to the pre-registered threshold, in the spirit of D76. The
threshold 5.750945e-02 is what a device gradient reads if it **equals float64**. A bf16 port
cannot equal float64, and the value a bf16 port of upstream's own quality reads is 8.1331e-02,
not 5.750945e-02. Both numbers are in the attainable range; they answer different questions.

## The transition bound

`host_f64.py` generalises of3t-adaln's `host_f64_rule` to the other verbs, with the backward
taken from `torch.autograd.grad` on the float64 forward rather than hand-derived — six new
hand-written backward rules between the measurement and the answer would be worse than no
bound. Installed on `layer_norm,multiply,multiply_,add,add_` on top of the softmax bound, it
bounds everything the conditioned transition does except its two matmuls, and, being on the
verb, it bounds the same ops everywhere else too. That makes it an **upper bound** on what the
transition path can contribute.

Per-verb intercept counts are published and a zero on any installed verb is a hard failure;
the verb census (`--verb-census`, one structure, no arithmetic changed) prices a bound before
it is paid: per structure, layer_norm 163, linear 137, multiply 197, multiply_ 66, add 105,
add_ 90, matmul 61, softmax 30.

Over the 48 structures it intercepted layer_norm 7,824, multiply 9,456, multiply_ 3,168,
add 5,040 and add_ 4,320, 29,808 in all, beside the softmax bound's 1,440. And the answer is
the sharpest thing in this row:

    arm                        v their step   v float64      x threshold   err-cos
    float64 softmax                7.777580e-02   5.930664e-02        1.3524   +0.0977
    + the transition bound         7.733138e-02   3.815464e-02        1.3447   -0.2938

**The transition bound buys 0.57 %.** It removes 35 % of our own error — 5.930664e-02 to
3.815464e-02 from float64, with the per-tensor median going 4.885781e-02 to 3.213349e-02, the
worst 2.618132e-01 to 1.557270e-01 and the count over the 5.0e-02 bar 295 of 547 to 43 of 547
— and the agreement with upstream's training step moves by half a percent. Our bounded
gradient is then 1.53x closer to float64 than upstream's own bf16 step is, and it still reads
1.34x its threshold, because the distance that is left is THEIR bf16 error and no amount of
precision on our side can remove it. Agreeing with an imprecise reference is a stricter test
than being accurate, and here the two come apart by a factor of 1.55.

The arm costs 13 s per structure against 2.5 s. It is a bound, not a lever.

## Where the error lives now, by leaf

`LEAF_ATTRIBUTION_AFTER_D83.json` groups every tensor by leaf op and reports the squared error
over and above upstream's own bf16 squared error at the same tensors. After D83, 90.6 % of
that excess is in two leaves, and both are AdaLN s-norm gains:

    51.6 %  blocks.N.attention_pair_bias.layer_norm_a.layer_norm_s.weight     25.5795 % mass
    38.9 %  blocks.N.conditioned_transition.layer_norm.layer_norm_s.weight    15.5125 % mass

Excess share now tracks mass share (1.33 against 1.65), which is what bf16 rounding spread
over the model looks like. Before D83 the top leaf was `mha.linear_o.weight` at 37.3 % of the
excess on 0.1019 % of the mass, which is what a defect looks like.

## Re-running

    perf/of3t_residual/devgrad_res.sh shipped      # ~2.5 min on one Blackhole card
    perf/of3t_residual/devgrad_res.sh sm64
    perf/of3t_residual/devgrad_res.sh permcot
    perf/of3t_residual/devgrad_res.sh sm64permcot  # the break control on the bound arm
    perf/of3t_residual/devgrad_res.sh census       # one structure, verb counts only
    perf/of3t_residual/devgrad_res.sh tr64         # the transition bound, ~11 min

Then, with no card:

    perf/of3t_residual/a25.py --f64 ... --bf16 ... --bf16-sha ff78d7bc... --arms label=...pt
    perf/of3t_residual/orientation.py --device ...pt --ref ...pt --ref-sha ff78d7bc...
    perf/of3t_residual/square_check.py --arms label=sidecar.json ...
    perf/of3t_residual/leaf_attrib.py --arm sidecar.json --floor sidecar.json

and `perf/of3t_trajectory/agreement.py` unchanged, which produced `AGREEMENT_AFTER_D83.json`.

## Controls

**A16**, the measured zero model, reads 1.000000e+00 against their step on every arm.

**Scale**: upstream's own arm4-vs-arm2 spread on this scope is 5.852284e-02, and their
permuted-draws control reads 4.018294e-01.

**The break control now works on the arm that nearly passes.** On the shipped arm the
permuted-cotangent control is saturated: it *lowers* the headline, 7.4262e+00 to 3.7059e+00,
because the port's own error already exceeds the perturbation. On the float64-softmax bound it
behaves properly in both statistics:

    bound                7.777580e-02   cos +0.997141
    bound + permuted     1.554075e+00   cos -0.014252     20.0x, and the cosine collapses

That is the arm the verdict rests on, and it is the arm on which the control discriminates.

## Clock

No timing figure is published: a gradient comparison is arithmetic, not throughput. The AICLK
is recorded anyway. Card 0, sampled every 2 s from `qbcard/cardtel.tsv` DURING the arms,
n=120, mean 1304 MHz, min 800, max 1350.

## What this does not close

Neither the port nor the harness is proven beyond this scope. 48.8642 % of the model has no
direct measurement here, `diffusion_conditioning` at 36.9462 % of it being the largest. And
**D83 is not confined to this row**: every artifact produced by `device_gradient.py` before
this commit scored its square weights against their own transpose, and any other instrument
that restores orientation by comparing shapes has the same hole. `of3t-direct`'s
`instrument_a_043_*` arms should be checked before their numbers are quoted again.
