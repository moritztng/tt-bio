# The Boltz-2 token diffusion-transformer layer, on two parts

The headline is a transfer failure, not a fraction. The same integrated arm is **1.0122x on
Wormhole and 1.1345x on Blackhole**, so the Wormhole screen this row was commissioned to run
would have killed a lever worth **0.41 fold-seconds** on the part the cell of record is measured
on.

Two sessions, each calibrated on its own part:

| | Wormhole B0, whglx `j10glx02` card 2, 8x9 | Blackhole p150a, pc card 0, 13x10 |
|---|---|---|
| dense bf16 HiFi4 4096-cube | 54.44 TFLOP/s | 128.39 TFLOP/s |
| starved 8192^2 add | 238.5 GB/s | 421.6 GB/s |
| A/A floor, one sampling step | 0.16 % | 0.27 % |

Code `dit_layer.py`, `dispatch_probe.py`, `verify_gamma_fold.py`; data `dit_layer_*.json`,
`dit_step_*.json`, `dispatch_probe_*.json`, `verify_gamma_fold.json`.

## The mechanism, demonstrated rather than named

`ttnn.layer_norm` on `[1,512,768]`, `[1,1024,768]` and `[1,2048,768]` is a straight line in the
row count. Extrapolated to zero rows:

| | Wormhole | Blackhole p150a |
|---|---|---|
| fixed cost per op | **20.68 us** | **12.77 us** |
| marginal cost per 512x768 block | 8.44 us | 3.69 us |

**That cost is on the device, not on the host.** The same arms under a captured ttnn trace, with
host dispatch removed entirely, return 29.12 us against a synced 29.35 us on Wormhole and
16.46 against 16.13 on Blackhole — trace replay moves the number by 0.4 %. The whole
`DiffusionTransformerLayer` behaves the same way: 1138.5 us synced against 1143.2 us traced on
Wormhole, 624.2 against 625.8 on Blackhole, with host dispatch at 528.5 and 506.2 us, hidden
underneath. So this is a program-launch floor, and `--diffusion_trace` cannot reach it, which is
consistent with that flag's measured 0.9779x.

Blackhole's floor is only **1.62x** lower than Wormhole's while its matmul rate is **2.36x**
higher. Relative to what the part can compute, a Blackhole op launch is **1.46x more expensive**
than a Wormhole one. Every consequence below follows from that one ratio.

## The layer

| | Wormhole | Blackhole |
|---|---|---|
| shipped layer | 1.1424 ms, 19.7 % of cube | 0.6265 ms, 15.3 % of cube |
| its matmuls alone (`mm_only`) | 0.5371 ms, **39.2 %** of cube | 0.2363 ms, **37.8 %** of cube |
| matmul share of the layer | 47.0 % | **37.7 %** |

The arithmetic is not the problem on either part: 39.2 % and 37.8 % of the dense cube puts this
class second in the fold behind OuterProductMean's 61.8 % and above the pair Transition's 28 %,
and the two agree to 0.964, matching `roof-shape-honest-roofs`' 0.969 transfer for a
fraction-of-own-cube. What differs is how much of the layer the matmuls are, and on the faster
part they are **less** of it.

## The lever: concatenate the conditioning half across all 24 layers

Six of the layer's eleven matmuls are a pure function of `s` — two AdaLN scale/shift pairs and
two output projections, 3.624 of 12.281 GFLOP. Every layer in a step sees the *same* `s`, and
`nn.LayerNorm(dim, bias=False)` is `gamma_i * s_hat` with `s_hat` shared, so folding `gamma_i`
into the i-th weight block makes all 96 AdaLN projections read one tensor. The whole step's
conditioning half becomes one parameter-free layer_norm and two `[768, 96*768]` and
`[768, 48*768]` matmuls in place of 144 matmuls and 48 layer_norms.

Same FLOPs, same dot products, 2 launches instead of 144 — then 144 slices to hand the blocks
back, and that tax is what decides it:

| | Wormhole | Blackhole |
|---|---|---|
| shipped, one step's conditioning half | 6.4063 ms | 3.9866 ms |
| what the concatenation gains | 1.5961 ms | **2.4771 ms** |
| what the 144 slices cost | 1.5822 ms (11.0 us each) | 1.2235 ms (8.50 us each) |
| net | **0.0140 ms** | **1.2536 ms** |

**The slice tax is nearly part-independent; the concatenation gain is not.** Blackhole's wide
matmul reaches 45.4 % of its own cube where Wormhole's reaches 33.0 %, against a cube that is
2.36x larger. On Wormhole the two terms cancel to nothing. On Blackhole they do not.

## Integrated, one whole sampling step, 24 layers

Never the per-arm numbers multiplied together — the concatenation restructures across layers, so
the A/B has to be a whole step.

| arm | Wormhole | Blackhole |
|---|---|---|
| `step_ship` | 27.3833 ms | 15.0004 ms |
| `step_ncat` | 27.0544 ms, **1.0122x** | 13.2218 ms, **1.1345x** |
| `step_ncat_L1` | 26.1438 ms, **1.0474x** | 12.8318 ms, **1.1690x** |

`step_ship / 24` is 0.6250 ms against the 0.6265 ms per-call arm, 0.24 % apart, so the two
constructions agree.

`L1` is a second, smaller lever stacked on top: `ttnn.experimental.nlp_concat_heads` replaces the
four-op head re-assembly in `AttentionPairBias` (0.0637 -> 0.0163 ms on Blackhole), at the cost
of a gate and output projection at width 1024 instead of 768, which is exact because SDPA's pad
lanes are zero. **On its own it is 1.0246x on Blackhole and 1.0401x on Wormhole, under the 1.05x
kill line on both**; inside the stack it is worth a further 1.0304x, which is why it is reported
as a stack and not as a lever.

## What it is worth in fold-seconds

The token DiT layer is **2.819 s of the 17.340 s cell** (`roof-budget`: 4800 calls x 0.838 ms
= 4.021 s/fold, scaled by 0.7011). Rate-transferring this row's own quiet p150a measurement to
the p300c cube instead gives 3.68 s, so treat 2.819 s as the conservative denominator — the
0.7011 scale is the one `roof-true-floor` flagged.

| | at 2.819 s | at 3.68 s |
|---|---|---|
| `step_ncat` | **0.334 s**, 14.5 % of the 2.309 s prize | 0.436 s, 18.9 % |
| `step_ncat_L1` | **0.408 s**, 17.7 % of the prize | 0.532 s, 23.0 % |

For scale, `roof-wrong-part-envelopes` correctly killed a 1.068-1.301x op lever worth 0.006 % of
the cell. This stack is **2.35 % of the cell**, 390x that.

## Accuracy

The gamma fold is the only step that changes any arithmetic. Against a **float64** reference,
never another approximation, over 8 trials at `[512,768]`:

    shipped   gamma on the activation   rel RMS 2.347569e-03   PCC 0.999997244
    folded    gamma on the weight       rel RMS 2.349034e-03   PCC 0.999997241

**The fold costs 1.0006x the shipped path's own distance from the truth.** It is not bit-exact
(the two differ by 3.32e-03, the same size as either one's own error, because they are two
roundings of one quantity), and under the standing ruling that the bar is accuracy rather than
bit-exactness that is a pass on the arithmetic. The concatenation itself and the slices change
nothing at all.

## Two levers that died, and why they are worth recording

**`ttnn.mac` for the AdaLN scale-and-shift: 0.9867x on Wormhole, 0.9911x on Blackhole.** Three
ops become two and the layer gets *slower* on both parts, because the shipped `multiply_`/`add_`
pair writes into buffers that already exist and `mac` allocates its output. Fewer launches is not
automatically faster when the launch you remove was an in-place one.

**Step-blocking the conditioning half 8 deep instead of concatenating across layers**
(legal — the sigma schedule is fixed, so `s` for a future step does not depend on the coordinates
of the step before it): 1.436x on the Blackhole conditioning half against the concatenation's
1.504x, and it costs 8x the live conditioning residency. The across-layer form is strictly better.

## Owed before this lands

1. **The fold-level arm.** Nothing here is implemented in `tt_bio`; the concatenation needs a
   load-time weight transform (concatenate 96 and 48 blocks, fold gamma) and a rewrite of
   `DiffusionTransformer.__call__` to hoist the conditioning half out of the layer loop. The
   number of record must be one benchlocked interleaved fold-level A/B, not this harness.
2. **qb2, the part of record.** Every Blackhole number here is p150a. The cell is p300c, whose
   cube is 104.93 against this session's 128.39, and the lever's value depends on the ratio of
   the launch floor to the matmul rate, which is exactly what differs between the two boards.
3. **The structure check.** 512 aa and 298 aa against the full 0.60 A bar with the 1.84 A and
   0.83 A seed floors beside them, and CA-lDDT.
4. **The size ladder.** 298 / 512 / 768 / 1024. The concatenated conditioning output is
   96 x 768 wide regardless of sequence length but scales with the token axis: 113 MB live at
   512 aa, 226 MB at 1024 aa, held across the whole 24-layer pass.
