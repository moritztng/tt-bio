# of3t-p10grad — how wrong is OF3T's shipped backward, op by op

VERDICT: GO
RESULT: **OF3T's backward has two systematic biases, and neither is the kernel the brief suspected.** Against a float64 reference validated by finite differences (worst fd residual 4.1e-09 against a 1e-06 bar), on 32 random draws per op: the **softmax** gradient `dx` reads **2.201e-02** relative L2 in fp32 and **2.391e-02** in bf16, with a consistent **-9.29e-03** systematic undershoot at 64 sigma — but the **backward closure alone reads 9.75e-05**, so the error is the FORWARD `ttnn.softmax`, whose device row sums come back **-4.57e-03** short of 1. Configuring that forward (`TT_BIO_SOFTMAX_CKC`, shipped but off) takes `dx` to **9.086e-04**, **24x**, and flips the verdict from BIAS to NOISE. The **layer norm** gradient is clean at K=128 (`dx` **1.481e-04** fp32) and MARGINAL at K=384 (**2.089e-03**, **+2.08e-03** systematic at 11327 sigma) because `ttnn.mean` carries a low-precision `1/K`: **1/128 is a power of two and exact, 1/384 is not.** Replacing the four means in the closure with `ttnn.sum(precise) * (1/K)` takes `dx` to **1.399e-04**, **15x**, and BIAS to NOISE, while the same substitution at K=128 reproduces the shipped number to all 17 digits of both statistics — which is the control the hypothesis predicted. `ttnn.moreh_layer_norm_backward` is **not on OF3T's path at all**. All 70 graded gradients repeated **bit-exact** in-process.
COMMIT: `942ca825685c2f07e6a892170ddcd7187f4f99d7` (branch `wk/of3t-p10grad`); the sweep ran on `d8f50f775`, bar pre-registered at `520b99c48` before the harness existed
CARD: qb2 (tt-quietbox2) card 0, sole holder, lease `worker:of3t-p10grad`
AICLK: 1350 MHz min/median/max sampled DURING the sweep, n=34 in-process between cases plus n=9 from a second shell (min 800 there is the sample taken at launch before the first kernel)

BAR: `perf/of3t_p10grad/PREREGISTRATION.md`, committed at `520b99c48` before `opgrad.py` existed. fp32 PASS <= 1.0e-3, WRONG >= 1.0e-1; bf16 PASS <= 3.0e-2 (8x bf16 unit roundoff), WRONG >= 1.0. Reference accepted only if a central difference agrees to 1e-6. BIAS if `|mean(s)|/se >= 5` and `|mean(s)| >= 1e-3` over M=32 draws, where `s = <dev-ref, ref>/||ref||^2`.

PEROP: `perf/of3t_p10grad/PEROP.md`, rendered from `PEROP_M32.json` by `table.py` — the table is built from the artifact, never typed. Headlines, fp32 then bf16, `dx` unless named:

| op | fp32 | bf16 | bias | bit-exact repeat |
|---|---|---|---|---|
| `layer_norm` K=128 (pair z) | 1.481e-04 PASS | 3.902e-03 PASS | NOISE | yes |
| `layer_norm` K=384 (single s) | 2.089e-03 MARGINAL | 4.317e-03 PASS | **BIAS +2.08e-03** | yes |
| `softmax` K=64 (TriAtt) | 2.201e-02 MARGINAL | 2.391e-02 PASS | **BIAS -9.29e-03** | yes |
| `softmax` K=64, backward closure alone | 9.750e-05 PASS | 2.490e-03 PASS | — | yes |
| `softmax` K=64 (AttentionPairBias) | 2.206e-02 MARGINAL | 2.392e-02 PASS | **BIAS -9.27e-03** | yes |
| `softmax` K=384 | 2.230e-02 MARGINAL | 2.250e-02 PASS | **BIAS -8.41e-03** | yes |
| `linear` dx / dw | 1.345e-03 / 1.500e-03 MARGINAL | 1.706e-03 / 4.889e-04 PASS | BIAS -1.2e-03 | yes |
| `matmul` da / db | 1.345e-03 / 1.500e-03 MARGINAL | 1.706e-03 / 4.889e-04 PASS | BIAS -1.2e-03 | yes |
| `multiply` | 2.530e-08 PASS | 1.653e-03 PASS | NOISE | yes |
| `sigmoid` | 7.041e-08 PASS | 2.959e-03 PASS | NOISE | yes |

Nothing reads WRONG. Two things read MARGINAL with a bias, and both are attributed below.

VERB: **`ttnn.moreh_layer_norm_backward` is not called anywhere in `tt_bio`.** `grep -rn moreh_layer_norm_backward tt_bio/` hits two docstrings (`autograd.py:1295`, `ops.py:27`) explaining why it was rejected, and nothing else. The taped layer-norm backward is one composed closure, `autograd._taped_layer_norm:2735-2779`, shared verbatim by `ops.layer_norm` and `taped_ttnn._VERBS["layer_norm"]`: `ttnn.mean`, `subtract`, `multiply`, `mean`, `rsqrt`, `add`, `multiply`, then `_sum_leading` for `dgamma`/`dbeta`. The `dx` it produces reads **3.902e-03 bf16 / 1.481e-04 fp32** at K=128 — which independently reproduces the 4.607e-03 / 1.480e-04 the composed closure scored when `of3t-lnbw` measured it, and is 7 orders of magnitude off the 2.741e+06 the moreh kernel scores. **The suspect is dead on this path.** Upstream #12349 stays true about that kernel; it is simply not OF3T's kernel.

The softmax backward is `autograd.softmax_bw:197-218`: `multiply`, `sum`, `sum`, `divide`, `subtract`, `multiply`, with the row-sum renorm on (`TT_BIO_SOFTMAX_BW_RENORM` default true) and the fused route off (`TT_BIO_SOFTMAX_BW_FUSED` default false, so `ttnn.moreh_softmax_backward` is not called either). One helper, three callers: `autograd.softmax`, `triangle_attention`, `taped_ttnn._v_softmax`.

BIAS: **bias, not noise, on both — and the softmax one is not a precision effect.** Over 32 draws the softmax `dx` undershoots by **-9.29e-03** at 64 sigma in fp32 and **-1.19e-02** at 77 sigma in bf16. Same size in both dtypes is the signature: widening the arithmetic does not move it, so it is structural. Layer norm at K=384 overshoots by **+2.08e-03** at 11327 sigma in fp32. Layer norm at K=128, `multiply` and `sigmoid` are NOISE. `linear`/`matmul` carry a -1.2e-03 bias in fp32 only, which is Blackhole's fp32 matmul running as a bf16 decomposition — it is why fp32 buys nothing over bf16 there (1.345e-03 against 1.706e-03).

Averaging does not remove any of these. A step runs 816 layer-norm and 240 softmax backwards, every one of them the same sign.

## Where each bias comes from, by ablation rather than by argument

Each suspect got a CONTROL arm that rewrites the shipped algebra here with nothing changed. Both controls reproduce the shipped `dx` digit for digit (softmax fp32 2.201e-02, layer norm fp32 2.089e-03), so the rig is inert and the arms below measure the knob and not the rewrite. **The controls are inert for `dx` only** — they reduce `dgamma`/`dbeta` with a plain `ttnn.sum` where the shipped closure uses `_sum_leading`, so only the `dx` column of a diagnostic arm is readable.

**Softmax — the forward, not the backward.** Two knobs, one arm each:

| arm | fp32 `dx` | bf16 `dx` | row sum - 1 (fp32) |
|---|---|---|---|
| shipped | 2.201e-02 | 2.391e-02 | -4.57e-03 |
| CONTROL (nothing changed) | 2.201e-02 | 2.391e-02 | -4.57e-03 |
| numerator `ttnn.sum` given `precise_config()` | 2.201e-02 | 2.391e-02 | -4.57e-03 |
| **forward `ttnn.softmax` given `precise_config()`** | **9.086e-04** | **2.948e-03** | **-4.44e-04** |

`softmax_bw_inner:166` runs its reduction with no `compute_kernel_config` while the renorm denominator one line below gets one. That asymmetry looked like the defect and **costs exactly nothing**: the arm is identical to the control to every digit. The forward is the whole of it, and `tenstorrent.py:3282` already names the mechanism — `ttnn.softmax` normalises through a reciprocal whose range reduction loses up to 2.9e-2 when the exp-sum sits at or just above a power of two, which a confident softmax always does. The device's rows sum to 0.99543 instead of 1, `dx` carries `y` multiplicatively, and the renorm repairs only the inner product, not the leading factor. The repair is `TT_BIO_SOFTMAX_CKC`, which ships and is off by default because it moves shipped inference numbers on four models.

Independent corroboration that the closure is sound: graded against the shipped expression evaluated in float64 on the card's own `y`, `dx` reads **9.75e-05** fp32 and **2.49e-03** bf16 — PASS in every softmax arm, shipped included.

**Layer norm — `ttnn.mean`'s `1/K`.** Three knobs, one arm each, fp32 `dx`:

| arm | K=384 | K=128 |
|---|---|---|
| shipped | 2.089e-03 BIAS | 1.481e-04 NOISE |
| CONTROL (nothing changed) | 2.089e-03 | — |
| all three unconfigured `ttnn.mean` given `precise_config()` | 2.089e-03 | — |
| `rstd` as `reciprocal(sqrt(v))` instead of `rsqrt(v)` | 2.089e-03 | — |
| **every mean as `ttnn.sum(precise) * (1/K)`** | **1.399e-04 NOISE** | **1.481e-04, bit-identical to shipped** |

The two obvious suspects are both inert. What is left is the constant: `1/128 = 2^-7` is exact in any float, `1/384` is not, and if `ttnn.mean` carries that reciprocal at reduced precision the closure's four means all inherit one signed constant. That predicts a bias at K=384, no bias at K=128, and no change from the substitution at K=128. All three hold, the last one bit-exactly. It also explains the shape of the shipped table: `dbeta` is the one layer-norm gradient that is a bare `sum(g)` with no mean in it, and it reads **8.023e-08** — five orders cleaner than `dx` and `dgamma`, which both carry `rstd`.

Only the single track is affected. `c_s = 384`, `c_z = 128`, so the pair-track norms (TriMul, TriAtt, the pair transition) are clean and the `s` norms in AttentionPairBias and the single transition are not. It is a function of the channel width, so it does not move with crop.

## Determinism: not reproduced at the op

`of3t-p10trainout` concluded the device layer-norm backward is nondeterministic run to run and that this, not precision, is what makes device-only OF3T training different training. **At the op, on one card, in one process, it is not.** All 70 graded gradients across 12 cases, both dtypes, repeated bit-exact on a second run with freshly uploaded identical inputs. That does not contradict the step-level finding; it narrows where it can live. What this row rules out is the arithmetic of these two closures at these shapes. What it does not touch: allocation-dependent core assignment across a whole tape, fan-in order under `add_grad`, and anything the optimiser does.

## What this row does not claim

The fixes are measured at the op and nowhere else. Neither `TT_BIO_SOFTMAX_CKC` nor the `sum * 1/K` substitution has been run through a training step, a held-out loss or a fold, and `TT_BIO_SOFTMAX_CKC` is release-gated precisely because it moves shipped inference numbers on Boltz-2, Protenix-v2, OpenFold3 and ESMFold2. Both stay on this branch. `of3t-p10smbw` and `of3t-p10lnbw` own taking them further; the harness is theirs to extend rather than to rewrite.

The brief's attribution table (softmax owns 7ohe's damage, layer norm owns 7vus's) is not confirmed or denied here. `of3t-p10trainout`'s own later finding is that those two-step readings are bimodal — ten runs at one seed split 6 blown / 3 fine / 1 mixed — so they are two faces of one coin rather than two attributions, and a per-op relative L2 does not predict a held-out loss either way.

HARNESS: `perf/of3t_p10grad/opgrad.py`, run by `perf/of3t_p10grad/arm.sh`, rendered by `table.py`.

    bash perf/of3t_p10grad/arm.sh <TAG> --draws 32 [--ops layer_norm,softmax] [--dtypes bfloat16]

Twelve cases on the trunk's own shapes. Each draws fresh inputs, pushes them to the card, **reads them back** so the reference runs on the card's own values and input rounding is not charged to the op, takes float64 grads through torch autograd, validates them once per case by central difference, then drives the shipped taped verb (`ag._taped_layer_norm`, `tp._v_softmax`, `ag._taped_linear`, `tp._v_matmul`, `tp._VERBS[...]`) and compares. An op that reads its own output also gets graded against the shipped expression in float64 on the card's `y`, which is what separates a forward defect from a backward one. `arm.sh` samples AICLK every 4 s from a second shell; `opgrad.py` samples it again between cases.

One guard worth keeping: the harness reports the FORWARD relative L2 beside the gradients, and that is what caught its own first bug. `ttnn.linear(x, w)` is `x @ w`, not torch's `x @ w.T`; the reference had the transpose and graded `linear` WRONG at 1.407 in both dtypes — `sqrt(2)`, two uncorrelated vectors of equal norm. The forward read 1.407 too, which cannot happen to a correct reference, because the forward is the one thing both sides compute the same way. Fixed in `d8f50f775`; `linear` reads 1.706e-03 bf16.
