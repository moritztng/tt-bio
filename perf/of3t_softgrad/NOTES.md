# of3t-softgrad — the two shippable softmax levers, on the gradient, at full scope

Pre-registration `PREREGISTERED.md`, pushed as commit `c69129695` before the first arm ran. Bands,
bars and controls are that file's and none of them moved afterwards.

## The answer

Over 547 tensors holding 51.1358 % of the model's squared gradient norm, against upstream
OpenFold3's own bf16 training step (digest ff78d7bc… verified before loading) and against a
float64 reference built from upstream 0.4.3:

    arm                     v their step   v float64        r   cos v them   err-cos   x thresh
    shipped                 7.426217e+00  7.568637e+00   7.7814    0.411379   +0.2071   129.1304
    softmax_precise         2.566369e-01  2.742371e-01   1.0681    0.971340   +0.3250     4.4625
    softmax_accurate        1.076374e-01  8.197720e-02   0.9523    0.995112   -0.1930     1.8716
    softmax_host_f64        7.777580e-02  5.930664e-02   0.9787    0.997141   +0.0977     1.3524
    ckc_on                  7.426217e+00  7.568637e+00   7.7814    0.411379   +0.2071   129.1304
    accurate + permuted cot 1.527615e+00  1.546972e+00   1.1679    0.012959   -0.1094    26.5629

Perfect-fix threshold vs their step 5.750945e-02. A28 reachability floor, upstream's own bf16 step
against the same float64 reference: 5.852018e-02, r = 1.017575, with 441 of its own 547 tensors
over the 5.0e-02 per-tensor bar carrying 41.6989 % of the mass.

`softmax_accurate` at 1.076374e-01 is **worse than the 0.0777758 host bound**, which is the third
pre-registered band: the device levers do not reach the host bound at scope and the block-8 ladder
was optimistic. On block 8 accurate read 3.58x its bar; at scope it reads 5.38x the 2.0e-02
mass-weighted bar. What the levers do buy is 69.0x of the shipped arm's gap for 1.661x the cost.

`ckc_on` is bit-identical to `shipped` — 547 of 547 tensors, rel_l2 exactly 0.0 — so its row above
is the shipped row and is printed only to make that explicit.

## The 0.4.3 boundary had to be rebuilt

`/home/ttuser/of3t_rebase/` was pruned with its row's worktree, and it held `diffcap043`, the
boundary every arm scores against. Five live scripts still point at it and die with a bare
`FileNotFoundError`: `perf/of3t_trajectory/devgrad_traj.sh`, `perf/of3t_residual/devgrad_res.sh`,
`perf/of3t_residual/square_check.py`, `perf/of3t_conditioning/capture_cond_boundary.py`,
`perf/of3t_rebase/devgrad043.sh`.

`/home/ttuser/of3t_refprec/bundle_ref` IS `bundle_min_043` — its own `MANIFEST.json` names the
pruned path as its canonical location, and the five declared hashes are the five the original
capture verified. `recap043.sh` rebuilds into `/home/ttuser/of3t_softgrad/diffcap043`, under
`$HOME` rather than inside a worktree that can be pruned the same way. Checked, not asserted:

| | recorded | rebuilt |
|---|---|---|
| forward loss | 1.2675874205688995 | 1.2675874205688995 |
| cotangent norm | 0.019426651390714835 | 1.942665e-02 |
| `vs_bundle` worst rel | 0.0 | 0.0 |
| `of3t-conditioning`'s structure-0 forward rel | 1.104135e-02 | 1.104135e-02 |

`OMP_NUM_THREADS` pinned at the 8 the original used: a float64 CPU reduction is a function of its
thread count. Both end arms then reproduced their published values to every digit, which is the
instrument floor this row rests on.

## Two defects, both in shared code

AMENDMENT 1 offered a source-grounded hypothesis for the non-finite backward: that
`_accurate_softmax` has no taped counterpart, so either the saved output `P` is not what the chain
produced, or the five ops are taped individually and the `divide` carries a 1/x^2 term. **Both
branches are refuted by the instrument.** The rule creates exactly ONE tape node, the same shape
as the shipped rule; a finiteness probe over all 30 softmax calls per structure found the incoming
cotangent, the saved output, the row-sum reduction and the emitted input-gradient all finite,
count 0; and wrapping all 37 tape verbs put the first non-finite value at forward index 134, the
softmax's own FORWARD output at shape [1, 14, 4, 32, 128]. The backward pair was never the defect.

**D110 — `setdefault` cannot install a lever over an explicit argument.** `of3t-adaln`'s rule did
`kwargs.setdefault("compute_kernel_config", precise_config())`, and both softmax call sites in
this path pass that argument explicitly (`openfold3_diffusion_transformer.py:209`,
`openfold3_atom_transformer.py:184`). `softmax_ckc(token)` returns `None` when the site flag is
off, so the key is present with value `None` and `setdefault` cannot write it. The arm fired 1,440
times and returned all 547 gradient tensors bit-identical to shipped — `rel_l2` exactly 0.0. The
intercept count could not catch it: a count proves the rule ran, not that it changed anything.
Fixed by assignment, and with `_SOFTMAX_PRECISE_CKC` rather than a second copy of it. The block-8
`softmax_precise 0.349441` row was scored through the same `setdefault` and should be re-taken
before it is quoted again.

**D111 — `ttnn.max` rounds to bf16, so the 5-op accurate softmax divides 0/0 on a fully-masked
row.** The atom encoder's block-sparse attention produces rows whose keys are all masked at -1e9
for a padded query block. `ttnn.max` returns a bf16-rounded maximum, which on such a row comes back
**1,755,648 above** the true row max; every exponent is then -1.76e6, `exp` underflows the whole
row to zero, the sum is zero and the divide is 0/0. `FULLY_MASKED_ROW_OVERFLOW.json` measures it
standalone: 114,688 non-finite entries, exactly the masked rows, where the fused kernel on the
identical input is finite. On real rows the same rounding undershoots by up to 0.0625.

The repair is one op, and getting it wrong twice is instructive:

| variant | finite | forward | per-tensor median vs float64 |
|---|---|---|---|
| `_accurate_softmax` as it ships | no | — | — |
| clamp exponent at -88 | no | — | — |
| clamp at -60 AND clip above at 0 | yes | 1.562432e-02 | 1.258712e-01 |
| clamp at -60 only | yes | 6.298853e-03 | 4.826816e-02 |

-88 fails because exp(-88) is subnormal in fp32 and flushes to zero, so the sum is zero again.
Clipping from above looks harmless and is not: it clips away the entries where `ttnn.max`
undershoots, and cost the arm 2.5x on its forward and 2.6x on its median. Both variants are kept
in this directory so the difference is auditable.

## The forward is blind, and worse than blind

`forward_rel_median` over 48 structures: shipped 8.474801e-03, softmax_precise 6.338715e-03,
softmax_accurate 6.298853e-03, softmax_host_f64 6.462246e-03. The forward spans **1.35x** while
the gradient spans **95.5x**. Block 8 read 1.56x against 52.75x, so the blindness is confirmed at
scope and is wider here. It also does not ORDER the arms: `softmax_accurate` has the best forward
and the second-best gradient, `softmax_host_f64` the third-best forward and the best gradient.

## Cost on the real arm, AICLK sampled during

Card 0, 48 structures per arm, clock read from `qbcard/cardtel.tsv` over each arm's own window.

    arm                 s/struct   median   x shipped   wall s   AICLK mean   min    max    n
    shipped                 1.23     1.20       1.000       66         1300   800   1350   33
    softmax_precise         1.23     1.20       1.002       65         1316   800   1350   32
    softmax_host_f64        1.77     1.70       1.441       90         1325   800   1350   45
    softmax_accurate        2.04     2.00       1.661      105         1329   800   1350   52
    accurate + permuted cot 2.02     2.00       1.642      104         1329   800   1350   52

`of3t-softmax`'s per-op factors — precise 1.46x, accurate 4.71x — are one softmax shape and do not
survive contact with the scope. Precise is free here; accurate costs 1.661x, not 4.71x.

## Per tensor, against both references (A27)

Bar 5.0e-02, out of 547. `their_step`'s denominator is upstream's gradient; `float64`'s is the
float64 gradient. Different measurements.

    arm                 ref              massw       median        worst   >bar   mass>bar %
    shipped             their_step  7.426217e+00 1.398141e-01 1.803328e+01    519     97.5871
    shipped             float64     7.568637e+00 1.250047e-01 1.850397e+01    459     94.4376
    softmax_precise     their_step  2.566369e-01 8.349172e-02 6.582536e-01    474     80.1320
    softmax_precise     float64     2.742371e-01 5.271298e-02 6.586190e-01    295     53.8270
    softmax_accurate    their_step  1.076374e-01 8.288914e-02 3.970744e-01    474     83.1242
    softmax_accurate    float64     8.197720e-02 4.826816e-02 2.817863e-01    255     48.0179
    softmax_host_f64    their_step  7.777580e-02 8.215627e-02 4.463192e-01    472     85.4292
    softmax_host_f64    float64     5.930664e-02 4.885781e-02 2.618132e-01    254     52.0828
    accurate+permcot    their_step  1.527615e+00 1.526611e+00 2.103859e+01    547    100.0000

`softmax_accurate`'s median against float64, 4.826816e-02, is already inside the host bound's
4.885781e-02 while its mass-weighted headline is outside it. That is A23 in one line: the median
describes the massless half.

## The ckc_on arm, which AMENDMENT 2 withdrew after it had already run

AMENDMENT 2 retired this arm as moot, on the grounds that the `ttnn.max` repair put a compute
kernel config onto the op by another route and `softmax_precise` at 2.566369e-01 had already
answered what a config does to this gradient. It was already running, so it is reported as the
cheap confirmation the amendment asked for rather than quietly dropped, and it confirms that
reasoning exactly.


`TT_BIO_SOFTMAX_CKC=1` is honoured (`_SOFTMAX_CKC` read True) and reaches nothing here. The census
of the config actually passed to each softmax: **`None`, 1,440 of 1,440** — the op's own default,
HiFi2 with math_approx on and no fp32 destination accumulation. `FP32_SOFTMAX_STATS` came back all
zero, so `_fp32_softmax_attention` at `tenstorrent.py:3824`, the only consumer of that flag, is
never entered on this scope. 547 of 547 gradient tensors bit-identical to shipped.

## Controls

**A16** zero model reads 1.000000e+00 on every arm, measured. **Instrument floor**: the two end
arms reproduced their published values to every digit on a rebuilt boundary. **Upstream's own
step**: 5.852018e-02 from float64. **A break control that fails**, run on a lever arm rather than
the saturated shipped one: the permuted cotangent takes `softmax_accurate` from 1.076374e-01 to
1.527615e+00, 14.2x, and the cosine against their step collapses 0.995112 → 0.012959 with all 547
tensors over the bar. Its forward is byte-for-byte the arm's own, which is what makes it a control
on the comparison rather than on the model.

## Re-running

    perf/of3t_softgrad/recap043.sh                                  # ~17 min CPU
    perf/of3t_softgrad/chain.sh shipped sm64 precise accurate accpermcot ckcon
    perf/of3t_softgrad/rank5.py                                     # the op in isolation
    perf/of3t_softgrad/rank5b.py                                    # the fully-masked row

then, with no card:

    perf/of3t_residual/a25.py --f64 … --bf16 … --bf16-sha ff78d7bc… --arms label=…pt
    perf/of3t_softgrad/bars.py --f64 … --bf16 … --arms label=…pt
    perf/of3t_softgrad/cost.py --logs label=…log

## Gated

Nothing ships. No default moved. Every arm is on `wk/of3t-softgrad` and nothing is merged.
`softmax_host_f64` is a host computation on a tape verb, a diagnostic bound, not a product
candidate. The -60 exponent clamp is a repair the arm needed in order to run at all; shipping it
would be a separate decision with its own inference evidence, which this row did not take.
