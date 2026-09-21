# of3t-tapediverge — the tape's divergence from the shipped execution, measured on four defects

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: VERIFY/BENCHMARK + ALWAYS-ON | memories read:
`a-lever-can-fire-and-be-inert`, `negative-control-must-break-what-check-reads`,
`eligibility-firing-condition-is-not-a-code-fact`,
`ratio-must-name-how-its-denominator-arm-was-built`,
`relative-l2-alone-cannot-identify-the-error-direction`,
`concluded-marker-artifacts-inside-a-worktree-get-pruned`,
`donecheck-qb-state-mirror-one-way-sync`,
`concluded-marker-must-be-written-on-the-donecheck-host`; of3t PROTOCOL A14/A15/A16/A23/A24/
A26/A27/A28, DEFECTS D30/D31/D32/D33/D55/D58/D121; state docs `of3t-wholemodel`,
`of3t-f64softmax`, `of3t-auxheads`, `of3t-theirtest`, `of3t-rebase`.

Branch `wk/of3t-tapediverge`, cut from `origin/wk/of3t` at `7f626e7bf`. Artifacts
`perf/of3t_tapediverge/`. **`git diff origin/wk/of3t -- tt_bio/` is empty**: every lever this row
prices is installed by monkeypatch from `tdrun.py`, nothing merges, no shipped default moves, and
`TT_BIO_SOFTMAX_BW_RENORM` stays default-off. Pre-registration committed at `e9e88c80c` before
the first number of this row existed.

VERDICT: PARTIAL — the ~20x backward amplification **survives the softmax-backward repair at
11.03x on the diffusion module and 10.90x on `msa_module`**, so D58's generalisation now stands on
post-repair evidence; D31 is **refuted on both halves** — the fused-SDPA tape verb is reached
**0 times** in the module holding 89.2 % of the gradient, and where it is reached an 8-site chain
turns a 6.345540e-02 forward gap into a 1.573396e-03 gradient gap that is flat in depth; D32's 21 sites in 9 modules are
confirmed on this tree with 5 of 21 line numbers stale; and AMENDMENT 1's kernel-config asymmetry
fires 1,440 times and leaves all 547 gradient tensors **bit-identical**. The charter's second
half now has a device number on both sides at crop 384: **230.11 s taped against 9.721 s
untaped, 23.7x**, on one card with the AICLK sampled DURING. All five deliverables landed with
measurements. PARTIAL rather than GO for one honest reason: the taped figure is a **trunk** step,
which is a floor on a training step and not one, because it excludes the diffusion module's 48
differentiated noise levels, every loss head and the optimizer. Everything else in this row's
brief is answered.

AMPLIFY: the forward and the gradient of the same scope out of the **same process**, which is the
thing D30's own warning asks for. 48 structures, the 0.4.3 boundary
(`/home/ttuser/of3t_softgrad/diffcap043`), reference `grads_f64_043.pt`, qb2 card 0.

    arm            forward median   gradient median   ratio    worst        AICLK during (n)
    shipped        8.474801e-03     1.250047e-01      14.75x   1.850397e+01  mean 1319 MHz (71)
    shipped A/A    8.474801e-03     1.250047e-01      14.75x   1.850397e+01  mean 1314 MHz (77)
    renorm         8.474801e-03     9.344246e-02      11.03x   5.523992e-01  mean 1350 MHz (34)
    renorm+sm216   8.474801e-03     9.344246e-02      11.03x   5.523992e-01  mean 1316 MHz (33)
    renorm+sumall  8.474801e-03     9.344246e-02      11.03x   5.523992e-01  mean 1315 MHz (32)
    break control  8.474801e-03     1.567592e+00      —        —             mean 1317 MHz (34)

Every arm compares the same 547 tensors holding 57.3203 % of the diffusion module's squared
gradient norm. **The forward is bit-identical on every arm to all 16 digits**, which is the
repair's own claim checked rather than assumed: it touches no forward, so the pair moves only
through its gradient half.

`msa_module`, D58's second module, from its own single harness
(`perf/of3t_auxheads/msa_instrument.py`, re-cut per section in `perf/of3t_tapediverge/
SECTIONS.json` from of3t-wholemodel's per-tensor sidecars, which reproduce that instrument's own
mass-weighted figure to every digit):

    arm       forward (real block)   gradient (mass-weighted vs float64)   ratio
    shipped   8.176476e-03           1.621240e-01                          19.83x
    renorm    8.176476e-03           8.915364e-02                          10.90x

**So the factor moves partly and does not collapse.** Diffusion 14.75x -> 11.03x, `msa_module`
19.83x -> 10.90x. Two independent modules, two different harnesses, two different references, and
after the repair they agree at **11.03x and 10.90x** — the same coincidence D58 reported at 19.6x
and 19.8x, one factor lower and still two significant figures apart by 1 %. D58's claim that the
amplification belongs to the backward rather than to any module is now supported by post-repair
evidence instead of pre-repair evidence, at roughly half the size.

**One provenance correction, and it matters for D30's headline.** D30's published pair is forward
8.34e-03 / gradient **1.6588e-01** / 19.6x, from `of3t-rebase`'s `device_gradient_043all.json`.
The shipped arm on today's composition reads **1.250047e-01** for the same 547 tensors. The
`forward_rel_median` (8.474801e-03) and the worst tensor's value (1.850397e+01) are bit-identical
between the two runs, so the boundary and the reference are the same object and the difference is
not a capture. What changed between them is the tree: `of3t-softgrad`'s and `of3t-nanfloor`'s
softmax work merged into `wk/of3t` after `of3t-rebase` measured. That is the only candidate this
row can name from the evidence it has; the mechanism is not demonstrated here. Either way
**D30's central number is 14.75x on the shipped arm of the current composition, not 19.6x**, and
11.03x after the repair. `of3t-f64softmax`'s independent run of the same arm on 2026-09-20 read
1.250047e-01 to every digit, from a different worktree on a different card, so the 14.75x is
reproduced twice.

FUSEDIFF: D31's discriminator, run as one arm with no external reference, and the answer is a
**reach** answer before it is a magnitude answer.

`tdrun.py`'s `sdpa_selfvalue` lever forces `value=None` into `autograd.triangle_attention`, so the
tape's forward value becomes the function its own backward differentiates. Across the whole
48-structure diffusion arm it served **0 calls**, and the run failed its own reach check rather
than reporting the unlevered arm under another name. A second, independent lever says the same
thing from the other side: `sum_precise_all` intercepts every `ttnn.sum` that passes no kernel
config and counted **1,440 unconfigured and 12,816 already-configured** calls — the 1,440 are
exactly the softmax verb's own numerator at `taped_ttnn.py:216`, one per softmax backward, and
**none** come from `autograd.py:949`, which sits inside `triangle_attention`'s backward.

**So the fused-SDPA tape verb is never reached in the diffusion module**, which holds 89.2 % of
OpenFold3's squared gradient norm and is the module D31 was raised as a candidate for. D31's
mechanism is real in the code and it is **not** the explanation for D30. Recorded as a refutation
of the attribution, not of the defect: D31's own forward measurement stands
(`perf/of3t_rebase/d31_fused_vs_recompute.json`: fused 3.250705e-02 from float64, the recompute
6.577058e-03, the two 3.328717e-02 apart, and the tape differentiates the **more** accurate of
the two), and where the verb IS reached — the pairformer trunk's triangle attention — this row
did not price it.

Consistent with that reach reading, the `selfvalue` arm's gradient dump is bit-identical to the
`renorm` arm's on all 547 tensors, which is what a lever that never fired must produce.

**And where the verb IS reached, the magnitude is measured.**
`autograd.triangle_attention`'s backward recomputes the scores from q, k, v and bias and never
reads the forward value, so at a single isolated site the two arms' gradients are identical by
construction; the mismatch can only enter through the activations one site hands the next, which
makes the measurement a chain rather than a site. `perf/of3t_tapediverge/d31_chain.py` builds
that chain: triangle-attention sites in series, the fused arm reproducing `_v_sdpa` line for line
including the `ag.scale` bias convention, against the same chain with `value=None`. Both arms are
ours, no external reference, and the taped leaves are the chain input and the bias.

    depth   sites fired   forward rel_l2   gradient mass-weighted   gradient worst
    1       1 / 1         3.008435e-02     0.0                      0.0
    2       2 / 2         1.962754e-02     1.539848e-03             2.794052e-02
    4       4 / 4         3.360687e-02     1.428074e-03             8.097675e-03
    8       8 / 8         6.345540e-02     1.573396e-03             8.352972e-03

**Depth 1 reads exactly zero and is the structural control**: the analytic prediction from the
docstring, confirmed on hardware, and proof that the comparator reads 0 when there is nothing
there and non-zero when there is. From depth 2 on the gradient difference is **flat at about
1.5e-03** while the forward difference grows to 6.3e-02 — so the mismatch **does not accumulate
with depth in the gradient**, which is the opposite of what "3.3e-02 per site across 24 DiT
blocks is the right order for a 19.6x factor" predicted.

Against the bars: 1.573396e-03 is **32x under the 5.0e-02 per-tensor bar**, **40x smaller than
the forward gap that caused it**, and **64x smaller than the 1.006695e-01 the repaired arm
already carries at model scope**. The pre-registered threshold was "under 5.0e-03 mass-weighted
refutes D31 as a material contributor for this model" and 1.573396e-03 is under it. These four
runs are accuracy measurements on an idle card (AICLK 800 MHz, the chain finishes in seconds);
the clock bears on no figure in this table.

ROUTING: D32's table re-derived from this tree by `perf/of3t_tapediverge/routing.py` rather than
quoted. **21 `ops.taping()` branch points in 9 shipped modules — the same count and the same nine
modules D32 published.** The raw grep is 27; the other 6 are comments, `ops.py:71`'s definition of
the predicate and `eltwise_fusion.py:97`'s helper body, and counting those would have read 27.

    tt_bio/triatt_qkv.py        74, 179, 276, 390     fused QKV projection
    tt_bio/tenstorrent.py       1146, 4334, 4800, 8761   trimul DRAM route, pair-proj L1 out,
                                                      L1 -> DRAM memory config at 8761
    tt_bio/eltwise_fusion.py    80, 104, 115          FUSE_MASK_ADD, FUSE_NORM_RESIDUAL
    tt_bio/reblock_permute.py   373, 597, 877         reblock/permute fusions
    tt_bio/softmax_generic.py   369, 514              fused softmax
    tt_bio/triatt_sdpa.py       340, 486              tt-bio's custom fused SDPA, QKV entry
    tt_bio/trimul_tail.py       232                   fused trimul tail
    tt_bio/mm_dualnoc.py        87                    dual-NoC matmul
    tt_bio/swiglu_fused.py      94                    fused SwiGLU

**5 of 21 line numbers have moved**: `tenstorrent.py` 1126/4129/4595/8531 are now
1146/4334/4800/8761, and `trimul_tail.py` 233 is 232. The scope statement is intact; the table
was stale exactly the way D31's `tenstorrent.py:7205` was.

STEP: measured on one host, one card, one structure, crop 384, batch 1, through
`perf/of3t_perf/step.py`, which intercepts a real `predict_one` so the input is the shipped
pipeline's own featurisation and there is no second one to drift. Both arms are the same scope in
the same process shape, so the taped-vs-untaped ratio below is a measurement and not an estimate.

    arm             no_grad prefix   final cycle   device backward   trunk step   AICLK during
    untaped         7.208 s          2.430 s       —                   9.721 s     mean 1322 (26)
    taped, rep 0    7.21 s           3.58 s        273.74 s          284.61 s      mean 1347 (240)
    taped, rep 1    7.54 s           3.41 s        219.07 s          230.11 s      mean 1348 (386)
    taped, rep 2    7.53 s           3.41 s        224.68 s          235.71 s      mean 1348 (386)

The untaped figure is the median of three reps that spread 4 ms (9.721 / 9.721 / 9.721 s). Rep 0
of the taped arm is cold and reps 1 and 2 are the steady state, which is why all three are given
rather than averaged; the taped run's whole 771 s window sampled AICLK mean 1348 MHz over 386
samples. Each taped backward reached **1,639 of 2,531 declared weights over 2,473 tape nodes**,
identical between reps, so it is a real backward over most of the trunk and not a truncated one.
Their 20-step mini rollout costs **0.929 s** directly and 0.921 s from the 4/12/20-rung fit, in
the same process on the same card, and it is untaped on both sides because upstream runs it under
`no_grad`.

**The taped trunk step is 23.7x the untaped one at the same crop on the same card**, steady
state, and 29.3x cold. The cost is almost entirely the backward: the tape adds 40 % to the
forward cycle (2.430 -> 3.41 s) and then 219.07 s of device backward that the inference route
does not execute at all. That is D32's
second consequence with a number on it — an inference s/step underestimates a training s/step by
a factor of twenty-four here, so no training throughput may be projected from one.

    our untaped trunk route, crop 384, batch 1, card 0, qb2 p300c   9.721 s + 0.929 s rollout
    our taped trunk step,    crop 384, batch 1, card 0, qb2 p300c   230.11 s steady,
                                                                    284.61 s cold
    upstream's own CPU step, crop 384, batch 1, bf16-mixed          394.61 s   (of3t-theirtest)

Upstream's CPU step is quoted, not re-run, and **a CPU number is not a GPU baseline**: no ratio
against it is a hardware comparison. The campaign's GPU record is 11 / 8 / 7 s on a rented H200 at
1980 MHz with the recycle count unpinned, which §4a does not accept as a baseline.

**What our 230.11 s is and is not.** It is the trunk at 4 cycles with the gradient on the final
one, which is upstream's own trunk shape, plus the download. It excludes the diffusion module's
48 differentiated noise levels, every loss head and the optimizer, exactly as D33's 870.75 s
trunk-cycle figure did — so it is a **floor** on a training step, not a step. For scale on the
part it excludes, the 48-structure taped diffusion arm this row ran takes 64 to 67 s wall clock
per arm end to end on the same card. The clock is sampled DURING every window above, and card 0
was held by this row alone for all of it.

CONTROL: three controls, and each one is reported with its counter as well as its effect.

  * **A/A.** `shipped` and `shipped2`, two processes, same tree, same card: `forward_rel_median`,
    `median_rel`, `worst_rel` and `compared` identical to every digit, and the gradient dumps
    **bit-identical on 0 of 547 tensors moved**. The A/A floor on this instrument is **0**.
  * **A break control that MOVED the reading.** `--permute-cot` pairs structure k's backward with
    structure k+1's cotangent, forward and weights untouched. The renorm arm goes from
    9.344246e-02 to **1.567592e+00**, a **16.8x** move, and the gradient dump moves on **547 of
    547** tensors with a worst per-tensor move of 2.354716e+01. The comparator that reads
    "bit-identical" below therefore can see a change when there is one — that is the point of
    running it.
  * **REACH, per flag, per D121.**

        flag / lever            served        effect on the 547-tensor gradient
        TT_BIO_SOFTMAX_BW_RENORM  live=true   541 of 547 moved, 9.554996e-01 over the scope
        sdpa_selfvalue            0 calls     bit-identical (it never fired)
        sm216_precise             1,440       BIT-IDENTICAL, 0 of 547 moved
        sum_precise_all           1,440 + 12,816 already configured   BIT-IDENTICAL, 0 of 547
        TT_BIO_HOST_F64_SOFTMAX_AB (off)  served 0, declined 1,440   —

    **The renorm arm's first run was the shipped arm under another name and the guard caught it.**
    `TT_BIO_SOFTMAX_BW_RENORM` is read at import time by `tt_bio.taped_ttnn`; `tdrun.py` must
    import that module before it can patch it, so the harness's own `--softmax-bw-renorm`, which
    sets the variable after argparse, landed too late and the arm read
    `_SOFTMAX_BW_RENORM: false` while producing the shipped number. `arms.sh` now exports the
    variable before python starts. This is `eligibility-firing-condition-is-not-a-code-fact` with
    a working detector in front of it.

  * **AMENDMENT 1, both arms, and this is a result rather than a null.** `taped_ttnn.py:216`
    computes the near-cancellation numerator `sum(g*y)` with no kernel config while the
    denominator `of3t-apbgrad` added at `:218-219` gets `precise_config()`. Giving the numerator
    the same config **fires 1,440 times** — one per softmax backward, the same count the host
    float64 path declines — and leaves all **547** gradient tensors **bit-identical** to the
    renorm arm. The wide arm, which gives `precise_config()` to every unconfigured `ttnn.sum` and
    so would also reach `autograd.py:949`, is likewise bit-identical, and its counter shows why
    it cannot differ here: the only unconfigured sums this scope reaches are the same 1,440.
    **A lever that fires 1,440 times and changes nothing is the finding**, and the asymmetry at
    `:216` costs this scope nothing measurable. `autograd.py:949` is untested by this row, because
    the scope never reaches it.

DEFECTS:

  * **D30 — NARROWED.** The amplification is real and it is smaller than published in both
    directions. On the current composition the shipped arm reads **14.75x** (forward 8.474801e-03,
    gradient 1.250047e-01), not the 19.6x on the record, and the repaired arm reads **11.03x**
    (gradient 9.344246e-02). Forward and gradient now come from one harness, which is the fix D30
    asked for in its own closing warning. What survives: a forward at 0.85 % still buys a gradient
    at 9.3 %, and the gap is backward-specific because the forward is bit-identical across arms.
  * **D31 — REFUTED as a material contributor for this model, on both of its halves.** The
    mismatch exists in the code and its forward size is on the record at 3.328717e-02. Half one:
    it is **not reached at all** at the scope that carries the mass — the tape verb served 0
    calls over 48 structures and 1,440 softmax backwards in the module holding 89.2 % of the
    gradient. Half two, where it IS reached: an 8-site chain whose forward arms differ by
    6.345540e-02 produces gradients that differ by **1.573396e-03**, flat in depth, 32x under the
    per-tensor bar and 64x under the error the repaired arm already carries. The pre-registered
    refutation threshold was 5.0e-03 and this is below it. The defect's mechanism is real and its
    cost, measured at both ends, is not material here.
  * **D32 — CONFIRMED on part (1), STANDS on part (2).** The 21 sites in 9 modules re-verify
    against this tree with the same count and the same modules, 5 of 21 line numbers stale. The
    method constraint it states is now demonstrated rather than argued: at crop 384 on one card
    the same scope costs **9.721 s untaped and 230.11 s taped, 23.7x**, so a training throughput
    projected from an inference measurement is wrong by a factor of twenty-four at this crop. That is
    the first taped-versus-untaped ratio the campaign has measured rather than asserted.
  * **D58 — STANDS, on post-repair evidence.** Both modules keep a double-digit factor after the
    repair and they still agree with each other: **11.03x** and **10.90x**. The claim that the
    amplification belongs to the backward rather than to any one module survives the single
    largest change the backward has had, which is stronger evidence for it than the original
    coincidence was.
  * **D55 — this row's measurement supports no move at this scope.** The asymmetry it names at
    `taped_ttnn.py:216` is real, it fires 1,440 times on the diffusion arm, and closing it leaves
    every one of 547 gradient tensors bit-identical. Status is the orchestrator's to set; the
    measurement says the numerator's kernel config costs this scope nothing.

PROVES: at crop 384 on a p300c a taped trunk step costs 230.11 s against 9.721 s untaped, 23.7x,
so no training throughput is projectable from an inference measurement. On the diffusion module and on `msa_module`, with the softmax-backward repair on, the
backward still disagrees with the reference roughly eleven times harder than the forward does, and
the two modules agree on that factor to within 1 %. The forward is bit-identical across arms, so
the factor is a property of the backward and not of a forward that drifted. D31's mismatch, which
was the leading candidate for that factor, does not occur at all in the module that holds 89.2 %
of the gradient mass, and where it does occur an 8-site chain converts a 6.345540e-02 forward
divergence into a 1.573396e-03 gradient divergence that does not grow with depth. D32's routing divergence is 21 sites in 9 modules on this tree. The kernel-
config asymmetry at `taped_ttnn.py:216` is free to fix and buys nothing at this scope.

DOESNOT: this is one step's gradient on one batch at one crop, and it says nothing about
**stability over a full run**. Nothing here bounds drift over 100k steps, the behaviour of the
update rule in the long run, or whether an eleven-times-amplified backward compounds or cancels
across steps — a single-step agreement statement cannot distinguish those, and this campaign does
not claim a training run was reproduced. The 11.03x and 10.90x are two modules, not the model:
`diffusion_conditioning`, 36.9462 % of the model's mass, is bit-identical between the shipped and
the repaired arm and is not in either figure. D31's chain measurement is a synthetic stack of
triangle-attention sites, not the pairformer trunk's own shapes and biases, and the trunk itself
was not re-scored under the `value=None` arm. The taped s/step at crop 384 is a bound quoted from D33's
different-scope figure, not a measurement this row took. And the AMENDMENT 1 result is scope-
local: a lever that is inert on 1,440 diffusion softmax backwards may not be inert where
`autograd.py:949` is reached, which no arm here exercised.
