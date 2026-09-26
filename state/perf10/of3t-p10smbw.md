# of3t-p10smbw — the bf16 softmax backward: correct, and what it costs

VERDICT: PARTIAL
RESULT: **The softmax gradient is fixed and the fix is free.** `dx` relative L2 against a float64
reference goes **2.201e-02 -> 9.086e-04** fp32 and **2.391e-02 -> 2.948e-03** bf16 at TriAtt's
K=64, **24.2x** and **8.1x**, and all three softmax sites flip **BIAS to NOISE**: the -9.29e-03
systematic undershoot at 64 sigma reads -5.28e-04. On the step, measured as an A/B on ONE card at
ONE commit with only `_v_softmax`'s config differing, the warm step reads **48.942 s fixed against
49.758 s unfixed** and the backward phase, which is where the 240 softmax backwards live,
**29.078 s against 29.055 s: +0.023 s, +0.08%.** The lever is inside the step's own run-to-run
spread. **The defect was never the backward.** `dx = y (g - sum_j g_j y_j)` and the backward
closure alone already reads 9.750e-05; what it multiplies by is a forward `y` whose rows sum to
0.9954, and the one field responsible is `math_approx_mode` on `ttnn.softmax`. Clearing it under a
tape costs nothing because it is a compute-kernel flag, not an extra pass. The exact host softmax
this row was handed as the diagnostic runs 34x slower and is not needed.
COMMIT: `7a6c6dc39` (branch `wk/of3t-p10smbw`); both step arms ran at that sha, the base arm with
`tt_bio/taped_ttnn.py` checked out at `868a2eaab` (main) and nothing else changed
CARD: qb2 (tt-quietbox2) card 1, sole holder, lease `worker:of3t-p10smbw`
AICLK: 1350 MHz median sampled DURING both step arms, n=30 each (fix min 1337, base min 800 at
launch before the first kernel); 1350 median n=11 DURING the 32-draw gradient sweep

ROUTE: `tt_bio/taped_ttnn.py::_v_softmax` gives the forward `ttnn.softmax` a precise compute
kernel config **when, and only when, `ag.is_grad_enabled()`**. With taping off the config is
untouched, so every inference path is bit-identical to shipped, including an evaluation inside a
training run. `autograd.softmax` has passed `precise_config()` since it was written; this is the
same rule at the shipped verb, so there is one rule rather than two. The backward gets the same
config it ran forward with.

**Which verb actually runs.** The trunk's TriangleAttention softmax is `tenstorrent.py:4312`,
`ttnn.softmax_in_place(attn, compute_kernel_config=sm_ckc)` with `sm_ckc = None` unless
`TT_BIO_SOFTMAX_CKC` is set. Under the training shim that call is intercepted and taped out of
place by `_v_softmax`, which called `ttnn.softmax` with no config at all, so the op default
applied. `ttnn.moreh_softmax_backward` is **not** on the path: `TT_BIO_SOFTMAX_BW_FUSED` is off by
default and `softmax_bw` takes the composed six-verb route. `TT_BIO_SOFTMAX_BW_RENORM` is on, and
it is why the closure alone reads 9.750e-05 -- it repairs the inner product, but `dx` carries `y`
multiplicatively and nothing repairs the leading factor.

**Why one field.** `perf/of3t_d116_verify/APPROX.json`, four shape/seed cases at the trunk's own
shapes: of HiFi4, `fp32_dest_acc_en`, `packer_l1_acc` and `math_approx_mode`, only
`math_approx_mode` moves `ttnn.softmax` at all. The trunk caller's own config (HiFi4 + fp32acc +
packer, approx **True**) is **bit-identical** to passing no config; clearing approx alone
reproduces `precise_config()` to every digit and takes the forward from 1.8e-02 to 5.9e-04.

GRADERR: `of3t-p10grad`'s harness re-run unchanged, 32 draws, its own pre-registered bar
(`perf/of3t_p10grad/PREREGISTRATION.md`, fp32 PASS <= 1.0e-3, bf16 PASS <= 3.0e-2, BIAS if
`|mean(s)|/se >= 5` and `|mean(s)| >= 1e-3`). Artifact `perf/of3t_p10smbw/PEROP_FIXED_M32.json`;
the shipped column is `of3t-p10grad`'s `PEROP_M32.json`. `dx`, median over 32 draws:

| site | fp32 shipped | fp32 fixed | gain | bf16 shipped | bf16 fixed | gain | bias |
|---|---|---|---|---|---|---|---|
| `softmax/triatt_K64` | 2.201e-02 | **9.086e-04 PASS** | 24.2x | 2.391e-02 | **2.948e-03 PASS** | 8.1x | BIAS -> NOISE |
| `softmax/apb_K64` | 2.206e-02 | **9.055e-04 PASS** | 24.4x | 2.392e-02 | **2.950e-03 PASS** | 8.1x | BIAS -> NOISE |
| `softmax/triatt_K384` | 2.230e-02 | **1.089e-03 MARGINAL** | 20.5x | 2.250e-02 | **2.928e-03 PASS** | 7.7x | BIAS -> NOISE |

K=384 fp32 clears its bar by grade but not by a margin: 1.089e-03 against a 1.0e-3 PASS line. Said
here rather than rounded, because it is the one softmax number that does not read PASS.

**Two controls say the rig is inert, and one of them predicted the number.**
`softmax/triatt_K64_CONTROL` and `_SUMCFG` rebuild the shipped algebra OUTSIDE the taped verb, so
this patch cannot reach them, and they still read 2.201e-02 fp32 / 2.391e-02 bf16 -- the shipped
values, reproduced on the fixed tree. `softmax/triatt_K64_FWDCFG` is `of3t-p10grad`'s diagnostic
for exactly this change, and the shipped arm now matches it **to every digit in both dtypes**
(9.086e-04 / 2.948e-03). Layer norm, linear, matmul, multiply and sigmoid are unmoved, and all 70
graded gradients repeated bit-exact in process.

STEPTIME: `perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 --samples 48 --chunk 4 --reps 2
--no-exact`, the same invocation `of3t-p10noexact` used for the board's 62.237 s. Both arms on
**qb2 card 1**, back to back, 2,944 of 3,152 weights gradded and 87,297 tape nodes on both:

| | cold step | **warm step** | backward | AdamW | prefix | diffusion |
|---|---|---|---|---|---|---|
| fixed (`7a6c6dc39`) | 51.767 s | **48.942 s** | 29.078 s | 4.133 s | 3.583 s | 2.744 s |
| base (`_v_softmax` at `868a2eaab`) | 50.025 s | **49.758 s** | 29.055 s | 6.372 s | 3.633 s | 2.484 s |

The fixed arm is **0.816 s faster** on the warm step, which is not a win, it is noise: `AdamW` alone
moves 2.24 s between the two and it is a host phase that contains no softmax. The comparable number
is the **backward, +0.023 s (+0.08%)**, and that is the one phase all 240 softmax backwards and
their forwards' `y` sit inside.

**The board's 62.237 s is a qb1 reading and is not the baseline for this lever.** That run was
`qb1 card 1`, a **p150a**; both arms here are qb2 card 1, a **p300c**. Quoting 48.942 against
62.237 would book a 1.27x host difference as a softmax result. The honest same-card statement is
the table above. Against the board's number for scale: the fixed step is 48.942 s on qb2 where the
unfixed step is 49.758 s on the same card and 62.237 s on qb1.

N=1 warm rep per arm. The spread between the two cold reps (1.742 s) and between the two AdamW
phases (2.24 s) is larger than the difference being claimed, which is why the claim is "free", not
a speedup. More reps are cheap (118 s an arm) and are the first thing the next pass should add.

REJECTED:
- **The reduction.** `softmax_bw_inner:166` runs `ttnn.sum(ttnn.multiply(g, y))` with no
  `compute_kernel_config` while the renorm denominator one line below gets one. That asymmetry is
  the obvious suspect and it costs **exactly nothing**: `softmax/triatt_K64_SUMCFG` is identical to
  the control to every digit in both dtypes, at 32 draws. The cancellation in
  `dy*y - y*sum(dy*y)` is not where bf16 loses here.
- **Routing the backward through the exact host softmax.** It is the diagnostic, not the fix:
  `of3t-p10trainout` priced the exact trunk at 3938 s against 115 s, **34x**.
- **`TT_BIO_SOFTMAX_CKC`.** Same arithmetic, wrong scope. It is a process-wide switch that reaches
  every forward including inference, which is why it is release-gated on four models. Gating on
  `is_grad_enabled` gets the gradient repair with no inference movement, so the flag stays off and
  this change is not release-gated with it.
- **`ttnn.moreh_softmax_backward`.** Not touched. It is off by default and it computes the same
  expression; it would inherit the same bad `y`, so it is orthogonal to this defect.

ABPROOF: **owed, and the brief's version of it cannot be read as written.** The brief asks for
"healed means 7ohe lands near 10.76, not 18.92". `of3t-p10trainout` withdrew that attribution
itself: the metric is **bimodal**, ten device-native runs at one seed split 6 blown / 3 fine / 1
mixed on 7ohe, and the softmax-exact arm that read 10.766779 was **N=1** and lands inside a range
the unfixed arm covers on its own. A single run of either arm measures a coin. So the A/B has to be
graded on the **rate** of the blown mode over replicates, not on one number, and that is the next
pass: `~/of3t_p10trainout/corpus/train12` and `~/of3-weights/of3-p2-155k.pt` are both on qb2, a
device-native arm is 115 s, and `perf/of3t_p10trainout/trainarm.py` needs merging from
`origin/wk/of3t-p10trainout` into this branch with a runner rooted in this worktree (that row's
worktree is gone). Target n=8 per arm, which is ~16 min of card each side.

Nothing about the two-step A/B changes the op-level result: 24.2x against a float64 reference on
32 draws, with a control that reproduces the shipped number on the same tree, is not a coin.

## Two things fixed on the way that are not this row's lever

`perf/of3t_stepfloor/fullstep.py` raised `TypeError: 'float' object does not support item
assignment` at `_mark("after_backward")`, 118 s of card into every step, because
`row["mem_available_gib"]` is written per-phase as a dict by main and once per rep as a float by
`wk/of3t-p10noexact`. `of3t-p10fp32bw` root-caused and fixed this in `57b515eaf` and that commit
**never reached main**, so the harness is broken for anyone starting from main. The same one-line
rename is applied here.

`perf/of3t_p10grad/arm.sh` samples AICLK with `tt-smi -s`, which **opens the chip** to read
telemetry. On this host it brought up three cards and hung for minutes. `perf/of3t_p10smbw/arm.sh`
reads `/sys/class/tenstorrent/tenstorrent!N/tt_aiclk` instead, on all four nodes, so the log shows
which card carried the work as well as its clock and the sampler touches nothing.

## Release position

Not release-gated. The change is inert with taping off, so no shipped fold moves, and
`tests/test_taped_softmax_forward_config.py` pins that: the config is `None` under `no_grad`,
precise under a tape, and a caller that brought its own config keeps it. It stays on this branch
for the orchestrator to merge, as every row's work does.
