# of3t-softgrad — the two shippable softmax levers, on the GRADIENT, at full scope

Pre-registration `perf/of3t_softgrad/PREREGISTERED.md`, pushed as commit `c69129695` before the
first arm ran. Bands, bars and controls are that file's and none of them moved afterwards.
Branch `wk/of3t-softgrad`, artifacts in `perf/of3t_softgrad/`, card 0 on qb2.

VERDICT: NO-GO — at full scope the best shippable softmax lever reads 1.076374e-01 against
upstream's own bf16 step, which is 5.38x the 2.0e-02 mass-weighted bar and 1.38x the
non-shippable host bound. It is the pre-registered third band. This is the row's verdict on its
own question, not the campaign's.

ARMS: all five at full scope, 48 structures each, 547 tensors holding 51.1358 % of the model's
squared gradient norm, same rebuilt boundary, same card, same process shape. Every arm publishes
its softmax intercept count and a zero is a hard failure: shipped 0 (no rule installed),
softmax_precise 1,440, softmax_accurate 1,440, softmax_host_f64 1,440, ckc_on 1,440, break
control 1,440.
**Both controls reproduce to every published digit**: shipped reads 7.426217e+00 against their
step where 7.426217 was required, and softmax_host_f64 reads 7.777580e-02 where 0.0777758 was
required. They reproduce on a boundary this row had to REBUILD, because `/home/ttuser/of3t_rebase/`
was pruned with its row's worktree and took `diffcap043` with it; the rebuild matches the original
capture's forward loss 1.2675874205688995, cotangent norm 0.019426651390714835 and `vs_bundle`
worst rel 0.0, and reproduces `of3t-conditioning`'s independently published structure-0 forward
rel 1.104135e-02. Five other scripts still point at the pruned path.

CKC: **AMENDMENT 2 withdrew this arm as moot while it was already running**, on the grounds that
the `ttnn.max` repair put a compute kernel config onto the op by another route and `softmax_precise`
at 2.566369e-01 had already answered what a config does to this gradient; it is reported here as
the cheap confirmation AMENDMENT 2 asked for rather than quietly dropped, and it confirms that
reasoning exactly. The `ckc_on` arm, run at full scope with `TT_BIO_SOFTMAX_CKC=1` and a census
that changes no arithmetic. The flag IS honoured — `_SOFTMAX_CKC` read True inside the run — and
the answer is that it does not reach this scope. **Which config actually reached the op: `None`,
on 1,440 of 1,440 softmax calls.** Not `precise_config()` and not the caller's; `None` is the op's
own default, HiFi2 with math_approx on and no fp32 destination accumulation, and that is what
every softmax in the diffusion module ran under in this arm. The reason is measured rather than
read: `TT_BIO_SOFTMAX_CKC` is consumed at `tenstorrent.py:3824`, inside `_fp32_softmax_attention`,
which is the trunk's triangle-attention path, and `FP32_SOFTMAX_STATS` came back all zero — fused
0, unfused 0, calls 0 — so that helper is never entered on the diffusion scope. The diffusion
module's own two softmax sites take `self._softmax_ckc = softmax_ckc(token)`, which returns `None`
unless `TT_BIO_SOFTMAX_PRECISE_AB` names the site, and `TT_BIO_SOFTMAX_CKC` does not touch that.
**Bit-identity against shipped: 547 of 547 tensors bit-identical, rel_l2 exactly 0.0, largest
absolute difference exactly 0.0**, with `forward_rel_median` 8.474800850934073e-03 and per-tensor
median 1.250047459812202e-01 on both. So `ckc_on`'s row in the table below IS the shipped row, to
the last bit, and it reads 7.426217e+00 against their step. That is the outcome D110 exists to
make impossible to miss: a flag that fires, is honoured, and is inert. It is reported here as a
measurement rather than inferred from the source, and it is what lets the verdict say that no
device configuration route tested reaches the bar.

REACHABLE: **no** — the third pre-registered band, stated plainly as that band requires. At scope
`softmax_accurate` reads 1.076374e-01 against upstream's own bf16 step, which is **at or worse
than** the 0.0777758 host bound, so the device levers do not reach the host bound at scope and the
block-8 ladder was optimistic. On block 8 `softmax_accurate` read 0.179144, 3.58x its 5.0e-02 bar;
at scope it reads 5.38x the 2.0e-02 mass-weighted bar, and 1.8716x the 5.750945e-02 perfect-fix
threshold. What the levers DO buy is large and worth recording: 7.426217e+00 to 1.076374e-01 is
69.0x of the gap, on a lever that ships, at 1.661x the arm's cost.

BOTH_REFS: `ours_vs_their_step` is against upstream OpenFold3's own bf16 training step, verified by
digest ff78d7bc0bf7a4355014a470fbe931592bd4896fe06a8b400c161c08b5607ccb before loading (A24), and
its denominator is THEIR gradient. `ours_vs_float64` is against the bundle's float64 gradient
`grads_f64_043.pt` built from upstream 0.4.3, and its denominator is the float64 gradient. They are
different measurements and A27 forbids conflating them.

    arm                     v their step   v float64        r   cos v them   err-cos   x thresh
    shipped                 7.426217e+00  7.568637e+00   7.7814    0.411379   +0.2071   129.1304
    softmax_precise         2.566369e-01  2.742371e-01   1.0681    0.971340   +0.3250     4.4625
    softmax_accurate        1.076374e-01  8.197720e-02   0.9523    0.995112   -0.1930     1.8716
    softmax_host_f64        7.777580e-02  5.930664e-02   0.9787    0.997141   +0.0977     1.3524
    accurate + permuted cot 1.527615e+00  1.546972e+00   1.1679    0.012959   -0.1094    26.5629

Per tensor over the 5.0e-02 bar, out of 547, with the mass outside the bar (A23), against their
step / against float64: shipped 519 (97.5871 %) / 459 (94.4376 %); softmax_precise 474 (80.1320 %)
/ 295 (53.8270 %); softmax_accurate 474 (83.1242 %) / 255 (48.0179 %); softmax_host_f64 472
(85.4292 %) / 254 (52.0828 %). A26-SCOPE: against a float64 reference only one side carries error,
so the reachable bar there is the threshold itself, no sqrt(2); the sqrt(2) applies only to the
ours-vs-their-step column, where an independent bf16 error of upstream's own size reads
8.1331e-02.

GRADIENTS: 547 of 547 device tensors compared, 51.1358 % of the model's squared gradient norm, and
the worst case is LOCATED rather than averaged. `softmax_accurate` worst tensor against float64 is
`diffusion_module.diffusion_transformer.blocks.10.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at 2.817863e-01, and against their step
`diffusion_module.atom_attn_enc.atom_transformer.blocks.0.attention_pair_bias.layer_norm_a_q.layer_norm_s.weight`
at 3.970744e-01. `softmax_host_f64`'s worst against float64 is the same AdaLN s-norm gain family,
`diffusion_module.diffusion_transformer.blocks.17.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at 2.618132e-01, so the worst tensor is a property of the family and not of the lever. `shipped`'s
worst is `diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at 1.850397e+01. Medians against float64: shipped 1.250047e-01, precise 5.271298e-02, accurate
4.826816e-02, host bound 4.885781e-02 — the device lever's median is already INSIDE the host
bound's, while its mass-weighted headline is outside it, which is A23 in one line.

FORWARD: taken for every arm, and it is blind. `forward_rel_median` over the 48 structures reads
shipped 8.474801e-03, softmax_precise 6.338715e-03, softmax_accurate 6.298853e-03,
softmax_host_f64 6.462246e-03. The forward spans **1.35x** across the four arms while the gradient
spans **95.5x** (7.426217e+00 over 7.777580e-02). Block 8 read 1.56x against 52.75x, so the
blindness is confirmed at scope and is WIDER here. Worse than a narrow span: the forward does not
even ORDER the arms — `softmax_accurate` has the best forward of the four and the second-best
gradient, `softmax_host_f64` has the third-best forward and the best gradient. A forward reading
cannot rank these levers, let alone pass them.

COST: measured on the real arm, never from the micro-bench, with the AICLK sampled DURING each
arm's own window from `qbcard/cardtel.tsv` on card 0. Seconds per structure over 48 structures:
shipped 1.23 at 1300 MHz mean (min 800, max 1350, n=33); softmax_precise 1.23, **1.002x**, at 1316
MHz (n=32); softmax_host_f64 1.77, 1.441x, at 1325 MHz (n=45); softmax_accurate 2.04, **1.661x**,
at 1329 MHz (n=52). `of3t-softmax`'s per-op factors — precise 1.46x, accurate 4.71x — are one
softmax shape and do not survive contact with the scope: precise is free here and accurate costs
1.661x rather than 4.71x, because the softmax is a small share of what the 48 structures run.

FLOOR: A28, beside every row. Upstream OpenFold3's own bf16 training step on this same scope,
scored against the same float64 reference, is **5.852018e-02** with norm ratio r = 1.017575, and
441 of its own 547 tensors sit over the 5.0e-02 per-tensor bar carrying 41.6989 % of the mass.
That is what a faithful bf16 implementation reaches. It does not widen any bar. Against it,
`softmax_accurate` at 8.197720e-02 from float64 is 1.40x upstream's own bf16 error, and
`softmax_host_f64` at 5.930664e-02 is 1.01x it.

CONTROL: four, all measured. **A16 zero model** reads 1.000000e+00 against their step on every arm,
measured rather than assumed. **Instrument floor**: the two end arms reproduced their published
values to every digit on a rebuilt boundary, which bounds the drift this row could have introduced
at zero. **Upstream's own step** is the A28 floor above, 5.852018e-02. **A break control that is
shown to FAIL**, and run on a LEVER arm rather than on the saturated shipped one: seeding structure
k's backward with structure k+1's cotangent takes `softmax_accurate` from 1.076374e-01 to
1.527615e+00, 14.2x, with the cosine against their step collapsing from 0.995112 to 0.012959 and
every one of 547 tensors over the bar carrying 100.0000 % of the mass. The forward is byte-for-byte
the arm's own (6.298853e-03 in both), which is what makes it a control on the comparison rather
than on the model.

MODELS: what this row found reaches further than its own scope. The compared set is 547 of 738
diffusion-module tensors. Two defects were root-caused and both are shared code. `precise_config()`
installed with `kwargs.setdefault` cannot reach a call site that passes the argument explicitly,
and 2 of 2 softmax sites in this path do exactly that (`openfold3_diffusion_transformer.py:209`,
`openfold3_atom_transformer.py:184`), so the of3t-adaln block-8 `softmax_precise 0.349441` row was
scored through the same `setdefault` and should be re-taken before it is quoted. `_accurate_softmax`
returns NaN on a fully-masked row, and 2 of 5 shipped models take that chain by default
(protenix-v2 and OpenDDE Pairformer sites, per `accurate_softmax_site`), so the exposure question
is whether either reaches a fully-masked attention row in inference.

AMENDMENT 1's source-grounded hypothesis for the non-finite backward is **refuted, and it was
refuted by the instrument rather than by argument**. Both of its branches predicted a broken
backward PAIR: either a saved output that the 5-op chain did not produce, or five individual tape
nodes whose `divide` carries a 1/x^2 term. Neither happened. The rule creates exactly ONE tape
node, the same shape as the shipped rule, and a finiteness probe on all 30 softmax calls per
structure found the incoming cotangent, the saved output, the row-sum reduction and the emitted
input-gradient ALL finite, count 0. Wrapping every one of the 37 tape verbs then put the first
non-finite value at forward index 134 — the softmax's own FORWARD output, shape [1, 14, 4, 32, 128]
— so the backward pair was never the defect and the NaN was already in the forward activation,
carried into a `layer_norm` whose backward guards its variance with eps and so could not have made
it. The hypothesis was a good one and cost nothing; the measurement is below.

The mechanism, because it is the reusable part. `ttnn.max` rounds its result to bf16. On a row
whose keys are ALL masked at -1e9 — which the atom encoder's block-sparse attention produces for a
padded query block — the reduction comes back **1,755,648 ABOVE** the true row max, so every
exponent is -1.76e6, `exp` underflows the whole row to zero, the sum is zero and the divide is
0/0. Measured standalone in `FULLY_MASKED_ROW_OVERFLOW.json`: 114,688 non-finite entries, exactly
the masked rows, where the fused kernel on the identical input is finite. Clamping the exponent at
-60 fixes it and -88 does NOT, because exp(-88) is subnormal in fp32 and flushes to zero. Clamping
the exponent from ABOVE at 0 looks harmless and is not: `ttnn.max` also undershoots by up to
0.0625 on ordinary rows, and clipping that away cost the arm's forward 0.0062989 to 0.0156243 and
its per-tensor median 4.826816e-02 to 1.258712e-01. Both variants are kept in
`perf/of3t_softgrad/` so the difference is auditable.

GATED: nothing ships and no default moved. All five arms live on `wk/of3t-softgrad` and nothing is
merged. `softmax_host_f64` is a host computation on a tape verb, a diagnostic bound, and it is NOT
proposed as a product candidate. `softmax_precise` and the clamped `softmax_accurate` are
release-gated changes to a precision path and stay on the branch. The clamp is a repair the arm
needed to run at all; shipping it would be a separate decision with its own inference evidence,
which this row did not take.

PROVES: at full scope, over the 547 tensors holding 51.1358 % of the model's squared gradient norm,
against a float64 reference built from upstream 0.4.3 and against upstream's own bf16 step, that
the two shippable softmax levers close 69.0x of the shipped arm's gap and still land outside the
2.0e-02 mass-weighted bar, in the third pre-registered band; that the arm's forward cannot rank
them; that the cost at scope is 1.002x for precise and 1.661x for accurate at 1300-1329 MHz; and
that two named defects — a `setdefault` that cannot install a lever, and a bf16-rounded `ttnn.max`
that NaNs a fully-masked softmax row — were the reason the levers had never been scored here.

DOESNOT: this is one gradient at one boundary, and it proves nothing about stability over a full
training run. The campaign proves an update rule over N steps; it does not prove that 100k steps
of it stay on upstream's trajectory, and nothing here bounds long-run drift. The reading is one
crop (384 tokens, 56 real), one checkpoint (`of3-p2-155k`) and one card, so it does not establish
that the same ladder holds at another crop or on another board. 48.8642 % of the model's squared
gradient norm has no measurement in this row at all. The levers were scored on the GRADIENT only:
no fold, no Angstrom reading and no seed floor was taken, so this says nothing about what either
lever does to a structure a user gets. And the `softmax_accurate` number is the CLAMPED chain, not
`_accurate_softmax` as it ships today, because as it ships today it returns NaN on this scope.
