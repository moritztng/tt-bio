# of3t-wholemodel — the whole model's gradient, both repairs on, against upstream's own step

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: VERIFY/BENCHMARK + ALWAYS-ON | memories read:
of3t PROTOCOL A14/A15/A16/A18/A23/A24/A26/A26-SCOPE/A27/A28, DEFECTS D17/D32/D35/D46/D51/D72/
D76/D96/D116, state docs `of3t-apbgrad`, `of3t-f64softmax`, `of3t-trajectory`, `of3t-direct`,
`of3t-trunkg043`, `of3t-auxgrad`, `of3t-l1`, `of3t-crop640`; fleet memories
`relative-l2-alone-cannot-identify-the-error-direction`, `a-lever-can-fire-and-be-inert`,
`negative-control-must-break-what-check-reads`, `ratio-must-name-how-its-denominator-arm-was-built`,
`worst-tensor-names-the-tail-not-the-locus`, `census-key-label-is-not-an-executed-shape`,
`per-op-cost-from-dividing-phase-total-assumes-linear-scaling`.

Branch `wk/of3t-wholemodel`, cut from `origin/wk/of3t` at `487cee1d2`. Artifacts
`perf/of3t_wholemodel/`. Pre-registration committed at `dd4c5dda0` **before the first model-scope
number existed**; every branch and every threshold quoted below was fixed there.

**Nothing under `tt_bio/` moved.** `git diff origin/wk/of3t -- tt_bio/` is empty. Both repairs stay
default-off, this row only turned them on for its own arms, and nothing is merged.

VERDICT: GO — on this row's own question, stated exactly as narrowly as the evidence allows.

With `of3t-apbgrad`'s softmax-backward repair on and nothing else, OpenFold3's gradient at model
scope reads **1.006695e-01** against upstream 0.4.3's own bf16 training step over 92.1568 % of the
model's squared gradient norm, against a reachable bar of **1.049545e-01** — **0.9592x**. Adding
the pairformer trunk as a composed term takes coverage to 97.9850 % and the reading to
**1.528664e-01** against a composed bar of **1.627551e-01** — **0.9392x**. Pre-registered branch
**B2** lands: our training step's gradient is as close to the ideal as an independent
implementation as good as upstream's own bf16 recipe would be, **on the renorm arm alone, with no
host round trip**. That is an agreement statement about one step's gradient on one batch. It is
not a statement about a training run.

MODEL: one mass-weighted `rel_l2`, assembled per tensor rather than averaged over scope
headlines. 907 of the 4,170 tensors in the 0.4.3 reference (`grads_f64_043.pt`, model squared
gradient norm re-measured here at 10.279642678524981, 3.5e-16 from the campaign's published
denominator) come from device arms driven by the model's **own** batch — `batch_step003`, crop
384 — and those 907 hold **92.1568 %** of the squared gradient norm. Both references named per
A27: **upstream 0.4.3's own bf16 training step** is `of3t-refprec`'s `arm4_bf16_autocast`, fp32
parameters under `torch.autocast(bfloat16)` so LayerNorm and softmax are exempt from the cast,
sha256 `ff78d7bc…07ccb` verified before loading; **float64** is `bundle_ref/grads_f64_043.pt`.
A26's `sqrt(2)` applies against the first because that reference carries error and does not apply
against the second (A26-SCOPE).

    arm                      vs their bf16   x reachable bar   vs float64     r        cos
    shipped                  5.551840e+00        52.90x        5.637920e+00   5.8540   0.3798
    renorm                   1.006695e-01         0.9592x      8.301029e-02   0.9983   0.9949
    renorm + host f64        8.526982e-02         0.8125x      4.563835e-02   0.9833   0.9964
    break control            2.009101e+01       191.42x        2.037042e+01  20.0633  -0.0028
    zero model (A16)         9.999999997e-01      9.53x        1.000000e+00   0.0000   —

The bars, computed from this measurement and not borrowed: upstream's own bf16 is
**7.525074e-02** from float64 on this exact set, its norm ratio is **1.0139691**, so a port that
reproduced float64 exactly would read **7.421404e-02** and two independent implementations of
equal accuracy read **1.049545e-01**. The instrument floor — upstream's own fp32 arm against
float64 — is **7.608574e-05**, three to four orders below every arm, so the scorer is not the
error. All figures `perf/of3t_wholemodel/MODEL_arms.json`, per-tensor arrays in
`/home/ttuser/of3t_wholemodel/sidecar_arms/`.

ARMS: four, and each one's flag was checked for **reach** as well as for effect, because
`a-lever-can-fire-and-be-inert` has both halves and this row hit both. `armrun.py` reports
`HOST_F64_SOFTMAX_STATS` and `taped_ttnn._SOFTMAX_BW_RENORM` out of the loaded modules after every
run; `armdiff.py` then says whether the numbers moved.

  * **shipped**, no flag. Reproduces the record: `of3t-direct`'s conditioning 6.463839e-02,
    `aux_heads` 2.360143e-01 and `msa_module` 2.260014e-01 all to every published digit, and the
    trunk's 9.025172e+00 against float64 exactly. The A/A runs are bit-identical across
    worktrees and processes: this row's `msa` shipped dump against `of3t-direct`'s, 0 of 154
    tensors moved; this row's trunk renorm dump against `of3t-apbgrad`'s, 0 of 2,736.
  * **renorm**, `TT_BIO_SOFTMAX_BW_RENORM=1`. Moves the diffusion module, the trunk, `aux_heads`
    (159 of 180 tensors, 3.021575e-05 over the scope) and `msa_module` (147 of 154,
    1.346112e-01). It is **bit-identical to shipped on `diffusion_conditioning`**, 0 of 26
    tensors moved — and that section is 36.9462 % of the model, the single largest, so a third
    of the model's gradient mass is untouched by either repair and was already at 1.0414x its
    own bar before them.
  * **renorm + host float64 softmax**, `TT_BIO_HOST_F64_SOFTMAX_AB=all`. **The host round trip
    reaches 51.1358 % of the model and no more, measured rather than inferred**: it served
    **0 calls** on the pairformer trunk, on `aux_heads`, on `msa_module` and on
    `diffusion_conditioning`, with `declined` also 0, so `site_softmax` was never reached on any
    of them. `host_f64_softmax_site` is consulted at exactly four construction sites
    (`openfold3.atom_transformer`, `openfold3.diffusion_transformer`, `protenix.atom_transformer`
    and the generic `softmax_site` in `tenstorrent.py:8231`) and those scopes' attention runs a
    fused SDPA instead, which is D31 seen from the other side. Its gradient dumps there are
    bit-identical to the renorm arm's.
  * **zero-gradient baseline**, A16, run through the same scorer rather than asserted: a model
    that computes nothing reads **9.999999997e-01** against upstream's bf16 step and exactly
    1.0 against float64. The shipped arm is 5.55x **worse** than that baseline and the renorm arm
    is 9.93x better than it, which is the difference the repair makes at model scope.

GRADIENTS: per parameter, in float64, A14 applied on the float64 gradient for every pair so a
tensor's exclusion does not depend on which pair is being read. **907 tensors compared of 4,170
in the reference, 900 measurable under A14, holding 92.1568 % of the model's squared gradient
norm** — the share of the norm and not a count, because 79 % of this model's tensors hold 6.54 %
of its mass (A15/D17) and the median tensor holds 1.305e-04 % (A23). The worst tensor is named and
located. On the shipped arm it is
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`
at **1.803328e+01** against upstream's bf16 and 1.850397e+01 against float64. On both repaired
arms the worst moves out of the diffusion module to
`aux_heads.pairformer_embedding.pairformer_stack.blocks.1.attn_pair_bias.mha.linear_g.weight` at
**7.683875e+00** against upstream's bf16 and 3.661561e+00 against float64.

**The per-tensor bar is not cleared and that is the honest half of this row.** Against the
5.0e-02 per-tensor bar the renorm arm is over it on **844 of 900** measurable tensors and the
renorm + host f64 arm on 823 of 900 — but **upstream's own bf16 step is over the same bar on 791
of 900 of the same tensors**, so the count is very largely what bf16 costs anyone at this depth
and not a property of the port. The mass-weighted statistic is the one A23 makes the headline and
it is the one that clears. Against float64 the renorm + host f64 arm reads **4.563835e-02**, which
is **1.65x closer to float64 than upstream's own bf16 step is** (7.525074e-02) — A28 cutting in
our favour, and reported because A28 requires it whichever way it points.

RECONCILE: the scopes compose to the model number, and the rule the campaign has been using to
compose them is wrong by half a percent.

  * **Composed with each arm's own reference mass, the composition is an identity** and
    reproduces the assembled headline exactly: relative difference **0.0** on shipped, renorm and
    renorm+f64, and 1.77e-16 on the break arm. So the assembly and the scope readings are the
    same arithmetic and neither contains a bookkeeping error.
  * **Composed the way the campaign states it — each scope's published reading weighted by its
    share of the model's squared gradient norm — it is off by 5.453e-03 relative on the shipped
    arm**, 3.189e-03 on renorm and 2.898e-03 on renorm+f64. The reason is a denominator, not a
    defect in any scope: a reading of the form `||d|| / ||g_bf16||` composes with **bf16** mass,
    while every share in this campaign is a share of the **float64** squared norm, and the two
    differ per section by `r_s = ||g_bf16|| / ||g_f64||`, which spans 0.9409 to 1.0469 here. The
    fix is one multiplication — weight by `share_s * r_s^2` — and it is in `model_scope.py`.
  * **The pairformer trunk is refused entry to the union**, and refusing it is the point of this
    deliverable. Its arms are driven by the captured 64-token boundary `boundary_c64.pt` (56 real
    tokens of 64), not by `batch_step003` at crop 384, so concatenating its gradient with the rest
    would be concatenating gradients of two different inputs. It enters as a composed term with
    **its own** floor (3.739355e-01) and **its own** r (1.029909) — the per-section floors here
    span 3.008e-02 to 2.362e-01, a 7.9x spread, so a borrowed floor would be wrong on most of the
    model. Re-scored in this row: shipped **8.713526e+00**, renorm **4.756011e-01** against a
    trunk-local reachable bar of 5.134656e-01, **0.9263x**. With it, 97.9850 % of the model reads
    1.528664e-01 against a composed bar of 1.627551e-01.
  * **One scope moved since it was published, and the reconciliation is what surfaced it.**
    `diffusion_module.atom_attn_dec` read 2.8139e-01 in `of3t-trajectory` and reads
    **4.856455e-02** here on the shipped arm; `atom_attn_enc` 2.1925e-01 against 2.131918e-01.
    Same scope, same reference, different tree: trajectory's dump is from pass ~176 and this row
    reads `of3t-f64softmax`'s pass-212 shipped control, which is the composition. The shipped
    headline moved 7.426742e+00 to 7.426217e+00 over the same interval, so the sections moved
    while the total barely did. The current tree's numbers are the ones above.

COVERAGE: stated as mass, not as count. **92.1568 % of the model's squared gradient norm is
compared on the model's own batch** (907 tensors), **5.8282 % more is measured on another
boundary** (the trunk, 2,736 tensors), and **2.0150 % has no reading at all**. Where that 2.0150 %
sits, measured from the reference here and not quoted:

    input_embedder                            0.8007 %   0 of 98 tensors
    diffusion_module.atom_attn_enc            0.8241 %   50 of 98 uncarried
    diffusion_module.diffusion_transformer    0.2715 %   96 of 552 uncarried (the fused qkv)
    msa_module_embedder                       0.0612 %   0 of 2
    diffusion_module.atom_attn_dec            0.0330 %   42 of 82 uncarried
    template_embedder                         0.0103 %   0 of 96
    msa_module                                0.0083 %   73 of 227 uncarried
    layer_norm_z / layer_norm_s / linear_z / linear_s  0.0059 %  0 of 6

**0.74055 % of it can never be read while the code is shaped as it is**, and that was verified by
reading the live file rather than carried forward: `openfold3_host_prep.py:256` does
`ql = ttnn.to_torch(ql_d)` and `:259` a host `torch.nn.functional.linear`, so the input embedder's
atom-encoder leg is host-applied and the cotangent that would reach those weights has to return
through a graph the forward already severed. Porting the op is the remedy; no instrument can
close it.

CONTROL: three, all measured, none asserted.

  1. **A16 zero-gradient baseline**, above: 9.999999997e-01 at model scope, 9.999995590e-01 on
     the trunk. Reported as a measurement because on a normalised relative L2 it lands near 1
     only when the reference's scale is what the relative divides by, and this campaign has had
     a scope where the arms scored *worse* than it.
  2. **A break control that moves the reading, and moves it a long way.** Structure `k`'s backward
     seeded with structure `k+1`'s cotangent on the diffusion scope, the reversed cotangent on
     conditioning and the scrambled cotangent on `aux_heads` and `msa_module`: **2.009101e+01**
     against upstream's bf16, cos **-0.0028**, 900 of 900 tensors over the per-tensor bar, worst
     `msa_module.blocks.0.msa_att_row.linear_z.weight` at 428.386. That is 199.6x the renorm arm
     and 20.1x the zero model, so the scorer is reading the gradient and not the plumbing. The
     trunk's own break arm reads 8.080415e+00 against its 4.756011e-01.
  3. **The instrument floor**, upstream's own fp32 arm against float64 on the same 907 tensors:
     **7.608574e-05**. Every arm above is at least 599x it.

  And two A/A controls that had to come out bit-identical and did: this row's `msa_module` shipped
  dump against `of3t-direct`'s (0 of 154 tensors moved) and this row's trunk renorm dump against
  `of3t-apbgrad`'s (0 of 2,736), across different worktrees, different processes and a day apart.

## The one thing this row did not take, and the arithmetic for why

LIMIT: a single process that computes the whole model's gradient end to end was not attempted,
and the reason is the objective rather than the card.

**Nobody has run one process that computes this model's whole gradient end to end, and the
blocker is the objective, not the card.** LEDGER R2: our loss-weight table does not cover
OpenFold3 — `train/losses.py:63` holds two keys, `"pretrain"` and `"finetune"`, carrying
Protenix's constants, and of3 appears nowhere in `objectives.py`, `losses.py` or `catalogue.py`.
A model-level cotangent composed here would be **an invention**, and a gradient of an invented
loss is not a gradient of OpenFold3's training step. That is why every arm in this campaign,
including all of the above, is driven by upstream's **captured** cotangent at its own boundary.
No amount of hardware changes it; writing OF3's objective is a port, and it is the honest next
row.

The card is a second, independent constraint and its measured figure is the trunk's alone: a
taped trunk cycle at crop 384 peaks at **16.693 GB** backward (`of3t-crop640`, re-measured
against `of3t-l1`'s 16.748 GB, -0.33 %) on a 34.22 GB Blackhole p300c, and that excludes the
diffusion module's 48 noised structures, the confidence rollout and the heads. The whole-model
peak is unmeasured because the run is refused earlier, for the reason above, and quoting a sum of
peaks taken in different processes would be exactly the composition error this row's RECONCILE
section catches.

What the assembled number **is**, said plainly: 907 per-tensor gradients, each the gradient our
port computes for that parameter when its own scope is driven by upstream's captured boundary,
scored against the gradient upstream's own step computes for the same parameter on the same
batch. What it is **not**: the gradient of one forward pass through our whole model. The
difference between those two is error compounding across scope boundaries, and it is unmeasured
and unbounded by anything here.

## Standing

Nothing merges and no default moved. `TT_BIO_SOFTMAX_BW_RENORM` and `TT_BIO_HOST_F64_SOFTMAX_AB`
are both default-off, `git diff origin/wk/of3t -- tt_bio/` is empty, and every arm above set them
through the environment for the measurement only. The row is release-gated by construction: it
changes no code, and the two repairs it measures are already gated on their own branches.
