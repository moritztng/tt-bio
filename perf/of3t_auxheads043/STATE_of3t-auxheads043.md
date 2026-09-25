# of3t-auxheads043 — the aux_heads reference, rebuilt at 0.4.3

TASK TYPE: VERIFY/BENCHMARK | PLAYBOOKS loaded: VERIFY/BENCHMARK | memories read: of3t PROTOCOL
A13/A15/A16/A18+addendum/A23/A24+amendment/A25/A26/A27, DEFECTS D23/D90/D93/D94/D95/D96,
of3t-auxheads and of3t-rebase state docs, fleet memories `reference-checkpoint-version-binding`,
`a-digest-pins-the-bytes-not-the-function`, `negative-control-must-break-what-check-reads`,
`relative-l2-alone-cannot-identify-the-error-direction`.

Branch `wk/of3t-auxheads043`, artifacts `perf/of3t_auxheads043/`. pc, CPU only, no card lease and
no Tenstorrent device opened. Pre-registration committed at `a9deec3c3` before the first number
existed; every branch below was named in it.

**The headline is the one the brief did not expect.** The `aux_heads` reference was **already built
at 0.4.3**. There was nothing to rebuild, the 3.6515e-01 was never a reading against 0.5.0, and the
revision cannot explain the A18 failure because it was never in it. Measuring what the revision
*would* have done if the premise had held says the same thing a second way: its whole effect is
5.9x smaller than the gap.

REFERENCE: The tree that produced `boundary_aux_heads.pt` is pinned by digest, not by its label
(A24-AMENDMENT — a digest pins the bytes, and for a reference the bytes are the function). sha256
over the sorted sha256 of every `.py` under `openfold3/`:

    of3pkg043, the tree that captured the boundary   8f035f4eb4470472...   293 files
    openfold3-0.4.3 sdist on pc                      8f035f4eb4470472...   293 files
    openfold3-0.5.0 sdist on pc                      d9f4840ade3ba68b...   321 files

**Byte-identical to 0.4.3 across all 293 files.** `PKG-INFO` agrees (`Version: 0.4.3`) but is not
what decides it. The rebuild was then run anyway, from the pc sdist in a fresh process on a
different host from the capture: upstream's own `AuxiliaryHeadsAllAtom` on the identical captured
inputs reproduces the captured reference outputs **bit for bit on 5 of 5 heads**, relative L2
**0.0**, max abs difference **0.0** (`reference_selfcheck_043.json`). Key gate, D23's condition:
**244 tensors loaded, 0 missing, 0 unexpected** on both release trees.

Dtype policy, per A27, because "float64" names a width and not a policy: every parameter and every
activation is float64, no cast anywhere on the path, `of3-p2-155k` weights upcast once at load, 8
CPU threads, torch 2.8.0+cpu. `torch.amp.autocast(device_type="cuda", ...)` — the mechanism both of
the revisions' precision differences act through — is **inert** on CPU tensors and torch says so
("User provided device_type of 'cuda', but CUDA is not available. Disabling"). So this arm measures
the two revisions' **structure** with their precision difference switched off by the platform. That
limit was stated in the pre-registration before the run, and the bf16 policy arm below is the arm
that does not have it.

FORWARD: Our port's A18 forward at this boundary, against a 0.4.3 reference, is the published
reading, because the reference it was taken against is the 0.4.3 one:

    distogram_logits                 2.8277e-03   PASS
    pde_logits                       8.7897e-02   FAIL
    pae_logits                       1.8199e-01   FAIL
    experimentally_resolved_logits   3.3315e-01   FAIL
    plddt_logits                     3.6515e-01   FAIL

D95's masking question is answered from the boundary's own masks rather than assumed
(`padding_fraction.json`). The token axis is **56 real of 384**, 14.583 %, and carries
`distogram`, `pae` and `pde`. The atom axis is **422 real of 422**, zero padding, and carries
`plddt` and `experimentally_resolved`. **So the two heads that fail A18 worst sit on an unpadded
axis: 3.6515e-01 and 3.3315e-01 are already real-token readings** and D95's dilution mechanism
cannot touch them. For the pair heads the reference's own cotangent puts **1.0000000** of its
squared mass for `pae` and **0.9999999999999998** for `pde` inside the real 56x56 block, so the
padded region carries no gradient signal there either.

The reading against a 0.5.0 reference — the one the brief believed had been taken — is bounded
without re-running our port, by the triangle inequality on the two references' measured separation,
in units of ||0.4.3|| (`forward_triangle_bound.json`):

    head                             port vs 0.4.3   revision distance   port vs 0.5.0 bound
    distogram_logits                 2.8277e-03      0.0000e+00          [2.8277e-03, 2.8277e-03]
    pde_logits                       8.7897e-02      1.3407e-02          [7.4490e-02, 1.0130e-01]
    pae_logits                       1.8199e-01      2.4622e-02          [1.5737e-01, 2.0661e-01]
    experimentally_resolved_logits   3.3315e-01      5.8870e-02          [2.7428e-01, 3.9202e-01]
    plddt_logits                     3.6515e-01      6.1927e-02          [3.0322e-01, 4.2708e-01]

**Every lower bound is above the 5.0e-02 bar.** Four of five heads fail A18 against either
reference, and the ratio of the port's gap to the whole revision distance is 5.66x to 7.39x.

The revision distance is reported with its direction, never as a relative L2 alone — norm ratio
`r = ||0.4.3||/||0.5.0||` and the cosine between the two outputs:

    plddt_logits                     rel 6.1759e-02   r 0.997294   cos 0.998091
    experimentally_resolved_logits   rel 5.7554e-02   r 0.977641   cos 0.998562
    pae_logits                       rel 2.4718e-02   r 1.003905   cos 0.999703   real 56x56 7.5429e-02
    pde_logits                       rel 1.3442e-02   r 1.002617   cos 0.999913   real 56x56 5.0492e-02
    distogram_logits                 rel 0.0000e+00   r 1.000000   cos 1.000000   bit-identical

MODULES: which of the two modules' differences fires on this input, measured by running both
release trees on the identical captured inputs, one tree per process, and forcing each knob.

**Neither of the two differences the brief names fires here. A third one does.** The
`prediction_heads.py` autocast pair is inert on CPU by construction. The `head_modules.py`
`single_mask` change reads `repr_mask_sum` **56.0**, `token_mask_sum` **56.0**, **0 differing
tokens**, so forcing the convention either way leaves the trees bit-identical on 5 of 5 heads. What
is actually live is `transpose_bias=True`, added in 0.5.0 at `base_blocks.py:395` on the pairformer
block's end-node triangle attention, where `triangular_attention.py` then permutes the pair bias
`(2, 1, 0)` instead of `(2, 0, 1)`. That is D90's trunk flag, reaching the confidence Pairformer.

CONTROL: Three, and each one carries its number.

**(i) The knob-forcing control the brief requires reads exactly 0.0.** 0.5.0 forced back to
`transpose_bias=False` against 0.4.3 as shipped: **5 of 5 heads bit-identical**, relative L2
**0.0**, max abs difference **0.0** (`arm2_control_tbias_off.json`). So `transpose_bias` is the
*only* live difference between the revisions on this path in float64, and the arm is not void.

**(ii) The vacuous control, and the one that breaks.** Forcing `single_mask` to 0.4.3's
`repr_x_mask` against 0.5.0's `token_mask` reads 5 of 5 bit-identical — because the masks are
equal here, that control cannot break anything, and a control that cannot break what the check
reads is not evidence. So four representative atoms were marked unresolved, making the two
conventions genuinely differ (repr 52.0 against token 56.0, 4 differing tokens), and the arms then
**diverge**: `plddt_logits` **3.7424e-02**, `experimentally_resolved_logits` **2.7665e-02**, 2 of 5
heads moved. The instrument can see the `head_modules` difference; on this batch there is nothing
to see.

**(iii) The A16 zero-model baseline** for this scope, measured rather than assumed by
`of3t-auxheads` at the same boundary: **exactly 1.000000e+00 on every tensor**, 108 of 108 over the
bar, 0.00000000 % of the mass inside it — a **439x** separation from the baseline's 2.277568e-03.

The autocast pair was then sized where it *is* live, by writing the dtype policy out by hand rather
than by setting a flag CPU ignores — D93's pass-176 method, that a flag selecting a dtype policy
can have its policy written out. Base bf16, both policies on the **0.4.3** tree so `transpose_bias`
is held fixed, control first: the policy machinery selecting the native policy reads **5 of 5
bit-identical**, so the machinery is inert when it should be.

    0.4.3's policy vs 0.5.0's policy, bf16   worst 1.7908e-02 (exp_resolved), plddt 1.5908e-02,
                                             pae 1.0566e-02, pde 5.0436e-03, distogram 0.0
    0.4.3's policy vs float64                plddt 2.3664e-02   exp_resolved 2.1101e-02
    0.5.0's policy vs float64                plddt 3.5432e-02   exp_resolved 3.2485e-02

Pre-registered branch **S2**: the precision difference is **under the 5.0e-02 bar on every head**
even in the regime where it is live. Worth recording beside it: 0.4.3's placement is **1.50x closer
to float64** on `plddt` than 0.5.0's, so the revision that removed it moved away from the reference,
not toward it.

GRADIENT: The forward does not agree, so A18's gate does not open and this row takes no new
gradient. The mass-weighted headline for the **2.843136 %** of the model's squared gradient norm
this scope holds is `of3t-auxheads`' **2.2996e-03**, and its footing is unchanged by everything
above: it is carried by `aux_heads.distogram.linear.weight`, **100.0000 %** of the section on its
own at **2.267842e-03**, and that is the one head whose forward **passes** at 2.8277e-03 and whose
reference value the finite-difference axis validated directly at 1.387e-05. `distogram_logits` is
also the one head the revision moves by **0.0000e+00**, bit-identical between 0.4.3 and 0.5.0, so no
part of that headline was ever exposed to the revision question. The other 243 tensors, holding
1.3e-07 of the mass, stay void.

GRADIENTS: Per-parameter, as published and re-read here against the rebuilt reference's own scope.
**176 of 176** taped parameters scored against the float64 0.4.3 reference, 0 skipped, 4 excluded
under A14. The compared set holds 99.9999943 % of `aux_heads`' own squared gradient norm, which is
**2.843136 % of the model's squared gradient norm**. Mass-weighted **2.2996e-03**; median
**1.0482e+00** over 176 tensors, against a measured zero-model baseline of exactly 1.000000e+00, so
the median carries no information by construction and is an instrument reading only. **Worst tensor
3.7941e+00, `aux_heads.pairformer_embedding.pairformer_stack.blocks.1.attn_pair_bias.mha.linear_g.weight`**;
171 of 176 over the 5.0e-02 per-tensor bar, together holding 1.35e-07 of the compared mass.

MODELS: **4 of 5** heads move under the revision and **1 of 5**, `distogram_logits`, is bit-identical
under it — which is the head that carries the section's entire mass. **0 of 5** heads are moved by
either of the two differences D94 named, in float64, on this input. **5 of 5** aux_heads outputs were
scored in every arm, and 2 of 5 sit on an axis with no padding at all. The live difference,
`transpose_bias`, is a property of the shared pairformer block rather than of OpenFold3's confidence
head, so it reaches every model in the fleet with a pairformer stack; `of3t-pairbias` and
`of3t-foldab` own that orientation question and this row does not restate their numbers.

VOID_OR_NOT: **`aux_heads` stays A18-void.** This is the unfavourable answer and it is the one the
measurements give. The scope's forward fails at 3.6515e-01 against a reference that was already at
0.4.3; rebuilding it at 0.4.3 reproduces it bit for bit, so the number does not move; and against a
0.5.0 reference the triangle bound puts it no lower than 3.0322e-01, still 6.1x over the bar. The
revision is refuted as an explanation from both directions. What survives void is the part that
never depended on the failing heads: the **2.2996e-03** mass-weighted headline over 2.843136 % of
the model, carried by the one tensor whose forward passes and whose head the revision does not move
at all. The remainder of the scope, 1.3e-07 of the model's mass over 243 tensors, is void and the
reason is our port, localised by `of3t-auxheads` to monotone per-block accumulation of ~1.7e-02 in
the confidence Pairformer's s-track, not the revision.

PROVES: That the reference `of3t-auxheads` measured against is upstream 0.4.3, by a whole-tree
digest over 293 files and by a fresh-host float64 rebuild that reproduces the captured outputs bit
for bit on 5 of 5 heads. That the entire live difference between 0.4.3 and 0.5.0 on the aux_heads
path in float64 is `transpose_bias`, proven by a forcing control that reads exactly 0.0 on all five
heads. That neither difference D94 named fires on this input — the autocast pair because CPU
disables it, the `single_mask` change because 0 of 56 tokens differ — with a break control showing
the instrument would have seen the second one at 3.7424e-02 had it fired. That the revision's whole
effect, structure and precision together, is 5.66x to 7.39x smaller than the port's own A18 gap, so
it cannot explain it under any assignment of the reference's revision.

DOESNOT: This proves the update rule's inputs at one boundary on one batch and says nothing about
stability over a full run: it is one step of one (stage, dataset), `batch_step003` / 5nw3 at crop
384, and no claim here extends to drift over the 100k-step schedule OpenFold3 actually trains for,
where an error this size may cancel or may compound over a long trajectory. It does not re-run our
port: this row holds no card lease, so no new device tensor was produced and the 3.6515e-01 is
carried in as an input rather than re-measured. The D95 refinement that would still be worth having
is a masked re-read of the port's own pair-head tensors, which needs those device tensors and is
named here rather than estimated. The bf16 policy arm emulates upstream's dtype policy by hand on
CPU; it is a faithful statement about the arithmetic and not about what a CUDA autocast context
does at every op. And an agreeing forward would have cleared nothing anyway (A18 addendum, D9): it
is a necessary condition, never a sufficient one.

VERDICT: NO-GO — the revision hypothesis is refuted. The reference was already 0.4.3, the rebuild
reproduces it bit for bit, and `aux_heads` leaves this row A18-void with 2.8431 % of the model's
squared gradient norm resting on the single tensor whose forward passes.
