# What the OF3T campaign found in shipped code

The campaign was chartered to prove that our OpenFold3 training step is upstream's. It has not
reached that claim yet. What it *has* produced is a list of real defects, several of them in code
that runs today and several affecting models other than OpenFold3.

**Everything below is fixed on `wk/of3t` and unmerged, or unfixed.** Nothing here has reached
`main`. `wk/of3t` is the single reviewable branch and it goes to Moritz's gate, not past it.

Maintained by `of3t-orchestrator`. Every entry names the row that found it and how it was
measured, because an unattributed defect list is a rumour.

---

## Affects inference that users get today

### MECHANISM MAP — three UNFIXED defects sit in the same pair track and no entry named the others

Added pass 83 because the record described them separately for forty passes and a reader (me,
twice) cannot assemble them from three entries that do not cross-reference. **Co-location is
stated as fact; a common cause is NOT asserted** — it is the open question, and this campaign
has been wrong twice recently by promoting a plausible shape to a finding.

| defect | what was measured | where |
|---|---|---|
| **D19** | forward masked-z error **7.811e-03** per pairformer block, composing **near-linearly** to **2.792e-01** over 48 blocks — five times the `sqrt(48)` rounding predicts, so **correlated**, i.e. systematic | pairformer block, forward |
| **D8** | assembled block's pair-track **gradients** 4.3e-01 to 1.4e+00 while every sub-module passes alone (0.0092–0.0172); graded by attention/pair involvement — `tri_att_end` 0.3838, `attn_pair_bias` 0.1470, `tri_att_start` 0.0865, and `single_transition`, the one sub-module with no attention and no pair coupling, is the **only passer** at 0.0212 | pairformer block, gradient |
| **D9** | `fp32_softmax` alone moves the triangle-attention weight gradient **3.2x** while the forward moves **12 %** | triangle attention, gradient-only |

**PASS 89 — D23 SUPERSEDES THE CAUSE OF D19 IN THIS MAP.** The reference bundle is upstream
0.5.0 loading the preview2 checkpoint, a combination upstream's own registry marks unsupported.
Two code changes separate the revisions: `transpose_bias` on the trunk's ending node, and a
`layer_norm_z` that 0.5.0's new `DiffusionAttentionPairBias` no longer has. D19 is the first of
those measured against a reference that cannot run these weights. **D8/D9 are NOT explained by
it** — they are gradient-scope and the trunk change is a forward one, so this map's co-location
claim stands for them. See D23.

**What follows, and it is load-bearing for the verdict.** D9 proves a class of error here that a
forward comparison **structurally cannot see**. Therefore **closing D19 would not make instrument
A pass**: a forward fix cannot reach a gradient-only defect, and D8's grading by attention
involvement says the gradient side has its own contribution. The two must be closed
**separately**, and any plan that treats the forward gap as the single blocker is mis-scoped.

**What is NOT established.** That D19 and D8 share a cause. They are different quantities
(forward activation vs weight gradient) measured at different scopes, and pass 47 explicitly
bounded D9 out as the explanation for D8 — projected onto block 0 at D9's own factors the block
median moves only 0.07813 → 0.07024 with 32 of 52 tensors still over bar. Three defects in one
pair track is a lead, not a mechanism.

### D1. OpenFold3's trunk pair bias arrives at 20 % of its intended value. Fix WRITTEN and MEASURED; must not ship alone (blocked by D10). Release-gated.

`openfold3_trunk.py:139` builds all 48 pairformer blocks with `scale_pair_bias=False`. The
token-level `AttentionPairBias` computes `(q@k^T + z) * head_dim**-0.5` **inline**, so the bias
is scaled along with the scores and must arrive pre-baked by `sqrt(head_dim)`. It does not, so
it lands at **1/sqrt(24) = 0.204** of the reference value in every OF3 fold we serve.

Found by `of3t-confidence`, by sweeping the coefficient `c` in `softmax(qk/sqrt(24) + c*bias)`
per block against the device output with real weights: `c = 1/sqrt(24)` fits all four blocks at
**6.9e-03 to 1.7e-02** while the reference `c = 1` is off by 9.1e-02 to 3.3e-01. `compute_bias`
itself is exact, so the z-projection was never the cause. On block 0 against the real golden the
update reads **2.5873e-02** with the current setting against **1.5725e-02** with the bias
correct.

**Why it is not simply flipped, established by `of3t-orchestrator`:** one `scale_pair_bias` flag
is forwarded to all three attentions in `PairformerLayer`, and the two classes need **opposite**
values for OF3. For `TriangleAttention` `False` is *correct* and was established by measurement
(the folded form makes the OF3 softmax ~5.7x too peaky, 0.892 against 1.000000 versus the
reference golden); for `AttentionPairBias` it is *wrong*. **Flipping the single flag fixes one
track and breaks the other.** The fix is to split the flag.

**PASS 92: the default was flipped ON in the composition anyway, and is now held at False with a
guard.** `of3t-pairbias`'s commit `bb59a9730` landed the split **and** the flip, against its own
verdict ("Land the MECHANISM. Do NOT flip the OF3 trunk default on this evidence"), so
`wk/of3t` — the single branch that goes to Moritz's merge gate — carried `scale_pair_bias=True`
while `origin/main` carries `False`. Merged as it stood it would have served the 1.245 A rank-0
regression to every OpenFold3 user. Found by `of3t-confhead` reading the branch rather than the
verdict, verified against git by the orchestrator, and reverted on `wk/of3t-orchestrator`, which
the compose merges last. **The split stays** — it is correct, unified, byte-identical on the other
four models, and `of3t-confhead` needs it to drive its arms. `compose_verify.sh` now asserts the
composed tree carries `scale_pair_bias=False, tri_att_scale_pair_bias=False` and exits 1 naming
the line otherwise; shown failing on the pairbias tree. When D1+D10 are approved, the assertion
changes in the same commit as the flip.

**Status: MEASURED end to end, and the answer is that the fix must NOT ship alone.**
`of3t-pairbias` split the flag (`tri_att_scale_pair_bias`, default-preserving) and folded 1UBQ
through the production CLI with a searched MSA, 5 diffusion samples, **nine seeds per arm**,
shipped arm from a detached `origin/wk/of3t` checkout **on the same p300c at AICLK 1350 sampled
during the fold**:

| | shipped | split |
|---|---|---|
| rank 0 vs experimental | **0.782 A** | **1.245 A** |
| best of five vs experimental | **0.679 A** | **0.629 A** |
| lever, arm vs arm, matched seed | **1.204 A** mean (0.439-1.626) | |
| **seed floor**, this target, 36 pairs | **0.324 A** mean (0.011-0.734) | |

**The lever is 3.7x the seed floor, so it is real.** And it splits two ways that must not be
averaged: **the corrected trunk samples better and serves worse**. Best-of-five improves
0.679 -> 0.629 while rank 0 degrades 0.782 -> 1.245, because the confidence head mis-ranks —
see **D10**. Shipping this fix on its own would make served structures worse even though the
model got better.

**One half of the original diagnosis is withdrawn.** The token-side claim holds:
`AttentionPairBias` at `False` reads 5.7577e-02 / PCC 0.998522 against a float64 transcription
of OF3's own formula, at `True` 1.7795e-02 / PCC 0.999977 — a 3.2x error reduction, `True` is
right. But `TriangleAttention` reads **3.4222e-02 at False against 3.4287e-02 at True — the flag
is inert there**, because `bias_scale_inv = 1.0/self._bias_scale` (`tenstorrent.py:7658`) and
`ttnn.multiply(bias, self.scale/self._bias_scale)` (7647, 7683) divide out what the load-time
bake multiplied in, leaving only bf16 rounding. Reverting the confidence head's tri override
moves the served structure by **0.000 A on five seeds**. The claim that the two classes needed
*opposite* values was half wrong; the split is still right engineering, because it is what let
the token side take `True` alone.

Other models unaffected and the control proves it can fail: Boltz-2, BoltzGen, Protenix-v2 and
AF2 each from its own checkpoint are **byte-identical**, while `--tri 0` changes all three
Pairformer digests.

**PASS 90: D1's convention survives the D23 audit, and it needed checking.** D23 showed the
campaign's reference ran the preview2 checkpoint on 0.5.0 and that two conventions differ between
the revisions, so a fix whose reference transcription came from the wrong one would have been the
same mistake in the other direction — and D1 changes every OF3 fold users get. It is clean: the
computational core of `primitives/attention.py::_attention` is identical in 0.4.3 and 0.5.0.
`q` is pre-divided by `sqrt(c_hidden)` by the caller (`attention.py:314` in 0.4.3, `:321` in
0.5.0), then `scores += b` adds the biases **unscaled to already-scaled scores**, then softmax.
The only difference between the revisions is where a dtype cast sits, inert in float64. **So the
bias must arrive pre-baked by `sqrt(head_dim)` under both revisions and D1's diagnosis is
revision-stable.** Owner `of3t-confhead`, dispatched pass 90 — D1 and D10 ship together or not at
all.

### D25. Replaying upstream against upstream across boxes, at one revision, reads median 6.224e-02 with 151 of 171 tensors over the bar. It is NOT a floor on our arms — it shares a term with them. **RECORDED** — a measurement that refutes a proposed floor; nothing further is owed. ('MEASURED' is not in the status vocabulary; corrected pass 324.)

Raised by `of3t-orchestrator` pass 93 and **substantially corrected by `of3t-rebase` the same
night**; the correction is the useful part and is recorded first.

**The measurement.** `perf/of3t_gradients/replay_vs_r0.json`: a qb2-CPU r = 0 replay against the
republished r = 0 tape produced on a rented A100 — **upstream against upstream**, same revision,
same batch `batch_step003.pt`, same replayed `draws_recycles0.pt`, dropout disabled both sides.
Median **6.224e-02** over 171 tensors, **151 over the 5.0e-02 bar**; per block 5.707e-02 /
6.198e-02 / 7.379e-02. **No model of ours is in it.** That much stands.

**What does NOT follow, and pass 93 asserted it anyway.** I wrote that this "bounds every
comparison taken against it" and that the `tb-off` arm reading 1.148e-02 — below 5.665e-02 — was
an inconsistency needing explanation. Both are wrong. `of3t-rebase` tested the two explanations I
offered and refuted each, with arithmetic over artifacts already in the branch
(`perf/of3t_rebase/box_floor_a14.py`, `box_floor_a14.json`):

| block | floor as published | floor, A14 applied | floor on the arm's own 52 | `tb-off` arm |
|---|---|---|---|---|
| 0 | 5.707e-02 (50/57) | 5.698e-02 (49/56) | 5.665e-02 (45/52) | **1.148e-02** (7/52) |
| 23 | 6.198e-02 (52/57) | 6.190e-02 (51/56) | 6.182e-02 (47/52) | **1.743e-02** (12/52) |
| 47 | 7.379e-02 (49/57) | 7.323e-02 (48/56) | 7.323e-02 (44/52) | **4.004e-01** (52/52) |

A14 is not it: it removes exactly three tensors, one per block, all
`attn_pair_bias.layer_norm_z.bias` at reference norms 1.089e-19 / 4.181e-19 / 1.932e-17. That
collapses the **worst** case — 2.2439 to 0.1744 at block 0 — and moves the **median** by nothing,
6.2244e-02 to 6.2147e-02 over all 171. The tensor set is not it either: matched 52 to 52, none
absent either side, the gap stands.

**The reason is that the two contrasts share a subtrahend.** The floor is
`upstream_CPU − reference_A100`; the arm is `ours − reference_A100`. Two differences against the
same reference do not order each other — the triangle inequality bounds `|arm − floor|` by the
distance between our gradient and upstream's, and says nothing about `arm ≥ floor`. The intuition
that "a comparison cannot be tighter than the floor under it" holds for an *independent* noise
floor and not for this. **So the published box figure is not a floor on these arms at all, and
D25's original framing is withdrawn.**

**And it inverts what the rebuild does.** Building BUNDLE-MIN-043 on qb2 CPU — the host the
capture runs on — does not *add* a box confound, it *removes* one: the published arms were
ours against upstream-on-A100, the corrected arms are ours against upstream-on-qb2. The
0.5.0-on-qb2-CPU control still earns its time, because it is what quantifies the A100-to-CPU term
at fixed revision so the revision effect can be stated alone.

**What survives, and it is the part that matters for D19.** D19 records a diffusion-scope figure
of **6.444e-02** from *"their model against their own bundle entries"* — the same kind of
measurement as this one, upstream against upstream — and attributes it to our forward gap, then
concludes *"until D19 closes nothing compared against the bundle at this scope can read below
6.4e-02."* **That is wrong twice.** A replay of upstream against upstream cannot contain our
forward gap, so the attribution is wrong; and a shared-subtrahend contrast is not a bound, so the
conclusion does not follow even if the attribution had been right. D19's floor claim should be
read as a measurement of cross-box reproducibility and not as a limit on anything.

**Provenance note, the reusable part.** `of3t-rebase` first sized its box term from
`capture_trunk_boundary_nodropout.json`, quoting worst 2.256 / 1.971 / 1.563 and forward
1.686e-02. That file carries `reference_standing.standing = HISTORICAL` and a `superseded_by`
field: its `capture_vs_bundle` block scores against the **D18-withdrawn** train-mode tape, whose
median of 1.096 is inside noise of the zero-model answer of 1.0. The current figures are 6.224e-02
and 6.735e-03. *The artifact said it was superseded and was read past* — which is why superseded
fields should be nulled or moved rather than labelled.

### D24. On a single chain, OpenFold3 ranks its samples with a rule two of whose four terms are identically zero — and it is the only model of five that leaves it unhandled. UNFIXED. Affects every monomer fold shipped today.

Found by `of3t-confhead` pass 1, machine-checked in `perf/of3t_confhead/rank_rule.py`. Independent
of D1: it is the shipped selector, on the shipped default.

**PASS 95 CORRECTION, by `of3t-confhead` against its own hypothesis: on ubiquitin the rule reduces
one step FURTHER, to `0.2*pTM` and nothing else.** `disorder` reads **0.0 on all five samples** —
a compact 76-residue fold never pushes a 25-residue smoothed RASA window past the 0.581 threshold
— so every `rank_score` in the captured fold is `0.2*ptm` to the last digit. **The disorder
mechanism below is therefore not what moves these picks**, and a fix aimed at the RASA term would
have measured nothing on this target.

Scope that correction precisely rather than over-applying it: `disorder = 0` is a property of a
compact 76-residue monomer, not of monomers. On a larger or genuinely disordered single chain the
term is non-zero and its 2.5x leverage over the only confidence term applies as written. What is
refuted is *disorder as the explanation for D10 on 1UBQ*, not the weighting analysis.

**And it makes the obvious fix provably inert.** With `iptm = 0` and `disorder = 0`, the
`shipped`, `family`, `no_disorder` and `ptm` rules all reduce to a positive multiple of pTM, and a
positive multiple cannot change an ordering — `rules.py` puts all four on **identical served
RMSDs, sample for sample**, which needs no further seeds to believe. **So giving OpenFold3 the
ipTM→pTM fallback the other four models carry changes nothing a single-chain user is served.** It
remains the right consistency change for complexes; it is not the D10 fix.

**Which relocates the fix.** The selector rests entirely on pTM, whose spread across the five
samples is **0.010953** while their Cα-RMSD spreads **0.58 A** — it is ranking a 0.58 A structural
difference with a signal that moves by one part in a hundred. The head also emits **pLDDT, PAE,
PDE and experimentally-resolved, and the rule reads none of them**. On the three `fix`-arm folds
captured so far `gpde`, `plddt`, `pae` and `boltz` each avoid the 1.60 A sample entirely and serve
0.803–0.830 A where pTM serves 0.966 A. The row states plainly that three runs is not a result and
the nine-seed table decides.

**The reproduction control under all of this is unusually strong and worth recording on its own.**
One fold, arm `fix`, seed 1, qb2 p300c card 0, **AICLK 1350 MHz with 34 reads during the fold,
min = max = median = 1350**, 371 s wall: its five ranked Cα-RMSDs reproduce `of3t-pairbias`'s
published `fix_s1` row to a **worst difference of 3.4e-08 A** — on a different card of the same
class, through a different driver, with the D1 arm applied by patching the trunk `Pairformer`
rather than by a second checkout. One comparison confirming the lever, the RMSD computation, the
ranked order and card-class reproducibility together.

`openfold3_fold.py:277` selects the returned sample with AF3 SI 5.9.3's full-complex metric,

```
0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash
```

which is faithfully upstream's (`openfold3 0.5.0 core/metrics/sample_ranking.py`). **On a single
chain two of those four terms are identically zero by construction**: ipTM averages over
cross-chain pairs that do not exist, and `has_clash` is an inter-chain indicator. Verified against
a verbatim transcription of upstream's own `compute_ptm` rather than asserted — one chain gives
ipTM **0.000000** in both implementations, two chains give **0.154491** in both, so the collapse is
the rule and not a broken call. (Our `_ptm_iptm` and theirs also agree on pTM to six decimals,
0.150017, which is a free parity reading.)

**So 0.8 of the weight budget is identically zero and what actually ranks a monomer's samples is
`0.2*ptm + 0.5*disorder`** — in which the RASA disorder term carries **2.5x the weight of the only
confidence term, with a positive sign**. A rival sample needs pTM higher by 0.125 to outrank a
sample 0.05 more disordered. That is a concrete reason for a selector to prefer the looser mode and
it predicts the sign D10 observes.

**PASS 103 CORRECTION, from reading the four sites in source: OpenFold3 is alone in the CODE and
is NOT alone in the resulting ORDER, and the actual outlier is Boltz-2.** Monomer ordering implied
by each site with `iptm = 0` or `None`:

| site | monomer score reduces to | orders by |
|---|---|---|
| `openfold3_fold.py:277` | `0.2*ptm + 0.5*disorder`; disorder = 0 on 1UBQ → `0.2*ptm` | **pTM** |
| `rf3/confidence.py:108` | `iptm_v := ptm_v`, so `1.0*ptm − 100*clash` | **pTM** |
| `worker.py:1065` | `iptm ≤ 0 → ptm` (pLDDT only if `ptm == 0`) | **pTM** |
| `boltz2.py:6238` | `4*complex_plddt + ptm` | **a pLDDT blend** |

**Three of the four order monomers identically, and OpenFold3 is one of the three** — `0.2*ptm`,
`1.0*ptm` and `ptm` are positive multiples of one another. That is the same algebra `of3t-confhead`
used to prove the family fallback inert, carried one step further: it is inert across rf3 and
protenix too, so adopting it would move OpenFold3 from one pTM ordering to another.

**Boltz-2 is the outlier and it is the only site that reads pLDDT at all** — which matters because
pLDDT is the better signal on this evidence, serving **0.709 A** against `of3_fix`'s 0.755 A and
shipped's 0.775 A on the same samples. So the chosen fix is **adopting Boltz-2's shape and
departing from rf3 and protenix**, which is the right direction on the measurement and is not
"bringing OpenFold3 into line with the family". Consistency with the family is the one
justification the source does not support.

**And UNIFIED-NEVER-PER-MODEL is not satisfied by making a fourth copy agree with three others
when the four are four different formulas.** If pLDDT ranks better, the unified answer is one
shared ranking function all four call, and rf3, protenix and of3 are all currently on the worse
side of the campaign's own measurement. That is a larger change than `of3t-confhead` should make
unasked; it is recorded here as the recommendation the evidence supports and as the thing that row
is deliberately not doing.

**OpenFold3 is alone in the family in leaving the degeneracy unhandled**, checked against the four
source sites:

| model | monomer handling | site |
|---|---|---|
| **openfold3 / openbind** | **none — 0.8 of the weight budget is identically zero** | `openfold3_fold.py:277` |
| boltz-2 / boltzgen | substitutes pTM for ipTM | `boltz2.py:6238` |
| protenix-v2 / opendde | substitutes pTM, then pLDDT | `worker.py:1065` |
| rf3 | substitutes pTM | `rf3/confidence.py:108` |

**What is NOT established**, and the row says so: that disorder is what flips these particular
picks. The previous row stored only per-run aggregates, never per-sample ptm/disorder/pLDDT, so the
term that moves each decision is unmeasured. `perf/of3t_confhead/rank_fold.py` is written for it
and needs a card.

**The obvious fix is not sufficient, and that is arithmetic rather than opinion.** With the family
ipTM→pTM fallback the monomer rule becomes `1.0*ptm + 0.5*disorder`, so the pTM edge a better
sample needs falls from **2.5x** the disorder gap to **0.5x** — a 5x reduction, not an elimination.
`rules.py --selftest` constructs a 0.05 pTM gap against a 0.30 disorder gap where **both** rules
still serve the 1.6 A sample over the 0.5 A one, while the seven candidate rules that do not reward
RASA serve the 0.5 A one. Nine candidate rules are committed **before** any per-sample number
exists, because a rule chosen after seeing which one wins on nine seeds of one target is a rule
fitted to nine seeds of one target.

Scope if a selection-rule change lands: `openfold3_fold.py:277` is **openfold3 and openbind**, so
it reaches 2 of the 5 named models by construction and the other 3 must be shown byte-identical
with a control that moves them. Owner `of3t-confhead`.

### D10. The confidence head mis-ranks diffusion samples, and it is what makes D1's fix serve worse. UNFIXED.

**PASS 101 — MEASURED END TO END BY `of3t-confhead`. D1 DOES NOT SHIP; D10 SHIPS AS A CORRECTNESS
FIX WITH NO ACCURACY CLAIM.** 1UBQ, production CLI, the same searched MSA `of3t-pairbias` used, 5
samples, 200 sampling steps, arms interleaved, p300c, AICLK 1350 sampled during. Four arms from
two fold campaigns, because a selection-rule change does not move the diffusion samples — six
seeds with both arms complete, three pairs still folding:

**FINAL, superseding the six-seed interim first recorded here** — nine ship-arm seeds, eight
fix-arm seeds, seed floor over **28 pairs**, read from `perf/of3t_confhead/analyze.json` on
`wk/of3t-confhead` at `5588d889a` rather than from the row's prose:

| arm | rank 0 (served) | best of 5 | seed floor, 28 pairs | picks best |
|---|---|---|---|---|
| shipped | **0.775 A** | 0.679 A | **0.226 A** | 1 |
| D1 alone | **1.201 A** | 0.616 A | 1.133 A | 3 |
| D10 alone | **0.760 A** | 0.679 A | 0.282 A | 1 |
| D1 + D10 | **0.924 A** | 0.616 A | 0.671 A | **0** |

The D1+D10 gap is **0.92369 − 0.77507 = 0.149 A**, not the 0.170 A of the six-seed interim. Every
number here moved and the sign did not.

**A nuance the `picks best` column carries and the means hide:** the repaired rule picks the
single best sample **0 times** on the D1 arm where the shipped rule picks it 3 times — and still
serves **0.28 A better on average**. It trades *picking the best* for *avoiding the worst mode*,
which is the right trade on a bimodal distribution and is why mean-served improves while
picks-best falls.

**D1 does not ship, and fixing the selector did not rescue it.** D1+D10 serves **0.149 A worse**
than shipped; the best rule in the whole candidate set, `gpde`, still serves 0.861 A, 0.086 A
worse. The corrected trunk continues to **sample better and serve worse** — best-of-5 0.634 A
against 0.663 A. The bar is "the served structure, rank 0, at or better than shipped", and it is
not met.

**The row states, correctly and unprompted, that both gaps are smaller than the shipped arm's own
0.275 A seed floor**, so neither the regression nor D10's 0.020–0.066 A improvement is separable
from re-running with another seed. D10 therefore ships as what it is — a rule that gave 0.8 of its
weight to a term that is identically zero (D24), now fixed, with the served structure no worse —
and **not** as a measured accuracy win. `plddt` alone would have served 0.709 A, better than the
chosen `of3_fix` at 0.755 A, and was not picked: a rule chosen for the best number on one target
is a rule fitted to one target, which is why the nine-rule candidate set was committed before any
number existed.

**Sign stability is reported rather than a single number**: the D1+D10 gap reads 0.062 A at four
seeds and 0.170 A at six, same sign throughout. Six seeds, one target, one checkpoint.

**Provenance, recorded rather than buried**: folds ran on qb2 outside the fleet lease system for
that host (chip 0 seeds 1–5, chip 1 seeds 7–9 plus a `fix` seed-3 re-run, chip 2 seed 6 after
`c14-land-tail` released the 2/3 board pair). One fold was refused when `of3t-rebase` opened chip
0 inside tt_bio's 120 s device-lease wait and was re-run; one wedged at 0 % CPU before building
the trunk and was killed by explicit pid and restarted. Load 39–43 on 16 cores throughout, both
campaign shells and all descendants at **nice 15 in one sweep so both arms carry equal priority**,
and no wall-clock trend is claimed. The seed-1 control is what makes cross-chip reporting safe:
the same seed on a different chip of the same class reproduces the published structure to
**3.4e-08 A**.


**PASS 92 CORRECTION TO THE PASS-90 REFRAME BELOW, by `of3t-confhead`, and it is right.** A
five-sample ordering has two one-sided marginals and they disagree here. Pass 90 quoted only the
first and overstated it.

| | ship | D1 fix | random |
|---|---|---|---|
| picks the best sample | 1/9 | 3/9 | 1.8/9 |
| **predicted** rank of the truly best sample | 3.56 ± 0.44 | 2.22 ± 0.32 | 3.00 |
| **true** rank of the sample the head picked | 2.67 ± 0.41 | 3.33 ± 0.60 | 3.00 |
| Spearman rho, predicted vs true | 0.178 ± 0.173 | 0.267 ± 0.155 | 0.000 |

Quote the first marginal and the head looks better with the fix; quote the second and it looks
worse. The symmetric statistic settles it: **fix minus ship is +0.089 ± 0.232 Spearman**, which
separates neither arm from the other nor convincingly from chance. **Ordering quality is
statistically unchanged**, so pass 90's "the head ranks better with the fix" is withdrawn — the
half it got right, and which stands, is that this is *not* a calibration regression to undo.

**What the row established instead is sharper and is a decision, not a distribution.** The
corrected trunk's samples go bimodal: pooled over 45 samples the fix arm spans 0.560–1.670 A with
its largest gap at 1.362 → 1.592 A, and the mode above that gap holds **10 of 45 samples (22 %)**.
The head serves rank 0 **from that worst mode on 5 of 9 seeds**, where random selection would do
it 22 % of the time — **P = 0.030**. The shipped arm's equivalent mode holds 4 of 45 and the head
serves from it **0 of 9** times. So the head is not ordering randomly and is not broadly
mis-calibrated; **it is actively preferring the worst mode.**

| policy | ship | D1 fix |
|---|---|---|
| what the head picks (rank 0) | 0.782 A | **1.245 A** |
| picking at random | 0.815 A | **1.079 A** |
| a perfect selector | 0.679 A | 0.629 A |

**In the corrected-trunk arm the head selects worse than a coin flip**, 1.245 A against 1.079 A,
against this target's own **0.324 A** seed floor. The mechanism candidate is D24.

**PASS 90 REFRAME, from data already in `perf/of3t_pairbias/fold_rmsd.json`: the head's ORDERING
IMPROVES under D1's fix. What changes is the sample distribution.** That file carries, per seed
and arm, the five per-sample RMSDs in the model's own ranked order, which measures the ordering
directly (`perf/of3t_orchestrator/revision/d10_ordering.py`, `d10_ordering.json`). Five samples
per seed, so a uniformly random selector picks the best 1 time in 5 and sits at mean rank 2.0:

| arm | picks best | mean rank of best | mean regret | within-seed spread | mean best | mean rank 0 | mean pLDDT |
|---|---|---|---|---|---|---|---|
| shipped | **1/9** | **2.56** (worse than random) | 0.102 A | 0.284 A | 0.679 A | 0.782 A | 0.7163 |
| D1 fix | **3/9** | **1.22** (better than random) | 0.616 A | **0.971 A** | 0.629 A | 1.245 A | 0.8011 |

**So "the corrected trunk changed its calibration" is the wrong half of the story.** The head
ranks *better* with the fix — mean rank of best 2.56 to 1.22, picks-best 1/9 to 3/9 — and the
shipped arm is **worse than random**. What the fix changes is **variance**: the within-seed
spread more than triples, 0.284 A to 0.971 A, and the samples go bimodal. Seeds 4, 5 and 6 each
produce a ~0.56 A structure and a ~1.60 A structure, with best values 0.566 / 0.566 / 0.560 and
rank-0 values 1.597 / 1.603 / 1.592 — near-identical across seeds, so these are two discrete
basins rather than a continuum (compare `b2z2-seed-basin-flips-under-any-bf16-perturbation`).

**The defect is therefore not a regression to undo.** The head has always been a weak ranker; it
was invisible because shipped samples are interchangeable, a 0.284 A spread against a 0.324 A
seed floor, so picking randomly cost 0.102 A. D1's fix produces a genuinely better mode — 0.560
to 0.566 A, better than anything the shipped arm reaches, whose best over nine seeds is 0.646 A
— and a worse one, and the weak ranker now costs 0.616 A. **D1 made the model better and made
the selector matter.** D10's own sentence "a worse sample distribution masked a bad selector" was
right; the clause about calibration next to it was not, and the record said both.

This narrows the remedy. Restoring the old calibration is not the goal and would throw away the
better mode. What is needed is a selector that separates two well-separated basins — which is an
easier problem than general calibration, and the head's ordering signal is already better in the
arm that needs it. **What the artifact cannot answer, and needs a card:** per-sample confidence
values were not stored, only per-run aggregates, so ordering-vs-calibration-vs-selection-rule
cannot be fully separated without re-folding. That is `of3t-confhead` deliverable 1.


On **6 of 9 seeds** the confidence head selects a **1.59 A** sample while a **0.56 A** sample
sits in the same batch of five. pLDDT rises over the same runs (**0.7162 -> 0.8014**), so the
head is more confident and no better at ranking — the corrected trunk changed its calibration.

This was invisible while the trunk bias was wrong, because a worse sample distribution masked a
bad selector. It is now the blocker on D1: **fix the ranking, then re-measure the fold delta**,
rather than shipping a trunk fix that improves sampling and degrades what is served.

### D15. Instrument C gated the optimizer check on OUR schedule alignment, not upstream's. FIXED pass 37.

Found by `of3t-orchestrator` when `of3t-updaterule` landed D11 and the instrument's two arms
**swapped places**: the gated arm went from 7.586e-08 to **1.499e-03** and the "info" arm went
the other way.

`instrument_c_optim.py` ran both alignments and printed both numbers, so nothing was concealed
-- but the **pass criterion** was our own convention, `lr(k)` read after the counter moves. The
reference was driven with the lr sequence our code produces, so a §5 PASS asserted that Adam's
arithmetic matches given a matching rate, and could not have caught a one-step schedule offset.
It did not catch D11; D11 was found by reading upstream.

Upstream's order verified on their own code rather than inferred: a real `torch.optim.Adam`
driven by a real `AlphaFoldLRScheduler` (built `last_epoch=-1`, so `_LRScheduler.__init__`
steps it once to 0 before training) reads `param_groups` lr **0.000e+00, 1.800e-06, 3.600e-06,
5.400e-06, 7.200e-06** at updates 1..5. Their update k runs at **lr(k-1)** and their first
update runs at exactly **0**.

The gated arm is now upstream's alignment; the other is kept, renamed
`of3_schedule_offset_by_one`, to size the offset. On a branch without D11's fix the corrected
instrument reads **FAIL at 1.499e-03**, which is the truth about that branch.

---

### D18. BUNDLE-MIN was taped in train mode, so the reference carries a dropout mask nothing can reproduce. **UNFIXED** — the rebuild reproduces bit-identically and publication is still pending, so work remains. ('FIX IN FLIGHT' is not in the status vocabulary, so this entry was invisible to every DEFECTS guard; corrected pass 324.)

Found b

---

## ROTATED 2026-09-22T09:32:13Z

This doc reached 223517 bytes over its campaign and was costing
more to re-read each pass than the passes were worth. The middle is archived verbatim at
`state/archive/of3t-DEFECTS.20260922-113213.md` -- nothing was deleted, and a human can still read it. What follows is the most recent
work, which is what the next pass needs.

---

e hundred'")

I hit both in consecutive composes and the second message is the tell — the guard printed the
exact string its sibling clause refuses.

**Fixed** by widening the pattern to `([a-z][a-z -]*[a-z])`, which accepts the word list's own
output and still refuses an empty or punctuation-only field. The census half is unchanged.

**The class, which is why this is worth an entry rather than a commit line:** a guard built from
two clauses can be individually correct and jointly impossible, and the failure appears only at a
value nothing had reached before. It reports a defect that does not exist while hiding the count it
was built to audit — for one compose the campaign's row census was unreadable and the reason looked
like a formatting mistake in the document. **Where a guard compares a rendering against a pattern,
the pattern must accept everything the renderer can emit**, and the cheap test is to run the
renderer over the range and match each output.

### D202. The 99.2594 % ceiling that TWO charter clauses are measured against counts 0.74055 of a class the campaign's own other artifact measures at 1.52024. FOUND by `of3t-trajwiden`, pass 348. **UNFIXED.**

`COVERAGE_CEILING_IS_NOT_100.json` derives **99.2594 %** as 100 minus **0.74055**, counting the
`input_embedder` host-applied weight. `READABLE_MASS.json` — a different namespace, neither this
row's nor mine — resolves the **same HOST_APPLIED class** over 17 tensors at **1.52024 %**:
`input_embedder` 0.76720 plus `diffusion_module.atom_attn_enc` 0.75304. Applied consistently the
ceiling is **98.47976 %**, a **0.77964-point overstatement**, and the diffusion arm's 1.1286 % that
`COVERAGE_CEILING` lists as RECOVERABLE BY MEASUREMENT is 0.75304 host-applied.

**It is safe to correct precisely because it flips nothing, and that is the test I applied before
touching it.** GRADIENTS' coverage clause reads 97.98499 % and TRAJECTORY's scope reads 88.0819 %;
both fail against 99.2594 % and both still fail against 98.47976 %. A bar correction that moves a
verdict is the failure mode D181 names — this one moves no verdict, so making the two artifacts
agree is bookkeeping rather than advocacy. **Corrected only when a row re-derives it**, not by me
editing a number in prose.

**And the class is not structural either way.** `TT_BIO_OF3_DEVICE_REFATOM`
(`openfold3_host_prep.py:187`, `env_flag(..., False)`) swaps `ref_atom_embed` for
`ref_atom_embed_device` at `worker.py:1561` and the `linear_q` aggregation head at
`host_prep:322`; the flag's own source comment says the eight linears and `linear_q.0.weight` then
run where a cotangent can reach them, and names the READABLE_MASS HOST_APPLIED class as what it
moves. **The op is ported; the number is a flag** — which is D184 seen from the bar's side. With it
on inside the instruments the ceiling is 100 % minus the 0.49477 % held by `FUSED_NOT_SPLIT`,
`NO_ARM`, `NOT_A_LEAF_LAZY` and `ARM_RAN_DID_NOT_CARRY`.

**It must stay default-off for INFERENCE** — a device linear in bf16/fp32 is not bit-identical to
the host float32 one — and the row proposes nothing for the shipped default. Worth 1.1286 points of
coupled headroom, 0.75304 of it behind that flag, with `linear_ref_pos` alone at **0.6734 %**,
59.7 % of the headroom in ONE tensor, at a collection cost of 1.9 h.

**One more `firing != code`, caught by the row rather than by me:** the flag is wired on the
shipped fold path while the trajectory builds `OF3DiffusionModule` directly, *"so the flag existing
is not the flag firing here"*. It needs harness wiring, not a port.

**Status, so this entry does not read as an open question.** Nothing here is a hypothesis. Both
figures are read out of committed artifacts, the inconsistency between them is arithmetic, and the
source of the class is named at file and line. What **remains open** is only which number the two
artifacts are made to agree on, and that is a row's re-derivation rather than an edit of mine.

### D199 UPDATE, pass 348. Both COVERAGE paths are demonstrated to FIRE against a real end-to-end OF3 training step — the first this campaign has ever run. **NOT composed into the charter yet, deliberately.**

`of3t-trainfwd` registered the adapter and ran it:
`tt_bio.train.catalogue.register('openfold3', tt_bio.train.openfold3.adapter)`, batch
`batch_step003.pt`, checkpoint `of3-p2-155k.pt`, 20 rollout steps, upstream's own loss weights,
padded width 384 / 56 real tokens on 5nw3, qb2 card 1, AICLK median 1350 over 159 DURING samples.

    loss                     0.1342802552371707
    forward / backward       8.8555 s / 479.0825 s, step 487.9380 s
    params_reachable         3952      params_with_grad 2498     nonzero 2489
    squared gradient norm    9.785634443843449   (the pinned model reference is 10.279642678524981)

**`model_forward`**: 2489 parameters carry a nonzero gradient with the path on and **0** with it
off, the off arm being the same forward with every objective seed replaced by zeros. A clean
negative control.

**`diffusion_rollout`**: with the rollout replaced by ground-truth structure fed to the confidence
heads and every other tensor identical, **2488 of 2498** parameter gradients move — and the row
reports the size honestly rather than only the boolean: the difference is
**0.0014221448386379067** of the squared gradient norm, with `confidence_head` moving 1.34e-08 of
its own and the trunk 0.0020 of its own. `lazily_minted_by_the_rollout: 0`, so nothing exists only
because of the rollout. The losses differ in the seventh digit (0.1342802552 against 0.1342805299),
which is the same story.

The rule the row applies is the charter's own and it states it: *"a path is covered when a
parameter gradient MOVES against an arm with the path off, not when a config key is set."* By that
rule both paths are covered, which would make COVERAGE — 8 of 8 loss terms, 11 of 11 paths — the
campaign's **first MET condition**.

**Why I have not composed it.** The row is live and its own gate still owes `INFERENCE:`, the
byte-identical shipped fold that proves this adapter changed nothing users get; it touched
`tt_bio/autograd.py`, `openfold3_host_prep.py` and `recipes.py` alongside the new module. Composing
a coverage win while the "did you break inference" check is outstanding is the wrong order, and a
condition flipping to MET on a live row's artifact is exactly where this campaign has been wrong
before. The merge is a one-line source addition to `merge_coverage.py` when the row concludes.

**Worth recording beyond the clause:** `params_with_grad` is 2498 against `params_reachable` 3952
and a 4,170-tensor reference, so this is a genuine end-to-end step but not yet the whole model, and
it does not by itself retire D187's five stitched legs. What it does retire is the claim that no
such step exists.

### D203. The obvious way to widen TRAJECTORY's scope fails SILENTLY and in the flattering direction — it would raise the scope percentage while making the trajectory less honest. FOUND by `of3t-trajwiden`, pass 348, before running it. **RECORDED** as a trap with its working alternative and its pre-check.

The tempting change is one line: at `perf/of3t_trajwide/trajwide.py:559` swap `HP.ref_atom_embed`
for `HP.ref_atom_embed_device`. `tt_bio/openfold3_host_prep.py:262` constructs
`RefAtomFeatureEmbedder` **inside** the function, calls it once and returns tensors — the module is
transient. Two consequences, each fatal on its own:

  * `weights_for(...)` discovers by `walk_device_weights(composed)`, and a module that **no
    attribute of `composed` holds** is not reachable by a walk of attributes, lists and dicts. The
    eight weights would be **absent from the parameter set with no error.**
  * The call sits **outside** the taped per-step `fwd()`, so even if discovered, no cotangent
    reaches them on any step and the optimizer steps eight weights whose gradient is permanently
    `None`.

**The direction is what makes it a defect rather than a bug.** A 20-step run with eight silently
frozen weights scores **581 tensors instead of 573**, so the coupled scope percentage **goes up**
while the trajectory covers less real training. In the row's own words: *"That is a number I would
have reported as progress."*

**Two standing lessons landing together**, and the row named both: `device-computed-weight-invisible-to-training`
(the parameter set comes from a walk and the walk runs the forward, so host-side, fused and
lazily-materialised weights look identical in a reach table) and `eligibility-firing-condition-is-not-a-code-fact`
(`TT_BIO_OF3_DEVICE_REFATOM` is wired at `worker.py:1561` on the shipped fold path, which the
trajectory never runs — the flag existing is not the flag firing here).

**The working change and a seconds-long pre-check are both written down**
(`perf/of3t_trajwiden/NEXT_ACTION_REFATOM.json`): hold the embedder as an attribute of the same
`composed` object, precompute only the inputs once, call it **inside** `fwd()`, drop the host call.
Then assert, before spending 0.93 h, that the 8 names are in the parameter set, that each resolves
a gradient after one taped backward, that `tape_resolves_after_step` counts 8 more than 980, and —
the assertion that catches a change of FUNCTION rather than of PLACE — that `cl0_d`/`plm0_d` match
`HP.ref_atom_embed`'s host output to the device/host precision floor.

**Dispatched as `of3t-refatom`** this pass, because the row that specified it ended while saying it
was holding itself open for exactly this work. The prize is 0.75304 coupled points —
88.08194359237523 % to 88.83498359237522 %, **66.72 % of the remaining headroom**, with
`linear_ref_pos` alone at 0.6734 % — and the cost is half the first estimate because the reference
side is already banked: `theirs/k20.npz` holds all eight entries at every one of the 20 rungs where
the shipped arm's file holds none. The row checked the key lists rather than reasoning about it.

### D204. Nothing checks that a concluded row's findings reach the ledger, and six rows plus one STOP verdict never did. FOUND by `of3t-orchestrator`, pass 349, auditing my own process. **UNFIXED** in general; the ratchet is built.

I absorb rows I dispatched and rows that report while I am watching. A row that concludes during a
pass spent elsewhere can sit unabsorbed indefinitely, and **no check says so**. Measured over the
union rather than asserted: of **99** concluded `of3t-*` rows, **8** are named nowhere in the
DEFECTS union — `of3t-auxgrad`, `of3t-crop768`, `of3t-d116-verify`, `of3t-d117`, `of3t-memory`,
`of3t-readable-mass`, and two stale markers of my own.

**And naming is too weak a test, which the worst case proves.** `of3t-ditcot` IS named in the
ledger, and its concluding verdict is not: a STOP, from 2026-09-21 20:01, refuting a chartered
figure and closing a defect. Both absorbed below, a hundred-odd passes late.

**Built:** a shrink-only ratchet — every concluded `of3t-*` row must be named in the ledger union,
frozen at today's 8 so the number can only fall. **Still UNFIXED:** a row can be named and its
verdict still unabsorbed, and nothing detects that.

### D129 UPDATE and D55 CLOSURE, pass 349 — absorbing `of3t-ditcot`'s STOP, which has been sitting unabsorbed since 2026-09-21.

**The 2.28x D129 was chartered to attribute to an op cannot survive its own boundary. REFUTED.**
The diffusion reference runs ONE shared `layer_norm_z` at all-ones (`max|w-1| = 0.0`) while the
checkpoint's **48 trained per-block tensors are dropped as `unexpected_keys`**. It is not a
`strict=False` slip: upstream 0.5.0's `DiffusionAttentionPairBias` **has no `layer_norm_z` member
at all**, so the reference cannot be built in the checkpoint's architecture. This is D108 arriving
as an arithmetic consequence rather than a note.

The row priced it instead of stopping at the discovery, and our port is the only side that can,
because it runs both layouts. Scope median rel_l2 against the float64 reference: **ours 0.7055 →
reference architecture 0.1065, 6.62x**, with 24 of 24 blocks improving at D129's own leaf and a
median per-block 7.08x. A random-weight control splits it **2.73x layout and 2.43x weight**, so the
gain is not the layout being kinder on its own. **6.62x is larger than the 2.28x**, so a per-op
attribution there would have named an op for an architecture difference, and the ablation stayed
halted. That is the right call and it is why the row ended on STOP rather than a mechanism.

**D55's backward half is CLOSED by the same row.** All four remaining unconfigured reductions are
inert: pull bit-identical at 547/547, 2736/2736, 3/3 and 4/4, with LoFi break controls moving
546/547, 2732/2736, 2/3 and 3/4 on the same sites. Reach was measured rather than assumed and it
mattered — **two of the four never execute in either model scope** (T1 is a dead twin of the rule
production dispatches, 0 entries against 1440; T3 needs the fused-SDPA verb), so those were
measured at op level. An "inert" resting on "never runs" is a different answer, and the row said so.

**A defect-writing rule from it:** the D55 census cites `attention:inner`, an expression **D56
deleted** when it folded that path into the shared `softmax_bw_inner`. Resolve targets by enclosing
function plus expression text, never by line — the third sighting of that trap after D31 and D55's
own re-location.

### D205. 512 is the largest crop that runs, and the ledger has never said so. FOUND by `of3t-crop768`, concluded 2026-09-21; absorbed by `of3t-orchestrator` at pass 349. **UNFIXED** — it is a capability limit users meet.

**Measured, not bracketed.** The row built the missing 544 fixture and ran it: **544, 576, 640 and
768 all refuse**, so there is no rung between 512 and 640 that clears.

**Two different walls, which matters because only one of them is predictable.** 640 and 768 die
with the card full — 23.7 MB and 6.2 MB free device-wide, 0.069 % and 0.018 %. **544 and 576 die on
CONTIGUITY with 6.30 GB and 6.67 GB free**: 576 refused a 2,717,908,992 B buffer inside
`ttnn::concat` → `tilize_with_val_padding`, short by 77,930,560 B per bank against the largest free
block, at 88.45 % occupancy. **A capacity extrapolation cannot see that wall** — the row's own 2.08
fit predicted 576 would clear with 14 % of margin.

**The 576 refusal is deterministic and card-independent**, reproducing byte for byte across card 0
and card 1: same 29,970,916,352 B high-water, same 5,622 allocations, same per-bank
allocated/free/largest-free-block.

**Every refused rung's high-water is a LOWER BOUND**, and the row corrected its own earlier
reporting to say so: the fit predicted 29,314,225,537 B for 576, which 576 passed by +2.24 % after
only 6 of 1,639 parameter gradients and then died. So 768's 1.558x overshoot is a floor.

**And a rule fell out of it:** tile parity. 480 (15 tiles) and 544 (17 tiles) are odd 32-tile
counts and both narrow the fp32-softmax L1 plan to **0 B**, where every even count measured (12,
14, 16, 18, 20, 24) keeps it.

### D206. A training forward that drives the diffusion modules directly skips a typecast the shipped caller does inline, so fp32 weights met bf16 activations — every loss looked sane while the squared gradient norm read 4.87e+11. FOUND by `of3t-trainfwd`, pass 350. **FIXED** by the row.

`OF3SampleDiffusion.__call__` performs the typecast inline. A training forward that drives `dc`/`dm`
directly never passes through it, so the arm ran **fp32 weights against bf16 activations** with no
error raised anywhere.

    squared gradient norm   4.87e+11      against the model denominator 10.2796

**Every loss value looked sane.** That is the whole defect: the loss is computed from activations
that are individually plausible, so nothing in the forward flags a precision boundary crossed the
wrong way, and the damage shows up eleven orders of magnitude out in the gradient — which is the
one number nobody reads until the end of a training step.

**Fixed as a shared method rather than a second copy**: the boundary is now a method both callers
use, and `ttnn.typecast` has a tape entry, so it is differentiable rather than a hole.

**The class, and why it is worth an entry.** A *loss that looks right is not evidence the gradient
is right*, and a precision boundary maintained inline in one caller is a trap for every other
caller by construction — the campaign has the same shape in D153 (a missing reference path resolves
to nothing and the run continues) and in `firing != code`. **Where two callers must cross the same
boundary, the boundary belongs to neither of them.** And if an arm reports a squared gradient norm
eleven orders off its own denominator, that is a configuration fault to find before it is a
finding: no model produces 4.87e+11 against 10.2796 by being inaccurate.

### D129 UPDATE 2, pass 351 — scoping my own pass-349 sentence, and clearing the campaign's main POSITIVE claim against it.

Pass 349 absorbed `of3t-ditcot` with the sentence *"the diffusion reference cannot be built in the
checkpoint's architecture"*. That is true of the thing the row measured and **ambiguous about which
reference**, which matters because one reading would invalidate this campaign's largest positive
result. Both halves checked this pass rather than assumed.

**What ditcot actually measured.** *"Loading `of3-p2-155k.pt` the way every diffusion reference
BUILDER does gives `missing_total 3` and `unexpected_total 48`"* — all 48 unexpected keys
`attention_pair_bias.layer_norm_z`. That reference module has exactly **1** `layer_norm_z` site,
all-ones, `max|w-1| = 0.0`, where ours runs **24 trained tensors** at mean 0.276-0.568, std
0.130-0.224. So the defect is in the **module the diffusion reference builders construct**, not in
every artifact with "reference" in its name. The row refused to form a transport ratio against that
denominator, per A27, rather than publish one.

**And the pinned bundle is NOT that module.** Checked directly on qb2:
`/home/ttuser/of3t_refprec/bundle_ref/grads_f64_043.pt`, sha256 `1d4ea9225f…`, holds **4,170**
tensors, **143** `layer_norm_z` entries, **28** of them under `diffusion_module`, spanning **24
distinct `diffusion_transformer.blocks.N` indices**. It carries the per-block layout.

**So the campaign's main positive claim stands, and it is worth saying that it was the claim most
worth attacking.** `MODEL_shipped.json`'s 92.1568 % of the mass at **1.1031x** upstream's own bf16
— "outside the pairformer trunk the gradient is at upstream's own accuracy" — is scored against
that bundle over four scopes (`diffusion`, `cond`, `aux`, `msa`, 907 tensors). Its denominator can
express the checkpoint's architecture, so ditcot's 6.62x does not reach it. **Unchanged, now for a
measured reason rather than by omission.**

**The correction I owe on my own entry:** "the diffusion reference" is not one object, and pass
349's sentence should have said *the reference builder's module*. A defect scoped to a builder read
as scoped to every reference would have put the campaign's best result in doubt for no reason —
the mirror image of the flattering direction, and the same imprecision either way.

### D199 UPDATE, pass 351. **CLOSED**, and COVERAGE with it — the campaign's first met charter condition.

`of3t-trainfwd` concluded at `bdbe43a95`. Coverage goes **9 of 11 to 11 of 11**, loss terms stay
8 of 8, and `CHARTER_EVIDENCE.json` reads **1 of 3** for the first time in the campaign.

**The precondition I set at pass 348 is met, which is why this composed now and not then.**
Inference is **byte-identical, verified TWICE** — after the adapter landed and again after the
sampler refactor — `ubq.cif` `6a8a43ca…` and `ubq_model_1.cif` `62d94b47…` matching base commit
`16eac05c1` exactly. A coverage win composed while "did you break inference" was outstanding would
have been the wrong order.

**Composed upgrade-only, and the guard says so.** `merge_coverage.py` takes
`COVERAGE_TRAINFWD.json` as a third source pinned by sha256 (`0fd2c866bb8eb1e9`), and **refuses**
if that row reports a path uncovered where `of3t-covpaths` has it covered: two rows disagreeing
about one path is a finding, not a merge. Both upgraded entries carry `superseded: {was: false}`,
so the artifact records that these were the two the census confirmed uncovered.

**Three findings the row left behind, all kept.**

*The objective's contract assumes every label is batch-side and two are not.* `plddt`'s label is a
function of the prediction, and `edm_scale`'s sigma is drawn per forward. That belongs in
`objectives.af3_loss`, **not** in a per-model adapter — a UNIFIED-not-per-model call made against
its own convenience.

*What it deliberately did not ship.* The one-step denoise arm is wired and runs, six of eight terms
firing, but **3 of 3400 parameter gradients come back non-finite** — always
`sampler.dc.w_lin_z/w_lin_s/w_lin_n`, whose dW reduces over 147,456 token pairs. Defaults off. **The
seeds are ruled out by measurement rather than argument**: the largest is `pred_xyz` at 11.5669,
`losses.mse` already applies upstream's stop-gradient Kabsch, and the seed norm is *identical* for a
structure rotated 1.1 rad and translated 12 Å. The amplification is inside the diffusion backward
and is not root-caused. That is an unowned object and it looks like `of3t-tapeamp`'s neighbourhood.

*P5 refuted and reported as a miss:* it predicted over 600 s per step and measured **496.9 s**
(crop 384, AICLK median 1350 sampled during; 20 steps = 2.76 h).

**And P4 HELD, which is the discipline worth recording.** There is still no unstitched model-scope
gradient and **D187 stays open** — because the pinned float64 reference was taken with upstream's
own cotangent and replayed draws, while this forward seeds from our loss and draws its own noise.
Scoring across that would be exactly the cross-frame error this campaign has already paid for once.
The row had the most attractive deliverable in the campaign within reach and **refused to form it**,
on D186's grounds, against its own interest.

### D197 UPDATE 2, pass 352. TRAJECTORY's coupled scope is **88.83498302148425 %**, the projection confirmed to 5.7e-07 — and the shared tensors moved too, in the flattering direction. Still **UNFIXED**.

`of3t-refatom` executed the change `of3t-trajwiden` specified, and ran D203's trap-check first.

**Neither silent failure is present, and the check was committed before it ran** (`26b7fb834`, its
result then committed pass-and-fail together at `717bf1d20`). It drives `build_ours`/`run_ours`
unchanged over a one-level one-step partition — the real arm's program, not a replica — for
**4.84 s**, about **0.3 %** of what it protects:

    1  the eight in the parameter set      PASS   8 of 8, ref_atom_feature_embedder.w_*
    2  each resolves a gradient            PASS   8 of 8 participating, 988 of 988 overall
    3  tape_resolves_after_step            PASS   988 = 980 + 8, of_walked 988, rebound 988
    4  place moved, not function           FAIL   cos and rel L2 pass, norm ratio misses

**581 is the honest 581.** D203 warned that a silent failure would also score 581 tensors instead
of 573; assertions 1-3 are what separate the two, and they were run before the 0.93 h rather than
inferred from the result.

**Assertion 4 failed and the bar was not moved.** `plm` norm ratio 0.9988837 against a 1.0e-3
tolerance — 12 % over — while cosine reads 1 − 1.5e-7 over 917,504 elements and rel L2 1.241e-3.
The row's proposed mechanism (a `compute_kernel_config=None` matmul on bf16-truncated operands) is
**refuted by its own control**: truncation overstates the device's norm bias 4.2x on both legs and
nearest-rounding gets `plm`'s sign wrong. What survives is bracketing — the device sits between two
**width** variants of one function, 1.8x finer than nearest bf16 and 4.2x coarser than fp32 — which
answers assertion 4's actual question without re-scoring the bar.

    coupled scope   88.08194359237523 %  ->  88.83498302148425 %   (581 of 761 tensors)
    points added    0.75303942911        projected 0.75304, confirmed to 5.7e-07
    resolves        988 of 988 walked at every one of the 20 rungs
    cost            1620.85 s, 81.04 s/rung   (the 3335.86 s estimate was a shared board pair)

**Against the 99.2594 % bar it still misses**, by 10.417 points.

### D207. Adding tensors to a scored set also moved the shared tensors' trajectory, in the flattering direction, and nothing explains it. FLAGGED by `of3t-refatom`, pass 352. **UNFIXED.**

The widening was expected to be **additive**: eight more tensors, the other 573 unchanged. They did
not stay unchanged. At k=20 the shared set reads `rel_d` **2.246887e-01** against the shipped arm's
**2.564253e-01**, and worst per-tensor **3.454769e-01** against **8.110072e-01** — on a *different*
tensor.

**The row reported it as unexplained precisely because it improves the number**, which is the
correct instinct and the reason this is a defect entry rather than a footnote. A scope gain that
quietly carries an unattributed accuracy change is two results presented as one, and the direction
means nothing would have prompted the question.

**What separates it:** the host-leg arm re-scored over the same 581 names. Not run; not owned.

**The general rule, which is new and belongs in PROTOCOL's vicinity:** *a change that ADDS tensors
to a scored set must be shown purely additive on the tensors it did not add*, or the gain and the
movement must be reported as two findings. Same family as D181 (a clause repair that lets you
declare success) and D195 (an instrument that names a carrier), and it arrived the same way all
three did — from a row checking the result it wanted to be true.

### D191 UPDATE 2, pass 352. **It is a STEP, not a per-block factor** — created in the backward of blocks 45 and 44, then carried and diluted over the remaining 44. Still **UNFIXED**, and my brief's premise is refuted.

`of3t-tapeattn` concluded. The cotangent ladder at both widths, in-frame float64, upstream's own
bf16 recipe as the floor at each rung, 56 real tokens:

    rung 46   7.1347e-01   7.4352e-01   1.042
    rung 45   6.1508e-01   1.1427e+00   1.858
    rung 44   6.9791e-01   1.5364e+00   2.201
    rung 43   6.9835e-01   1.5888e+00   2.275
    rung  0   8.1204e-01   1.4000e+00   1.724

**A per-block factor would compound. This decays monotonically from rung 43**, which is the shape
of a step that is created once and then diluted. And **rung 44's 2.201x sits against D191's
model-scope 2.1795x, so all of D191 is present four blocks in** — the other 44 blocks add nothing.
Pass 347's relocation to "a per-block factor of ~2.9x on the single track's cotangent", which I
carried into the brief, is **refuted**.

**100 % of the growth is ours**, and that is measured rather than inferred: the float64 reference at
crop 64 and padded 384 is **bit-identical on the 56 real tokens, rel_l2 exactly 0.0 at all 49 rungs,
both tracks**.

**The softmax candidate fell to a DECOMPOSITION rather than to a smaller number**, which is the
methodologically interesting part. `VERBS.json`'s pooled **1.7797x** for softmax is a
reference-norm-weighted RMS, so it moves when the mass moves:

    384 errors, 64 weights    0.9369x    softmax's own arithmetic, weights held
    64 errors, 384 weights    2.0923x    reweighting alone, errors held

**Softmax gets 6.3 % BETTER per firing at 384**; only 10 of 48 firings are worse. `matmul` and
`multiply_` are flat by every reading. That is D195 and "a share moves when its denominator
collapses" applied to a pooled ratio by the row that owned it.

**Floors taken before any scoring, not after:** A/A determinism exactly 0.0 at both widths (98/98
bit-identical), wall clock 1.92 % over the byte-identical 384 pair, instrument inert by content sha
at both widths, and the checkpointed reference bit-identical to the plain one. AICLK n=122, min 800,
median 1350, qb1 card 2 p150a.

**What it does NOT claim, stated by the row rather than found by me.** The step is **not attributed
to a named op**. The census sees only the single track (heads = 16), so **the pair track and the
triangle multiplications run in the same blocks 45/44 and remain live candidates**. And the ladder
is **saturated** — O(1) relative error from rung 47 down at both widths, floor included — which
bounds what it can localise.

**The next arm is named and cheap**: pin the single track's softmax backward to its float64 VJP and
re-score; `tapecensus.py` already computes that VJP.

### D58 UPDATE, pass 353 — the ~20x forward-to-gradient factor is the FUNCTION's, not the tape's, so D58 as filed is **REFUTED**. It and D30 stay **UNFIXED** pending the row's own closure statement.

`of3t-tapeamp` built the control the object had never had: **upstream's own forward-to-gradient
ratio**, at the same boundary, against the same float64 reference, on the same 547 tensors.

    upstream 0.4.3 bf16 recipe      7.666x        ours  11.026x
    upstream 0.4.3 fp32 recipe      9.326x        and four orders lower in ABSOLUTE error

**Our arm beats upstream's own on BOTH halves** — the forward by **1.959x**, the gradient by
**1.362x** — and the arithmetic closes exactly: **1.959 / 1.362 = 1.438**, which *is* the ratio
excess. Our factor is the larger one precisely because the denominator is the half we beat it on
hardest. A forward-to-gradient ratio is not a defect measure; it is a quotient of two accuracies,
and improving the numerator less than the denominator raises it.

**Three precisions span 7.7x to 11.0x, and that range is itself the discriminator.** A dtype
boundary cannot survive a four-order change in absolute error. The conditioning of the
Jacobian-transpose product can, and `of3t-bwdaccum` measured exactly that on the trunk. So D58's
claim that *"the ~20x amplification belongs to the tape, not to any module"* is **refuted**: the
two independent sightings at 19.6x and 19.8x agreed to two significant figures because they are the
same property of the same function, not because a shared tape component injected it.

**And the D206-class boundary I put FIRST in the brief is refuted by a call count, not a route
read.** 1,879 node firings, **zero dtype reconciliations**. The tape has exactly one place a
cotangent's dtype is reconciled to its forward value's — `autograd.py`'s backward loop — and on the
diffusion scope it **never fires**: the whole backward runs fp32 against fp32. The only dtype
crossings are 96 calls of the model's own explicit typecast verb, which is taped and
differentiated. My lead was worth pricing and it is dead.

**Controls, which are why this reads as a result rather than a story.** A/A bit-identical on all 48
forward values, the gradient median, the mass-weighted gradient and the ratio — determinism floor
exactly **0**. A break control that **rolls the cotangent moves the gradient 10.058x and leaves the
forward bit-identical**, which is the control a ratio needs and the one nobody had run. The
wall-clock floor is 54 % on that box at load 9-13 and **no claim rests on a timing**. D141's
fingerprint guard is armed and passes — 761 parameters, 24 per-block `layer_norm_z`, 0 unexpected,
0 missing — so this is not `of3t-ditcot`'s architecture case. The device arm reproduces bit-for-bit
across hosts and cards, and upstream's bf16 and fp32 gradients reproduce the qb2 record to every
digit at `OMP_NUM_THREADS=8`.

**Not closed here.** The row is live and its gate owes a `DEFECTS:` field saying which of D30, D58,
D129 and D55 it closes, narrows or leaves untouched. I am not closing a USER-FACING defect on my
own reading of a live row's commit message — that is the closure-plan failure this campaign already
had, in the other direction.

### D30. **CLOSED as not a defect**, pass 353, by `of3t-tapeamp` against the correct denominator and the correct reference.

Filed at forward 8.34e-03 / gradient 1.6588e-01 / **19.6x**. The number was corrected twice before
this row touched it: `of3t-tapediverge` read the shipped arm at 1.250047e-01 (**14.75x**) with the
forward bit-identical, and `of3t-ditref` repaired the denominator to 9.344246e-02 (**11.026x**).

This row prices **7.666x of that 11.026x inside upstream 0.4.3's own bf16 recipe**, with the
remaining **1.438x** accounted for arithmetically by our forward being **1.96x** more accurate than
upstream's. And the direct reading settles it: **our gradient is 0.734x upstream's own** on the same
547 tensors and **0.5865x mass-weighted** on the 761 (`of3t-ditref`'s `VS_FLOOR.json`,
`ours_over_floor_x` 0.5865346734980963) — **below the reference's own floor**. There is no
amplification defect at the correct denominator against the correct reference.

**What this does NOT close, named by the row rather than found later:** the tail. A worst tensor at
**1.850397e+01** is a different object from a median factor and belongs to whichever row owns the
per-tensor outliers. Unowned.

### D129 UPDATE 3, pass 353. **CLOSED** — dissolved by `of3t-ditref` and absorbed here, the leaf reading below its own floor on every like-for-like statistic. The ratio in circulation for it is not like-for-like.

`of3t-tapeamp` notes, correctly, that D129 was already dissolved and that my brief was wrong to list
it as waiting on that row. `VS_FLOOR.json`'s `D129_LEAF`
(`conditioned_transition.layer_norm.layer_norm_s.weight`, n=30 both sides) reads **below its own
floor on every like-for-like statistic**:

    median          ours 0.08544921051930113   floor 0.12498386673173574   ratio 0.6836819243455522
    mass-weighted   ours 0.12190333295342548   floor 0.15931532097065704   ratio 0.7652

**The 0.536x in circulation mixes statistics** — it divides the MEDIAN numerator 0.0854492105 by the
MASS-WEIGHTED floor 1.5931532097e-01. The verdict is unchanged, since 0.684x and 0.765x are both
under 1, but the quoted figure understates by about 1.3x **in the flattering direction**, and the
artifact records `ratio_median` explicitly one field away from the two it used. Corrected here
rather than carried: the like-for-like readings are **0.6837x median** and **0.7652x mass-weighted**.

### D58 UPDATE 2, pass 353. Still **UNFIXED** — narrowed to one leg, and the untouched half is named.

`of3t-tapeamp` refutes the *"belongs to the tape"* half **on the diffusion track** and supplies the
mechanism that explains the two-significant-figure agreement without an amplifier. It does **not**
measure `msa_module`: there is no upstream bf16 arm for that boundary anywhere in the campaign, and
building one is a boundary capture plus a float64 reference, not an afternoon. So D58 stands with
one leg re-explained and the other unmeasured, and the prediction that would close it is registered
in that row's DOESNOT.

**D55's forward arm is UNTOUCHED** and the row says so in those words: it measured no
kernel-config lever, set no `precise_config()`, and did not run the forward-side census D55's
forward half needs. Its backward half was closed by `of3t-ditcot` on four reductions proved inert
with measured reach.

**The row stated all four in those words — closes, narrows, leaves alone — and gave its reason:**
*"the campaign's closure plan has already carried a concluded row as a live owner for days and a
narrowing described as a closure is how that happens."*

### D58 UPDATE 3, pass 354 — the full reading from `of3t-tapeamp`'s conclusion, and my own triage entry was carrying a figure two revisions stale. Still **UNFIXED**.

**The number.** The "~20x" is **11.026x** (diffusion) and **10.903x** (`msa_module`) at the repaired
denominator. `of3t-ditref` and `of3t-tapediverge` had already moved it, and **both my brief and
`UNFIXED_TRIAGE.json`'s D58 entry were still quoting the old one** — corrected this pass.

**The measurement nobody in the campaign had made** was upstream 0.4.3's own **forward accuracy** at
this boundary. Its gradient was on record (median 1.2940662e-01) and its forward *seconds* were; its
forward *accuracy* was in no artifact. Both halves, one process, one host:

    arm                        forward          gradient (547)     factor
    ours, device               8.4748009e-03    9.3442464e-02      11.026x
    upstream 0.4.3 own bf16    1.6601749e-02    1.2727639e-01       7.666x
    upstream 0.4.3 own fp32    1.2939393e-06    1.2067747e-05       9.326x

**69.53 % of our factor is upstream's own.** To reach upstream's 7.666x we would have to make our
forward **1.96x worse**.

**Three independent confirmations it is not a D206-class boundary**, and the third is new:

  * a call census on device — **0 dtype reconciliations in 1,879 node firings**, every firing
    fp32-against-fp32, the only crossings being 96 calls of the model's own taped `typecast`;
  * the factor **survives fp32**, four orders of absolute error away;
  * **the per-block cotangent curve has the SAME SHAPE in both precisions** — same five-block ramp,
    an 8.191x / 5.726x step at the *same* boundary (19→18), same plateau — while the absolute errors
    sit **11,792x to 18,350x** apart.

**And the shape answers `of3t-bwdaccum`'s discriminator with NEITHER of its two options.** Not flat
at the bf16 floor (a wrong leaf backward), not monotone with depth (per-block injection), but
**ramp / step / saturation** — that discriminator's *third* pre-registered shape, first seen on the
DiT-24 and present in the reference's own arm. **The step moves relative error 8.191x at 1.399x
magnitude, so it is a cancellation event**, not an amplification.

Worth setting beside `of3t-tapeattn`'s trunk result from pass 352: a step at blocks 45→44 there, a
step at 19→18 here, both precision-independent, both cancellation-shaped, in different scopes.
Whether that is one phenomenon is not established and is not claimed.

**`msa_module` is measured at 10.903x but has no upstream comparison**, because there is no upstream
bf16 arm for that boundary anywhere in the campaign. So D58's two legs are now: diffusion
re-explained, msa priced but uncompared.

### D208. TRAJECTORY's clause is REPOINTED to a reference-derived bar of 89.2105 % with a coupled-scope requirement, and no artifact emits the coupled field. **UNFIXED** — the condition reads NOT MET on both new clauses.

`of3t-trajwiden` proposed this and I checked it before making it, because **the bar moves 10.05
points toward passing** and that is the direction that needs a reason.

**What the clause now reads**, against `perf/of3t_refatom/traj_refatom.json`:

    steps >= 20                                  20                       MET
    scope.pct_of_model_sq_grad_norm >= 89.2106   88.83498302148425    NOT MET by 0.376
    scope.coupled is True                        key absent           NOT MET
    per_step moves (d_theirs, d_ours)            19 of 19 fitted rungs    MET

**Why the old bar was unsatisfiable rather than demanding.** 99.2594 % is the ceiling on what the
**STATIC single-step** instrument could measure — 100 minus the 0.74055 % applied on the host after
a `ttnn.to_torch`. A trajectory needs four things where the static instrument needs one: a taped
device forward **and** backward at scope, a captured 0.4.3 boundary whose cotangent is complete,
upstream's module runnable standalone at arbitrary weights for 20 steps, and 20 affordable
optimizer steps on both sides. **Giving two instruments of different reach the same number made the
weaker one unsatisfiable** — D181 from the other direction, where that rule forbids coupled clauses
reading different scopes and this was one number read by two scopes.

**The four things I checked before lowering a bar toward passing:**

1. **It does not let the campaign declare success.** 88.83498 against 89.2106 leaves it unmet by
   0.376 points, and the coupled clause fails outright. Two substantive failures.
2. **The bar is a measured property of the REFERENCE.** 89.2106 % is the share of upstream's own
   float64 gradient mass inside `diffusion_module`, summed from `grads_f64_043.pt` in
   `SECTION_MASS_MEASURED.json`. It would read the same if our port did not exist.
3. **Nothing is lost by the swap.** Pass 346 declined this repoint on the suspicion that a
   `clauses` block would be dropped. The `moves` check reads `per_step`, and the new artifact
   carries all 20 entries with both norms, 19 non-zero each (k=1 is the AF3 warmup no-op). Checked,
   not assumed.
4. **What would raise the bar is named**: a capture spanning more than one section AND a taped
   whole-model device forward+backward. The reference half exists and is digest-pinned
   (`bundle_min_043`); the device half does not.

**`scope.coupled` is required and unemitted, which is the defect this entry carries.** A UNION of
per-section trajectories would satisfy a scope number while testing none of the coupling: every
boundary the campaign holds is a frozen capture of upstream's r = 0 step, so an `aux_heads`
trajectory reads upstream's step-0 trunk outputs at every k, never our step-k ones. The proposing
row **killed its own pass-1 number of 97.9849 %** on exactly that ground, which is why the proposal
is worth taking.

**One mechanical note.** I first wrote the clause with an `==` operator, which `charter_evidence.py`
does not have — its break control raised `KeyError: '=='` and refused to publish rather than
emitting an artifact with an unevaluated clause. The vocabulary already had `is`, doing strict
equality with a type check. The guard refusing to publish on an operator it cannot evaluate is the
right failure.

### D209. A bar set by rounding a measured CEILING up is unsatisfiable by construction. FOUND at pass 356 by the orchestrator against its own pass-355 clause. UNFIXED

Pass 355 repointed the TRAJECTORY charter clause onto the diffusion module's own share of
upstream's float64 gradient mass and wrote the bar as **89.2106 %**. The measured share is

    9.170528876862544 / 10.279642678524981  =  89.2105802084 %

so `>= 89.2106` demanded **more than 100 % of the module the bar is a property of**. Covering every
one of the 761 reference tensors at that boundary would have read 89.2105802084 and failed the
clause by 0.0000198 points. The arithmetic was never wrong; the transcription rounded a CEILING up.

Corrected at pass 356 to **89.2105**, rounded down, in `workstreams/_of3t_donecheck.py`. The
correction moves toward passing and so was checked against the rule for that direction before it was
made: it flips no verdict. Today's best artifact reads 88.83498302148425 and misses the corrected
bar by the same 0.3756 points it missed the old one by, and the clause's sibling `scope.coupled` is
still unemitted. Two substantive failures before and after.

**The general rule.** A bar that is a MEASURED CEILING must be rounded DOWN. A bar that is a floor
may be rounded up. The direction that is safe for a target is unsafe for a limit, and the two are
easy to confuse because both read as "be at least this good". This is D201 arriving from the other
side: there, two clauses of one guard were jointly impossible at their limit; here, one clause is
impossible against its own limit. Both were found only by asking what the BEST POSSIBLE artifact
would score, which is a cheap question and is now the standing check before any bar is written.

**How it was caught.** Not by a guard. I was sizing a row against the gap and computed what full
coverage of the module would read, which is the one arithmetic that exposes it. No composition
check tests a bar for reachability, and `audit_evidence.py` cannot — a bar's ceiling is not in any
artifact it reads. The check that would have caught it is the one I now owe every future bar.

**Scope.** CAMPAIGN-INTERNAL. It gates no user-facing behaviour; it gated the campaign's own exit
condition, which is worse for the campaign and invisible to anyone who runs the model. Filed against
myself: pass 355 wrote it, pass 356 found it, one pass of exposure.

### D210. The diffusion transformer trains 14.2M parameters upstream does not have (UNFIXED, USER-FACING)

Found by `of3t-trajfull`, pass 357, outside its own scored set and reported anyway, which is the
reason it is here at all.

Our DiT fuses q, k and v into one `qkv_w` and pads head_dim from **48 to 64**
(`openfold3_diffusion_transformer.py:134-151`). The pad lanes are part of the optimizer's parameter
set. At `w_0` they are **exactly 0.0**; by k = 20 of the trajectory the largest reaches
**3.494e-04**. **Adam's scale invariance is the mechanism**: the update is `-lr * m / (sqrt(v) +
eps)`, so a numerically tiny device-backward gradient in a lane that should have none still
produces a full lr-sized step. 14.2M elements.

Measured, not argued: `perf/of3t_trajfull/traj_trajfull.json`
`controls.split_roundtrip_every_rung.fused_pad_max_abs = 0.00034944515209645033`, with 48 of 48
fused slots round-tripping bit-exactly at every rung.

**It moves no number the campaign currently quotes.** The pad columns are not in upstream's
parameter space and are sliced off before v is used, so the 89.21058020840096 % trajectory scope,
its `rel_d` and every gradient reading exclude them. That is exactly why it can sit here unnoticed:
nothing that is graded reads it.

**Why it is USER-FACING rather than campaign-internal.** It is a property of the shipped port's
optimizer, not of an instrument. Any training run on this port drifts 14.2M parameters that
upstream holds at zero, and whether that stays inert depends on the slicing being exact on every
path, at every width, forever. An inert-by-construction parameter that trains is a latent
correctness question, not a cosmetic one.

**Repair, and why it is not this row's.** Masking the pad lanes out of the optimizer is a model
change and belongs to the row that owns the DiT. The fix must carry an inference A/B, because the
DiT is on the shared diffusion path.

Status: **UNFIXED** -- no owner as of pass 357, and named in the orchestrator's `GAP:`.

### D211. "GRADIENTS fails on coverage alone" was scoped to the narrow artifact (UNFIXED as prose, CORRECTED pass 357)

`PROVES:` carried the sentence *"the gradient is inside the reachable bar at 0.9592x and no worse
per-tensor than upstream's own step, and GRADIENTS now fails on coverage alone"*. Both halves are
readings of `perf/of3t_ditmodel/MODEL_d56_retake.json`, which scores **907** tensors over
**92.15682156952718 %** of the model's gradient mass. The live gate has graded GRADIENTS on
`perf/of3t_modelboundary/MODEL_withtrunk_n384.json` — **3643** tensors, **97.98499306866148 %**,
identical float64 / upstream-bf16 / upstream-f32 references by sha256 and the identical denominator
10.279642678524981 — since that artifact landed.

On the graded artifact the shipped arm reads **0.520124** against its own A26 bar **0.147353**,
**3.5298x** over. So raising coverage does not leave the accuracy clause standing: **the widening
that closes the coverage clause is what breaks the accuracy clause**, and the two cannot be
satisfied by any artifact the campaign holds today. "Fails on coverage alone" was true of a scope
nobody grades.

**The arithmetic, computed at pass 357 from the graded artifact's own `per_section` block.** One
section carries all of it:

    section                                    %mass      ours   theirs       x
    diffusion_module.diffusion_transformer   43.6221    0.1086   0.0578   1.878
    diffusion_module.diffusion_conditioning  36.9462    0.0646   0.0625   1.034
    pairformer_stack                          5.8282    2.0151   0.3148   6.402
    diffusion_module.atom_attn_enc            4.7348    0.0988   0.0709   1.395
    aux_heads                                 2.8431    0.2360   0.2394   0.986
    diffusion_module.atom_attn_dec            1.2843    0.0482   0.0500   0.963
    msa_module                                1.2317    0.1887   0.1788   1.055
    diffusion_module.layer_norm_s             0.9835    0.0675   0.0310   2.176
    diffusion_module.layer_norm_a             0.2666    0.0349   0.0370   0.945
    diffusion_module.linear_s                 0.2445    0.1124   0.0654   1.718

Hold every other section where it is and put `pairformer_stack` at upstream's own bf16 level and
the model-scope figure falls from **0.5010 to 0.1240**, which is **0.841x** the 0.147353 bar and
passes. (0.5010 is my re-pool of the section table; the artifact's own 0.520124 pools per tensor,
a 3.8 % difference that changes no conclusion.) **The trunk at 5.83 % of the mass is the whole of
GRADIENTS' accuracy miss.** Nine of the ten sections sit between 0.945x and 2.176x of upstream's
own bf16, and the two above 1.7x carry 1.23 % of the mass between them.

**Scope.** CAMPAIGN-INTERNAL, as a reporting defect: the shipped port is what it is either way.
The cost is that the campaign has been describing itself as one wiring job from GRADIENTS when it
is one accuracy defect plus a wiring job, and the accuracy defect already has a live owner
(`of3t-trunkact` from pass 357, on the trunk's forward activations). Corrected in `PROVES:`,
`GAP:` and `VERDICT:` at pass 357.

Status: **UNFIXED** -- the prose is corrected, but the condition it describes still holds: no
artifact the campaign holds satisfies GRADIENTS' coverage and accuracy clauses at one scope.
Named in the orchestrator's `GAP:`.

### D212. A bar-repoint proposal priced every route through a shipped env flag the training path does not read. FOUND at pass 358 by the orchestrator, against of3t-covdefault BAR section. FIXED by the same pass -- the repoint was refused and never adopted. The coverage leg it was about stays open in GAP, owned by of3t-refcov.

`of3t-covdefault` concluded NO-GO on flipping `TT_BIO_OF3_DEVICE_REFATOM` and that half is right:
the flag ON costs **+198.476 ms cold / +50.231 ms warm** on openfold3 and **+192.515 / +48.185** on
openbind at 1350 MHz on matched hardware with an A/A floor of 0.0 A, and it moves the fold digest.
Nothing here reopens it.

Its BAR section then proposed repointing `coverage_total.pct_of_model_compared` from **99.2594** to
**97.9849** — the shipped arm's own measured value — on the reasoning that *"every route that
reaches it requires the flag ON, which this row measures as an inference regression... The clause
can only be satisfied by breaking the hard constraint it sits beside."*

**The flag is not on any route, and three code facts say so:**

    tt_bio/train/openfold3.py:332      calls the HOST `ref_atom_embed` unconditionally. The
                                       training adapter never reads the flag, so flipping the
                                       shipped default adds nothing to the training arm.
    tt_bio/train/openfold3.py:324-332  host prep runs ABOVE `with ag.tape():` at line 362. Even
                                       flag-ON puts `linear_q` on the card with no tape open, so
                                       no cotangent reaches it and coverage does not move.
    perf/of3t_diffusion/device_gradient.py:586-605
                                       the arm that DID reach the eight. It constructs
                                       `RefAtomFeatureEmbedder` directly, hoists
                                       `ref_atom_device_inputs` out of the loop and calls
                                       `ag.parameter()` on the walked weights. No env var, no
                                       shipped-code change, zero inference reach.

`tt_bio/train/` is training-only — nothing under `tt_bio/` imports it on a fold path (`main.py:1694`
lazy-maps the `finetune` CLI entry; `autograd.py:1116` imports `train.losses` lazily inside a tape
closure). A change confined to it cannot reach inference **by construction**, which is the gating
principle D137 already established for the float64 softmax and `ops.taping()` implements for nine
fused kernels.

**So the bar is reachable without touching inference**, from the row's own artifact
(`perf/of3t_covdefault/COVERAGE_CEILING.json`, denominator recomputed 10.279642678524986 against
the published ...981, rel diff 5.18e-16):

    shipped, measured                                     97.98499306866148
    + diffusion_module...ref_atom_feature_embedder (8)   +0.7530394291090192  -> 98.73803249777050
    + input_embedder.atom_attn_enc.linear_q.0.weight     +0.7405050029283714  -> 99.47853750069887
    the bar                                               99.2594                        clears

**Decision: 99.2594 stands, GRADIENTS' coverage leg reads UNMET, and the wiring is dispatched as
`of3t-refcov`.** `CHARTER_EVIDENCE.json` and `_of3t_donecheck.py` were checked and carry the
original bar; nothing had adopted the repoint, so there is nothing to unwind.

**The lesson, and it is the general one.** The repoint was the flattering direction — a bar equal
to the measured value turns a miss into a ratchet — and it was argued from a real, well-measured
inference cost, which is what made it credible. `of3t-trajwiden` killed its own 97.9849 % bar on
exactly this ground and the campaign still nearly took it a second time. **When a row prices "the
only route" to a bar, check that the route it priced is the route the code takes.** Here the
priced route was a shipped inference flag and the actual route is a training-only file the flag
does not appear in. A cost measured on the wrong route is a true number about the wrong thing.

### D213. main's test suite is RED: D116's landed code cites two artifacts that stayed on the row branch. FOUND by `d126-main-reach` pass 358, CONFIRMED against git by the orchestrator. UNFIXED, and it is not this campaign's to land.

`tests/test_perf_citations.py` on `origin/main` requires every `perf/...` path cited from shipped
source to exist. `tt_bio/autograd.py` on main cites two that do not:

    perf/of3t_d116/amplify.json    MISSING on origin/main, present on wk/of3t and wk/of3t-d116
    perf/of3t_d116/rowsum.json     MISSING on origin/main, present on wk/of3t and wk/of3t-d116

Verified with `git cat-file -e` against each ref rather than read off a row's report.

This is the exact failure the test's own docstring names: *"When a lever is hand-ported onto main
from a worker branch, the comment comes across and the artifact it names does not, leaving a
pointer to nothing."* D116's `softmax_bw_inner` and its renorm branch were hand-ported; the two
JSONs behind the numbers in their comments were not.

**The fix is to land the two artifacts, not to reword the comments.** The citation exists so a
reader can reach the measurement; deleting the pointer deletes the evidence trail and leaves the
tuning constant unsourced, which is what the test was written to prevent.

Handed to `land-standing`, which owns landings on main. This campaign does not merge to main and
a red main blocks every row on the fleet, not only OF3T.

### D214. The trunk's headline 6.402x is CROSS-FRAME, the honest in-frame figure is 2.2341x, and the trunk's own scope bar is too loose to close the model clause it feeds. FOUND at pass 359 by the orchestrator, recomputed from the graded artifact's own reconciliation block. UNFIXED as a target, CORRECTED as prose.

Two separate errors, both in numbers this campaign has been quoting, including in my own `GAP:`
and in both briefs dispatched at pass 358.

**One: the trunk section of the graded artifact is a cross-frame comparison.** `of3t-frame384`
measured the two float64 references against each other over the same 2,736 trunk tensors:
`ref_f64_n384` against `grads_f64_043` reads **1.8416532191335382** mass-weighted rel L2 at cos
**0.36514** and norm ratio **1.46424**, both sides float64 with no device op anywhere — *"nothing
inside the port can reach that distance."* The trunk's device file is a capture-driven crop-384
walk; the model reference is the full-model float64 gradient. The other nine sections do not have
this problem. So:

    trunk, cross-frame (what the artifact and my GAP quote)   2.0151163033   vs their 0.3147698294
    trunk, IN FRAME, the charter's exact shape (frame384)     1.0293953378
    ours / upstream's own bf16, in frame, both vs f64         0.8354121633 / 0.3739383921 = 2.2341x

**The honest statement is 2.2341x upstream's own bf16, not 6.402x.** And 6.402x was cross-REFERENCE
as well as cross-frame: it divides ours-vs-their-bf16 by theirs-vs-float64. Same-reference
cross-frame is 2.1595271217 / 0.3147698294 = 6.861x.

**Two: the correction does not save the clause, and `of3t-frame384`'s own figure for it is off.**
That row proposed the in-frame clause reads **0.3092307842**, 2.0986x the bar. It composed from the
published float64 mass shares; the artifact's own reconciliation says that weighting is wrong
(`rel_difference_published` 0.0368) and that the arm's own `ref_sq` reproduces the headline
exactly. Re-pooled from `ref_sq` — control reproduces the published 0.5201243840984896 at relative
difference **0.0** before any rung is quoted:

    trunk section reads                             model scope     vs bar 0.14735268326440318
    2.0151163033  today, cross-frame                0.5201243841    3.5298x  fails
    1.0293953378  in frame                          0.2785749654    1.8905x  fails
    0.5268825373  its own in-frame A26 bar          0.1653149249    1.1219x  FAILS
    0.3147698294  upstream's own bf16               0.1259060033    0.8545x  passes
    0.0000000000  exact                             0.0973798898    0.6609x  passes

**The load-bearing row is the third.** A trunk that hits the bar `of3t-frame384` derived for its own
scope still leaves GRADIENTS failing at 1.1219x. **The trunk must reach ≤ 0.4361680548 against
upstream's own bf16 for the clause to pass**, which is 0.828x its own scope bar. It sits at
1.0293953378 in frame, so the factor still to find is **2.360x**, not the 4.62x the cross-frame
figure implies and not the 13.0x or 6.4x earlier passes recorded.

**Sensitivity, because the substitution holds the model-arm weight while changing the reading.**
The trunk's `ref_sq` is 0.669159435898655 against 9.739910148597525 for the other nine. Over a 6x
weight band the in-frame clause reads 1.4395x to 2.9541x the bar and the threshold moves 0.5990 to
0.2791 — **it fails at every point**, so the verdict does not turn on the weight and only the
magnitude does. An in-frame re-pool that carries its own `ref_sq` is what would make the middle
column exact; this is an estimate with the weight held, and says so.

**No verdict flips.** GRADIENTS failed before this correction and fails after it, which is what
makes publishing a correction that moves toward passing defensible here rather than convenient.
What changes is the distance the campaign tells Moritz it has to go, and the target the two live
rows are working to.

### D215. Every per-section ratio the campaign quotes is CROSS-REFERENCE, and it hides that we beat upstream's own bf16 step on five of ten sections. FOUND at pass 360 by the orchestrator. FIXED as prose in the summary; the unowned second object it exposes stays open.

`PROVES:` and `GAP:` have been saying *"nine of ten sections between 0.945x and 2.176x of
upstream's own bf16 step, most of them at or below 1.1x."* Every one of those ratios divides
**ours-vs-upstream's-bf16** by **theirs-vs-float64** — two different references. The honest
same-reference ratio puts both sides against float64, and the two columns disagree
systematically, always in the direction that makes us look ordinary:

    section                                  pct   ours/f64  their/f64  SAME-REF  as quoted  their share
    diffusion_module.diffusion_transformer  43.62    0.1167     0.0578    2.019x     1.878x        53.2 %
    diffusion_module.diffusion_conditioning 36.95    0.0079     0.0625    0.126x     1.034x        96.7 %
    pairformer_stack                         5.83    2.1595     0.3148    6.861x     6.402x        15.6 %
    diffusion_module.atom_attn_enc           4.73    0.0727     0.0709    1.026x     1.395x        71.7 %
    aux_heads                                2.84    0.0023     0.2394    0.010x     0.986x       101.4 %
    diffusion_module.atom_attn_dec           1.28    0.0198     0.0500    0.395x     0.963x       103.8 %
    msa_module                               1.23    0.0892     0.1788    0.499x     1.055x        94.7 %
    diffusion_module.layer_norm_s            0.98    0.0386     0.0310    1.246x     2.176x        46.0 %
    diffusion_module.layer_norm_a            0.27    0.0094     0.0370    0.255x     0.945x       105.8 %
    diffusion_module.linear_s                0.24    0.0861     0.0654    1.315x     1.718x        58.2 %

`their share` is upstream's own distance to float64 as a fraction of our measured distance to
their bf16. It is above **94 %** in five sections and above **100 %** in three, which is what
being closer to float64 than they are looks like from inside a bar that grades agreement.

**Five of ten sections are strictly better than upstream's own training step against the float64
ideal**, by 2.0x to **100x** (`aux_heads`: we read 0.0023 where they read 0.2394). The quoted
column records that section at 0.986x — indistinguishable from parity.

**The protocol already had the rule and the reporting never applied it.** A26 point 3: *"Do not
spend effort reducing ||e|| below ||t|| in pursuit of agreement. Past that point the reading is
dominated by ||t|| and improvements are invisible. **Say so** rather than optimising into it."*
The campaign did the optimising and then did not say so.

**What it does NOT change.** The clause and its bar are same-reference on both sides
(`renorm_vs_UPSTREAM_BF16` 0.5201243841 against `A26_reachable_bar_vs_their_bf16` 0.1473526833),
and A26 already grants sqrt(2)x the floor for exactly this reason, so the GATE is not unfair and
no verdict moves. Pooled over the nine non-trunk sections with the arm's own `ref_sq`, ours reads
**0.0834300442** against upstream's **0.0754253407** vs float64 — **1.1061x, parity** — because
`diffusion_transformer` at 43.62 % of the mass is 2.019x and outweighs the five wins. Against the
clause's own reference those nine read **0.1006694647** against the 0.1473526833 bar, **0.683x
and passing on their own.**

**And it exposes a second object nobody owns.** `GAP:` says *"ALL of it is `pairformer_stack`"*,
which is true arithmetically — the trunk alone can close the clause. But same-reference,
**`diffusion_transformer` is 2.019x upstream's own bf16 on 43.62 % of the model's gradient mass**,
the largest single share in the model, and the cross-reference column showed it as 1.878x, which
read as unremarkable next to nine siblings. A26's own worked example still cites this module "at
129x"; it is 2.019x today, so that example is 64x stale.
