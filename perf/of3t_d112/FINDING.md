# D112 — the 0.4.3 reference is restored, and it turns four of D8's six arms around

## 1. The reference

`grads_f64_043.pt`, sha256 `1d4ea9225f854afe06a8adedcb5f8aa1c53656fbf8cefdaa3a451d0327895cc4`
— the value recorded inside every 0.4.3 instrument-A arm on `wk/of3t-rebase` — survived D112 at
`/home/ttuser/of3t_refprec/bundle_ref/`, rebuilt there after the prune and never connected back
to the arms that name it. The 0.4.3 package tree survived in three places. All four were inside
a directory named after a **concluded** slug, which is the condition that killed
`/home/ttuser/of3t_rebase/`.

Restored to `/home/ttuser/of3t-campaign-refs/` (qb2) and `~/.coworker/artifacts/of3t-refs/` (pc),
with `MANIFEST.json`, `EXPECTED_DIGESTS.json` and a check that exits non-zero when a tree moves.
`REFS_README.md` states the three prune mechanisms and why neither path is reachable by any of
them. A fresh `pip download openfold3==0.4.3` on 2026-09-21 reproduces the campaign's recorded
tree digest `1b27f575…` (293 `.py` files), and 0.5.0 reproduces `092fb575…` (321), both under
`of3t-trunk043ref`'s own A24 rule — so the package half is rebuildable from upstream and
checkable, not merely copied.

## 2. What actually happened to D8

`of3t-rebase` **did** run the substitution D8's pass-90 caveat asks for, and filed
`perf/of3t_rebase/score_arms_043.json`. D112 then pruned its worktree, the row concluded, and the
result never reached the defect it answers. D8's entry still carries the 0.5.0 table and still
carries the caveat as open. Nothing was lost — it was never read.

The arms below are that file's inputs, rescored here with
`perf/of3t_orchestrator/revision/d8_vs_endnode.py` unchanged. The crop-64 column reproduces
`score_arms_043.json` digit for digit, which is the control on the scorer; N384 and the
sub-module decomposition are new.

## 3. D8's pass-90 six-arm table, at both references

`scale_pair_bias` is FALSE on both sides, `transpose_bias` is the arm, crop 64, same checkpoint,
same 52 A14-kept tensors. `replay` is `capture.block_grad_vs_bundle.worst_rel`: the instrument
recomputing this block's gradients from the captured boundary and comparing them to the bundle it
is about to score against.

| arm | ref | median | | over bar | norm share | replay | worst tensor |
|---|---|---|---|---|---|---|---|
| block 0 SHIPPED | 0.5.0 | 0.0648 | FAIL | 36/52 | 23.7 % | 2.2560 | `tri_att_end.layer_norm.weight` |
| | **0.4.3** | **0.0121** | **PASS** | 9/52 | 5.1 % | **0.0000** | `attn_pair_bias.layer_norm_z.weight` |
| block 0 tb-off | 0.5.0 | 0.0115 | PASS | 7/52 | 1.9 % | 2.2560 | `attn_pair_bias.linear_z.weight` |
| | **0.4.3** | **0.0740** | **FAIL** | 37/52 | 24.1 % | **0.0000** | `tri_att_end.linear_z.weight` |
| block 23 SHIPPED | 0.5.0 | 0.0930 | FAIL | 46/52 | 64.1 % | 1.9709 | `attn_pair_bias.layer_norm_a.weight` |
| | **0.4.3** | **0.0192** | **PASS** | 15/52 | 25.0 % | **0.0000** | same |
| block 23 tb-off | 0.5.0 | 0.0174 | PASS | 12/52 | 53.7 % | 1.9709 | same |
| | **0.4.3** | **0.1008** | **FAIL** | 46/52 | 70.0 % | **0.0000** | same |
| block 47 SHIPPED | 0.5.0 | 0.4270 | FAIL | 52/52 | 100.0 % | 1.5629 | same |
| | 0.4.3 | 0.2013 | FAIL | 52/52 | 100.0 % | **0.0000** | same |
| block 47 tb-off | 0.5.0 | 0.4004 | FAIL | 52/52 | 100.0 % | 1.5629 | same |
| | 0.4.3 | 0.2386 | FAIL | 52/52 | 100.0 % | **0.0000** | same |

**Four of six arms cross the median bar, and the two arms swap roles.** D8's pass-90 conclusion
— *"At blocks 0 and 23 the median goes FAIL to PASS once both sides compute the same function"*,
with the shipped arm failing and `tb-off` rescuing it — is the mirror image of what the correct
boundary says. Against 0.4.3 the **shipped** orientation passes at blocks 0 and 23 and `tb-off`
fails. The magnitudes barely move (0.0648/0.0930 against 0.0740/0.1008); it is the attribution
that inverts. `of3t-trunk043ref` reached the same conclusion from the forward side — the flip
`of3t-trunkfwd` flagged would have been a regression — and this is the gradient side of it.

**The ending-node localisation does not survive.** D8's pass-90 headline is *"The worst tensor in
the shipped block-0 arm is `tri_att_end.layer_norm.weight` — the ending node, the one sub-module
D23's trunk change modifies. The instrument pointed at the cause for forty passes."* Against
0.4.3 the worst tensor in that arm is `attn_pair_bias.layer_norm_z.weight`, and `tri_att_end`
appears as the worst tensor of the arm that now FAILS instead.

**The published table was taken on a boundary the instrument rejected.** `replay` reads 2.2560 at
block 0, 1.9709 at block 23 and 1.5629 at block 47 against a 5.0e-02 per-tensor bar — the
captured boundary does not reproduce the bundle it is scored against. Every 0.4.3 arm reads
exactly 0.0000. So this is not one measurement read against two references: the 0.5.0 side failed
its own validation, and D8's entry does not mention it. The table should be withdrawn, not
corrected line by line.

## 4. The full 384-token window at block 0 — new

`score_arms_043.json` covers crop 64 and leaves its `stacks` entry at `against_0.4.3: null`. The
per-block n384 arms exist on both sides and pair cleanly (same 384 tokens, same `s_norm`,
`scale_pair_bias` false on both):

| arm | ref | median | | over bar | norm share | worst |
|---|---|---|---|---|---|---|
| block 0 SHIPPED, 384 tokens | 0.5.0 | 0.0672 | FAIL | 37/52 | 23.9 % | 1.3071 `tri_att_end.layer_norm.weight` |
| | **0.4.3** | **0.0118** | **PASS** | 9/52 | 7.9 % | 0.9377 `attn_pair_bias.layer_norm_z.weight` |

So the flip is not an artefact of the crop. The uncropped window says the same thing.

## 5. D8's pass-47 sub-module decomposition — new, and it is the ranking that breaks

D8 calls this table *"mechanism-shaped, and the sharpest statement the campaign has about D8"*.
Recomputed on the n384 pair, over the tensors present and non-zero-reference on both sides:

| group | n | 0.5.0 | 0.4.3 | ratio | over bar 0.5.0 | over bar 0.4.3 |
|---|---|---|---|---|---|---|
| `tri_att_end` | 8 | 0.3852 | **0.0157** | **0.041x** | 8/8 | **1/8** |
| `attn_pair_bias` | 6 | 0.1042 | **0.1313** | **1.260x** | 6/6 | 5/6 |
| `tri_att_start` | 8 | 0.0886 | 0.0271 | 0.306x | 8/8 | 2/8 |
| `single_transition` | 5 | 0.0207 | 0.0215 | 1.039x | **0/5** | **1/5** |
| `pair_stack`, the rest | 25 | 0.0550 | 0.0079 | 0.145x | 15/25 | **0/25** |

**The ordering is not preserved and the mechanism claim goes with it.** D8's reading is *"The one
sub-module with no attention and no pair-track coupling is the only one that passes, and the
failure is graded by pair/attention involvement rather than by tensor size or depth."* Against
0.4.3, `single_transition` is no longer the only passer — it is the only group besides
`attn_pair_bias` that gets *worse*, and it now has one tensor over the bar while `pair_stack, the
rest` has none of twenty-five. `tri_att_end` moves from worst to second best, 24.5x better. What
survives is one group: `attn_pair_bias` at 0.1313, 5 of 6 over the bar, and the only sub-module
that the correct boundary makes worse rather than better.

**Pass 85's asymmetry inverts.** End over start reads 4.35x on the median and 3.52x on the worst
at 0.5.0, and **0.58x and 0.76x** at 0.4.3. The claim *"composition hits the ending-node axis
about three times harder than the starting-node axis"* is a property of the 0.5.0 reference.

**Pass 47's own source arm had already withdrawn itself.**
`perf/of3t_gradients/instrument_a_bundle_block0.json` — which reproduces pass-47's published
0.3838 / 0.1470 / 0.0865 / 0.0680 / 0.0212 exactly — carries a `superseded_by` field reading
*"Taken against the withdrawn train-mode tape. Its reference side is one dropout draw (D18),
whose own draw-to-draw floor is median 0.550 — an order of magnitude above the 5.0e-02 bar."* A
0.550 floor is 1.4x the largest number in the table. That is independent of any boundary
version, it was written in the file, and D8 kept quoting the table for 150 passes.

## 6. Which of D8's numbers do NOT change

- The **pass-222 restatement**, which is D8's live statement: 2,713 of upstream's own 2,736
  tensors over the bar, 1.65x by mass, 22 of 48 blocks outside A26 holding 66.67 %, worst block
  46 at 2.8742x on 41.333 % of the mass. Already 0.4.3 — `perf/of3t_apbgrad/SCOPE_c64.json` names
  `reference.float64.tree = /home/ttuser/of3t_trunk043ref/of3pkg043` and `how_built = "upstream
  0.4.3, every parameter and activation float64"`. Unaffected.
- The **pass-232 closure** by `of3t-lnaffine`: 0.849x / 0.837x / 0.849x / 0.747x of upstream
  **0.4.3**'s own bf16 floor. Already 0.4.3. Unaffected.
- **Block 47 fails at every reference and every flag setting** — 52/52 over the per-tensor bar
  and 100 % of the norm in all four block-47 arms. Its magnitude is reference-dependent (0.4270
  at 0.5.0, 0.2013 at 0.4.3) and its verdict is not. This is where D8 is real.

## 7. The brief's own premise, corrected

The brief says the substitution is *"worth a factor of 30,245 at the LayerNorm island"* for D8.
That factor is **D120's**, from `of3t-fp32islands`, on LayerNorm's affine and input gradients as
single-op islands against upstream's own reference — 4.256869e-03 against 1.407498e-07 at 0.5.0.
No D8 arm reads it, and D120 already records the 0.4.3 side: it inverts, and we are 1.55x more
accurate on d(gamma). The figure was never D8's to move. What the substitution is worth to D8 is
in §3 to §5: four verdict flips, an inverted attribution, a destroyed sub-module ranking, and one
sub-module left standing.

## 8. A provenance trap left in place

`perf/of3t_orchestrator/block47/instrument_a_bundle_043_block{0,23,47}_crop64_tb{shipped,off}.json`
are **byte-identical** to `perf/of3t_rebase/mispinned_spb_on/`, the six arms `of3t-rebase`
quarantined because they moved the pair-bias convention and the upstream revision together. The
row moved them into a directory whose name says so, explicitly *"because a label beside a wrong
number does not stop the number being quoted"*. The orchestrator then copied the same bytes into
its own directory under the plain name. This row's first pass at the table used them, which is
how it was found; `arms/mispinned_spb_on/` holds them here under a name that says so. The valid
0.4.3 arms are `instrument_a_bundle_043spb_*`, and they are what §3 uses.

## Files

| file | what |
|---|---|
| `d8_at_043.py`, `D8_AT_043.json` | the three tables, scorer unchanged from `d8_vs_endnode.py`, comparability checked per pair |
| `arms/` | the twelve crop-64 arms and the n384 pair, extracted from git |
| `arms/mispinned_spb_on/` | the quarantined six, named |
| `make_manifest.py`, `EXPECTED_DIGESTS.json` | the reference manifest and its check; negative control exits 1 |
| `tree_digest.py` | `of3t-trunk043ref`'s A24 rule, verbatim, so the digests are comparable |
| `REFS_README.md` | deployed to both reference directories as `README.md` |

No card was used and none was needed: every arm was committed before the prune, and the
substitution is a reading. No perf figure is claimed, so no AICLK is quoted.
