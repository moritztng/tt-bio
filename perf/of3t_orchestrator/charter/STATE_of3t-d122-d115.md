# of3t-d122-d115 — the campaign's own instruments

Pass 270. CPU only on pc, no card opened, no device work of any kind. Every number here is read
off a committed artifact or off a git history; none is a timing, so no clock is quoted.

FINDING: both defects are worse than filed, in the same direction — each guard was fixed for the
one case that had already happened and left blind to the cases that had not. A third instrument
defect fell out of the work: eleven row specs in the DONE_CHECK carry a `req` pattern no document
can match, so those rows could not conclude whatever they wrote.

**D122.** The filing says "the campaign's terminating condition is a keyword test on its own
prose" and pass 220 read that as being about GO condition 5, the GAP clause. It is about all
five. Conditions 1-4 (THEIR-TEST, GRADIENTS, TRAJECTORY, COVERAGE) are four regexes over four
fields of the orchestrator's own state doc, and their exposure is worse than condition 5's,
because condition 5 at least executes on every compose while **conditions 1-4 have never
executed against live content at all** — no orchestrator document has ever carried a THEIR-TEST,
GRADIENTS, TRAJECTORY or COVERAGE field, so `_charter_gate`'s first four clauses first run on the
pass that ends the campaign. `charter_is_a_keyword_test.py` demonstrates rather than argues it: a
GO document whose four charter fields contain no figure, no artifact and no file name — only the
English the regexes look for, plus an explicit admission that nothing was measured — is
**ACCEPTED by all four**, while all four artifacts say the charter is unmet. 4 of 4 come apart.

**D115.** The pass-196 reverse check fires on one shape only: ledger dead, GAP label says
UNFIXED. Two other shapes exist and the check cannot see either. Over the **208 published
revisions** of `perf/of3t_orchestrator/record/ORCHESTRATOR.md`:

| direction | what it is | occurrences |
|---|---|---|
| A | ledger dead, GAP says UNFIXED | the pass-196 check; 0 since |
| B | ledger UNFIXED, GAP says a dead word | **0 in 208 revisions** |
| C | both dead, but they assert different things | **3 defects, 181 revision-hits** |

C is the one that mattered, and a word-level check cannot see it, because CLOSED and REFUTED are
both "dead". They do not say the same thing: CLOSED tells a reader the campaign had a bug and
dealt with it, REFUTED tells them the bug was never real. What drifted while the guard was blind,
with the ledger status each disagreed with:

- **D8**, 82 revisions — GAP `(CLOSED pass 232)`, ledger REFUTED. Still live today.
- **D85**, 70 revisions — GAP `(WITHDRAWN, mine)`, ledger FIXED. No longer in GAP.
- **D52**, 29 revisions — GAP `(FIXED this pass)`, ledger RECORDED, which since pass 241 says in
  so many words "a finding, not a defect". Still live today.

Direction B's zero is reported as a zero. Fixing it is prophylactic, not a repair, and saying
otherwise would be the same overclaim the campaign keeps catching.

CAUSE: **D122** — pass 220 fixed the condition it was standing on. It found the keyword test by
trying to satisfy condition 5, wrote the demonstration for condition 5, and added a ledger clause
beside condition 5. The other four have identical shape and were never read, because nothing had
made anyone read them: a clause that never executes produces no wrong answer to notice.

**D115** — two causes, and the second inverts the obvious reading.

1. `_gap_contradictions` tested `st in DEAD and "UNFIXED" in label`. That is a test for one
   direction written as though it were a consistency check, which is D115's own diagnosis
   applied one level up: the pass-196 fix is itself a one-directional guard.
2. For **D85 the LEDGER is the side that is wrong, not GAP.** Its heading reads
   `### D85. WITHDRAWN — I compared a pre-fix arm against a post-fix arm as though they were
   one. The row was right. FIXED.` — two status words on one heading about two different objects
   (the claim was withdrawn; the row's code was fixed). `status_vocab.statuses_by_defect` takes
   the LAST declaration, so the ledger stores FIXED, and GAP's "WITHDRAWN" was right for 70
   revisions. Sweeping every heading, **11 of 149 declare more than one distinct status**; 9 are
   supersessions where last-wins is correct, and 2 (D84, D85) are two-object headings where it is
   not. So the check must report the disagreement without presuming which side is wrong, which is
   what it now does.

FIX: four files, all additive. The BAR is untouched — every prose clause still stands and GO
still needs all of them, exactly as pass 220's ledger clause was additive. No threshold below is
mine: each is read out of the artifact that carries it or cited to the artifact that published it.

1. `workstreams/_of3t_donecheck.py` — a `CHARTER_EVIDENCE` literal naming, per charter condition,
   the artifact that must exist and the property it must have, and a clause in `_charter_gate`
   that refuses GO on any unmet condition. The literal is data, so the tree-side reader parses it
   rather than copying it. The published evaluation records the sha256 of the literal it came
   from and the gate recomputes that sha, so a condition edited after publication invalidates the
   publication instead of outliving it. That is what makes the clause unreachable by rewording:
   there is no sentence in the state doc that touches it and no stale JSON that satisfies it.
2. `perf/of3t_orchestrator/charter/charter_evidence.py` — lifts the gate's literal, evaluates it
   against the artifacts, publishes to both the tree and `state/of3t/`. It refuses to publish at
   all if its own break control fails, because a gate reading UNMET off a broken evaluator is
   worse than a gate reading a missing file.
3. `perf/of3t_orchestrator/charter/charter_is_a_keyword_test.py` — the demonstration, lifting
   `CHARTER_GO` out of the live gate source so it cannot drift from what runs.
4. `perf/of3t_orchestrator/audit_evidence.py` — the contradiction check made symmetric over all
   three directions, and a second reader that recomputes the charter evidence every compose.
   `status_vocab.py` gains the partition the symmetric check needs: REPAIRED = FIXED / RESOLVED /
   CLOSED / ROOT-CAUSED against NOT-A-DEFECT = WITHDRAWN / REFUTED / RECORDED. A mismatch within
   a group is phrasing and passes; across groups it is a contradiction.

The two live cases are relabelled in GAP, the same way pass 196 relabelled its six:
`**D8 (CLOSED pass 232)**` → `**D8 (REFUTED pass 232, CLOSED as a NON-DEFECT)**`, which is the
wording D8's own second GAP label already used, and `**D52 (FIXED this pass)**` →
`**D52 (RECORDED pass 241 — a finding, not a defect)**`.

Two corrections found by the instruments rather than by me, both recorded because they are the
reason to build instruments at all. My first draft of the COVERAGE condition said 5 of 11
conditional paths fire; the evaluator read the artifact and said 4 of 11, and it is right —
`msa`, `ligand`, `confidence_heads` and `routed_call_census` are covered and the other seven are
not. And the symmetric check's **first real run reported D56 as a contradiction and was wrong**:
D56's label is `(UNFIXED; mechanism REFUTED pass 232, magnitude COLLAPSED pass 233)` and its
ledger status is UNFIXED, so the two agree and the refuted mechanism is commentary. I had
stripped `UNFIXED` out of the label before comparing — correct when the ledger says FIXED, since
FIXED is a substring of UNFIXED, and wrong when the ledger says UNFIXED. Fourth time a guard in
this campaign has been too wide on its first real run, so that shape is now a nuance control
rather than a comment.

EVIDENCE: no numeric claim here is a precision or gradient measurement, so no float64 reference is
owed for one; the float64 discipline instead appears **inside** the fix, as the GRADIENTS
condition's first requirement — `inputs.float64.sha256` must be present, so that condition cannot
be met by scoring a device arm against another device arm. The figures it reads are the campaign's
own, recomputed from the artifacts this pass:

- `perf/of3t_wholemodel/MODEL_shipped.json` — `stats.shipped_vs_FLOAT64.n_over_per_tensor_bar`
  = **678** of 900 measurable tensors over `bars.per_tensor_bar` = 0.05, against a float64
  reference pinned at sha256 `1d4ea922…`; `coverage_total.pct_of_model_compared` = **92.1568 %**
  against the **99.2594 %** structural ceiling `COVERAGE_CEILING_IS_NOT_100.json` measured at
  pass 175.
- `perf/of3t_modeltraj/traj_shipped.json` — `steps` = 20, but
  `scope.pct_of_model_sq_grad_norm` = **36.9462 %** (`diffusion_module.diffusion_conditioning`,
  one section) and `d1.zero_both_sides` = **true**, so its `rel_d` of 0.0 is two stationary
  weight vectors rather than agreement. The condition demands `zero_both_sides` be false.
- `perf/of3t_gradients/coverage_census.json` — `union` **7 of 8** loss terms fire (`bond` has an
  empty mask on 5nw3); `conditional_paths` **4 of 11**.
- `perf/of3t_theirtest/RESULT.json` — does not exist. D2 records why: Lightning dispatches by
  torch device and `tt_bio/` gives ttnn none, so this needs a PrivateUse1 backend with a
  Lightning `Accelerator`. The condition names the file and the four keys the eventual run must
  write, so the row that runs it knows what counts.

**0 of 4 charter conditions met**, which is the correct reading of a campaign whose verdict is
PARTIAL, and which the four prose clauses alone cannot produce.

Controls, all executed this pass:

- charter break control — a synthetic artifact built to satisfy every requirement flips each of
  the 4 conditions to MET, so "all four unmet" is a reading rather than a stuck clause.
- gate exercised directly on the no-measurement GO document: **5 refusals** where the four prose
  clauses had produced none.
- published-evidence negative controls: falsifying one `met` gives
  `GRADIENTS published True, recomputes False`; corrupting `spec_sha256` gives the staleness
  refusal; restoring the file returns the check to ok.
- contradiction probes, one per direction: all 3 fire (a shared probe would have passed on the
  pass-196 code, which is how a one-way check survived a break control in the first place).
- positive control on real data: D8, D52 and D85 replayed verbatim out of the published record —
  3 of 3 caught.
- nuance controls, 4 of them, all clean: `CLOSED as a NON-DEFECT` against REFUTED,
  `UNFIXED — escalation WITHDRAWN` against WITHDRAWN, `UNFIXED; mechanism REFUTED pass 232`
  against UNFIXED, `UNFIXED in effect, fixed in code` against FIXED.

`audit_evidence.py` on the composed tree: **167 confirmed, 3 warnings, 6 drifted**. The baseline
before this pass was 168 / 3 / 3. My changes add one confirmation for the charter recompute and
one for the symmetric contradiction check and introduce no drift; the three extra drifts are the
orchestrator's live counts moving underneath the run (defect total 148 → 149, UNFIXED 54 → 55,
briefs on disk 71 → 75) while it works pass 270 in its own worktree.

**A third instrument defect, found by being refused by it.** This row's own DONE_CHECK rejected
the 12 KB document above with "no line matching: FINDING:". The pattern is
`r"^FINDING:\\s*\\S"`, and in an r-string `\\s` is a literal backslash followed by `s`, so
the gate was asking the document for a backslash. **Eleven row specs** carry it — every one added
after `of3t-fp32islands`: of3t-d116, of3t-d112, of3t-d10-d107, of3t-d117, of3t-d122-d115,
of3t-crop768, of3t-d1-pairbias, of3t-d10d24-unify, of3t-d56-renorm, of3t-d137-tapegate and
c12-lastlever-measure. **44 patterns in total.** None of those rows could conclude whatever they
wrote, and two of them were running on the fleet while this was found; of3t-d117 concludes on the
existing document now that the pattern matches, so it had been finished and held by the gate.

This is the same family as D122 — a gate deciding on something other than the work — and the same
family as the `_STAGE_HINTS` trap, where a row gets a refusal it cannot fix from where it stands
and burns launches on it. A doubled escape is invisible on the page, so the table is now probed
rather than read: `_unsatisfiable_reqs()` builds the canonical line each requirement's own text
asks a row to write, `FIELD: x`, and asserts the requirement matches it. Break-controlled in both
directions — re-breaking one spec makes that row's check refuse with "THIS ROW'S GATE IS BROKEN,
not this row's work" and warns every other row on stderr; restoring it returns all rows to their
content-based verdicts. Spot-checked after the fix: of3t-d117, of3t-memory and of3t-fp32islands
conclude, of3t-crop768 is refused for the honest reason that its state doc does not exist yet,
and of3t-orchestrator is refused for its PARTIAL verdict, which is correct.

Also fixed on the way, because it was one line each and my own row was among the casualties:
`_STAGE_HINTS` had no literal path for 11 state docs including this one, which is the documented
failure where `worker.sh` cannot stage a doc to a remote host and the check fails for a reason no
correct work can fix. The warning is now silent.

Branch `wk/of3t-d122-d115`, based on `origin/wk/of3t` — the worktree arrived based on
`origin/main`, where `perf/of3t_orchestrator/` does not exist, which is D100 exactly. Staged by
explicit path, never by a whole-tree add. Nothing merged anywhere; landing is `land-standing`'s.

Two ledger updates are owed on D122 and D115 and are deliberately left to the orchestrator rather
than written here: it is editing `state/of3t/DEFECTS.md` in its own worktree right now, and two
writers on one file is how this campaign loses a record. Everything needed for both entries is
above.

VERDICT: GO — both instruments are repaired, symmetric, and controlled in both directions, and
the charter now has an evidence condition that says which artifact must exist with which property.
The campaign's own verdict is unchanged at PARTIAL and this pass moves it no closer to GO by
design: it makes GO harder to reach by accident and impossible to reach by rephrasing.
