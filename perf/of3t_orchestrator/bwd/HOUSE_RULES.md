
## HOUSE RULES for the backward sprint — of3t-orchestrator, 2026-09-25 16:2x CEST

Added after dispatch because the four briefs went out without them. A decision that is not in your
brief does not reach you, so it is here rather than in a state doc.

- **ARTIFACTS: `perf/of3t_<your-slug>/`, and nowhere else.** Yours alone. Four rows writing into one
  perf tree is how a merge comes out textually clean and semantically broken
  (`sibling-perf-campaigns-need-namespaced-output-paths`).
- **SCRATCH: `/tmp/of3t/<your-slug>/`.** Never a shared generic name. Two of3t rows collided on
  `/tmp/of3t/state.md` and one row's state doc silently became a verbatim copy of another's —
  another row's branch, instruments and VERDICT, installed under the wrong slug. Nothing errored
  (LEDGER K18).
- **BRANCH: `wk/<your-slug>`, pushed every pass.** I compose the sprint onto `wk/of3t-bwd`; do not
  merge to main and do not compose each other. `wk/of3t` is finished history: verified this pass,
  it is fully contained in `origin/main` (`git merge-base --is-ancestor` clean), so do not branch
  from it or treat it as the campaign's tip. Branch from `origin/main`.
- **`tt_bio/autograd.py` and `tt_bio/taped_ttnn.py` are CONTESTED and moving right now.** BCX is
  rewriting the head-verb backwards in both: `wk/bcx-heads` at `237f53064`, with `wk/bcx-bwdplan`
  building on it this pass. If your fix lands in either file, keep the diff minimal, say so in your
  state doc under its own heading, and expect to re-run BCX's bar rather than trusting a clean
  textual merge — these are exactly the two files where `land-d264` showed a merge error is
  invisible to inference and visible only to a taped arm.
- **Say the AXIS on every number** (chip-to-chip, box-to-box, phase-matched, step-to-step), and the
  clock: a DURING-sampled AICLK and the board class. qb1 p150a and qb2 p300c run the same 512 aa
  fold 17.39 s against 14.59 s at the same 1350 MHz, so a board class is part of a measurement.
- **The target stays honest.** ~8.5x compute / ~11x bandwidth is hardware; ~6-7x of the 58-67x is
  software and that is what this sprint is for. 5x chip-to-chip is below the silicon floor — do not
  promise it, in a state doc or anywhere else.
