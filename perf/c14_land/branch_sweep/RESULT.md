# Branch axis of the c14-land-tail sweep, 2026-09-20

Input: unshipped.sh, 82 concluded-but-unmerged tt-bio branches (branches.txt).

## 1. Does any of them propose a flag default change?
origin/main tip: 57 env_flag defaults

BRANCHES THAT PROPOSE A DEFAULT CHANGE (against their own merge-base)
========================================================================
  none

proposing a default change : 0
no default change at all   : 82
unreadable                 : 0

## 2. Could any change shipped behaviour without a flag?
unshipped branches with any tt_bio/ delta vs their merge-base : 51
  of those, purely ADDITIVE (zero deletions)                  : 2
  of those, containing DELETIONS (can change shipped lines)   : 49

 deletions  insertions  files  branch
       156       16663     53  wk/of3t-confhead
       155       16628     52  wk/of3t-reopen
       155       16628     52  wk/of3t-diffusion
       155       16532     52  wk/of3t-gradients
       155       16485     52  wk/of3t-entity
       153       16382     51  wk/of3t-l1
       147         304      5  wk/trix-bankspread

## Verdict for this row

**The branch axis is empty. No unshipped branch carries a landable lever.**

Sweep 1 is the load-bearing one: across 82 concluded-but-unmerged branches, **zero** propose a
change to any `env_flag` default. This campaign gates one lever behind one flag by its own
discipline, so that is the axis its levers live on.

Sweep 2 covers what sweep 1 cannot: a branch can change shipped behaviour with a constant, an
unconditional fast path or a dispatch-table row, none of which touch a flag. 51 branches have a
`tt_bio/` delta and 49 of those contain deletions, so the deletion count alone does not narrow it.
What narrows it is ownership: the large ones are whole-feature branches belonging to other
campaigns — `of3t-*` (16k insertions each, the OpenFold3 training work), `train-*` (5.5k each) and
`trix-*` — and those are their own rows' to merge, not this row's. This row lands measured levers;
it does not merge another campaign's feature branch.

Two are in C12/C14 scope and neither is a candidate:

    wk/c12-diffusion-head-major   97 deletions    VERDICT: NO-GO
    wk/c12-reblock-delete         40 deletions    VERDICT: NO-GO

## What this closes

The brief told this row to sweep `state/` and the open `wk/*` branches for every lever with a
fold-level measured value that is not a default on `origin/main`. The flag axis was enumerated by
the orchestrator (43 default-off flags on main, the promising ones already fold-tested and
correctly off) and re-screened by this row's own firing census in pass 1, which dropped five of
eight candidates on executed counters. The branch axis is now closed here, and it is closed
wider than the earlier check that reported a single stranded branch: 82 branches, all of them.

So the landing tail on the Boltz-2 512 aa fold is **`TT_BIO_APB_CONCAT_HEADS` and nothing else**.
`TT_BIO_MM_SHORT_M_BW` was the other candidate and this row closed it NO-GO as a default on a hard
stop (its output depends on the core grid). That is a finding, not a shortfall: it means no further
hunting is owed on this axis and the row's remaining value is entirely in landing APB.

## Limit of the method, stated so it is not over-read

Sweep 1 reads `env_flag(NAME, True|False)` literals out of six files. A default set by any other
construction — computed at import, read through a helper, or living in a file not in that list —
would not be seen. The six files are where every flag in this campaign's record lives, and the
origin/main tip scan finds 57 of them, but this is a text scan and not an execution.
