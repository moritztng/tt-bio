# The land-standing backlog, audited against git

2026-09-26. Every candidate this row's charter names, with the command that settles it. The short
version: **none of them is landable, and most were already shipped.**

| candidate | charter says | git says |
|---|---|---|
| `c13-land-first`, 0.4798 s | "long recorded as SHIPPED: 0.0000 s" | branch 0 ahead / 5567 behind, **contained in main**. The 0.4798 s was a two-lever stack and one lever's approval was withdrawn; the survivor, `TT_BIO_DIT_COND_HOIST`, is on main and default-on at +0.2052 s |
| Region T, `TT_BIO_TRIATT_B8` | merged but defaulting off, +0.1750 s | `env_flag(..., False)` on main, and correctly so — `825f18772` closed it as a card-dependence **hard stop**, not an accuracy trade |
| `out_block_h`, 1.2894x | a lever to land | `_pair_proj_program_config` is on main with all three caps non-None (`_PAIR_PROJ_BW=16`, `_NARROW_PROJ_BW=1`, `_PAIR_PROJ_L1_BW=16`). **Shipped, default-on, four call sites** |
| `TT_BIO_TRIATT_NARROW_Q_FALLBACK`, 16.7 s | "needs a Blackhole A/B at more than one size" | `env_flag(..., True)` at `tenstorrent.py:1717`. **Already default-on**; `234283b50` even re-took the 896 aa cell and found its 9.7 % floor was the harness |
| `pvx-custchart`, five commits | carries corrections other rows cite wrongly | **merged** as `b51dceda4`; the branch is 0 ahead / 5312 behind |
| the C14 stack and bfp8 region T | "PARTIAL with gate arms outstanding" | `bfp8-l1-chunking`, `-orchestrator`, `-region-t-fold`, `-region-t-land`, `-sdpa-unlock` are all **contained in main**. The three that are not (`-accuracy-envelope`, `-revive-accuracy-killed`, `-z-accumulator`) are 1-3 ahead and **5596-5665 behind**, which is a record of a pass, not a candidate |
| `TT_BIO_TRIATT_DIVIDING_K` | (added later, by handoff) | default off, and it should stay there — see `DIVIDING-K-VERDICT.md`. Reach is one length |

## What this means for the row

The charter opens with "dozens of `wk/*` branches sit ahead of `origin/main`" and treats that as a
queue. It is not one. Ranked by commits BEHIND rather than ahead, every branch here is either
contained in main already or thousands of commits stale, which is the same finding
`a-branchs-ahead-count-is-not-a-merge-signal` records at a larger scale.

So the row is not idle for want of effort; it is idle because the levers its charter names were
landed by the campaigns that measured them. Future candidates have to come from live campaigns
handing them over, not from this list.

## The one thing still genuinely open

`TT_BIO_TRIATT_DIVIDING_K`, and it is open on evidence rather than on a merge. Reach is settled at
one length (832), kernel accuracy is favourable, structural accuracy needs a confident 832-token
fixture this box cannot produce, and the speed term is unpriced. Details and the exact next
measurement are in `DIVIDING-K-VERDICT.md`.
