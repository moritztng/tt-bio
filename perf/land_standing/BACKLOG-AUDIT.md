# The land-standing backlog, audited against git

2026-09-26. Every candidate this row's charter names, with the command that settles it. The short
version: **none of the charter's named levers is landable, and most were already shipped.** The one
that was genuinely open has since been measured, flipped and gated — see the bottom of this file.

| candidate | charter says | git says |
|---|---|---|
| `c13-land-first`, 0.4798 s | "long recorded as SHIPPED: 0.0000 s" | branch **0 ahead / 5745 behind**, contained in main. The 0.4798 s was a two-lever stack and main's own comment says it "does not describe any default". **hoist** (`TT_BIO_DIT_COND_HOIST`) is default-ON since 2026-09-18 at **+0.2052 s**, already shipped by the row that measured it. **silu** (`TT_BIO_UNFUSED_SILU`) is default-OFF and **held there by Moritz on measured accuracy** — Protenix-v2 loses 0.05088 / 0.07210 CA-lDDT vs 1HCL, arms fully rank-separated over four seeds. Accuracy-failing levers are his call, so the +0.2336 s stays unclaimed |
| Region T, `TT_BIO_TRIATT_B8` | merged but defaulting off, +0.1750 s | `env_flag(..., False)` on main, and correctly so — `825f18772` closed it as a card-dependence **hard stop**, not an accuracy trade |
| `out_block_h`, 1.2894x | a lever to land | `_pair_proj_program_config` is on main with all three caps non-None (`_PAIR_PROJ_BW=16`, `_NARROW_PROJ_BW=1`, `_PAIR_PROJ_L1_BW=16`). **Shipped, default-on, four call sites** |
| `TT_BIO_TRIATT_NARROW_Q_FALLBACK`, 16.7 s | "needs a Blackhole A/B at more than one size" | `env_flag(..., True)`. **Already default-on**; `234283b50` even re-took the 896 aa cell and found its 9.7 % floor was the harness |
| `pvx-custchart`, five commits | carries corrections other rows cite wrongly | **merged** as `b51dceda4`; the branch is 0 ahead / 5312 behind |
| the C14 stack and bfp8 region T | "PARTIAL with gate arms outstanding" | `bfp8-l1-chunking`, `-orchestrator`, `-region-t-fold`, `-region-t-land`, `-sdpa-unlock` are all **contained in main**. The three that are not are 1-3 ahead and **5596-5665 behind**, which is a record of a pass, not a candidate |
| `TT_BIO_SDPA_WIDE_K_UP`, 1.1554x | (not in the charter; found by scanning main for default-off levers) | **EXAMINED AND REJECTED on measurement.** It fires on every call — 560 of 560 on a Boltz-2 fold at 384, moving `q384 k192` to `q384 k384` — and the fold A/B reads it **0.281 s SLOWER** at the median against a **1.568 s A/A floor**. A 1.1554x op ratio became nothing at the fold |
| `TT_BIO_TRIATT_DIVIDING_K` | (added later, by handoff) | **MEASURED, FLIPPED ON and GATED — see below.** This row's earlier "it should stay off" is withdrawn |

## What this means for the row

The charter opens with "dozens of `wk/*` branches sit ahead of `origin/main`" and treats that as a
queue. It is not one. Ranked by commits BEHIND rather than ahead, every branch here is either
contained in main already or thousands of commits stale, which is the same finding
`a-branchs-ahead-count-is-not-a-merge-signal` records at a larger scale.

So the row was not idle for want of effort; the levers its charter named were landed by the
campaigns that measured them. **Candidates come from two places instead: live campaigns handing
them over, and scanning main itself for default-off levers with a number attached** — which is how
`TT_BIO_SDPA_WIDE_K_UP` was found, examined and rejected in one pass.

## `TT_BIO_TRIATT_DIVIDING_K` — this file's earlier verdict is WITHDRAWN

An earlier version of this table said *"default off, and it should stay there"*, and the section
below it said the structural accuracy *"needs a confident 832-token fixture this box cannot
produce"* and that *"the speed term is unpriced"*. **All three statements are now false**, by this
row's own later measurements, and a document that tells the next reader the opposite of the truth
is worse than no document:

- **The speed term is priced**: **+50.999 s, 1.6351x** at 832 tokens, A/A floor 1.306 s, effect 39x
  the floor, AICLK sampled DURING every leg at 1350 MHz with arms interleaved.
- **The box did produce the confident fixture.** The blocker was never the box, it was that every
  arm had been folded `--single_sequence`. The same tiled CDK2 reads pLDDT 0.503 single-sequence,
  0.602 at MSA depth 36 and **0.882 at depth 513**; the deep alignment was extracted from a
  depth-13232 search already on this host.
- **Accuracy clears on every instrument**: 0.450148 A against the 0.60 A bar, a 1.974757 A seed
  floor beside it, a control of exactly 0.000000 A, and both confidence heads favourable. Changing
  the lever *and* the seed moves the structure 1.948630 A — no more than the seed alone.
- **Flipped and gated**: `_TRIATT_HIFI_DIVIDING_K_DEFAULT = True`, ten gate arms green at the tip
  and the two most relevant re-green on the merge tree.

Full evidence and the history of the withdrawn readings are in `DIVIDING-K-VERDICT.md`.

## `TT_BIO_TRIATT_FUSE_QKV` — CLOSED, unreachable on shipping defaults

Found only by widening the lever census past `tenstorrent.py`. "ROOF Phase A: the qkv projection
moved inside this kernel", moving 603.9 -> 402.7 MB at boltz-2s 512 aa triangle attention, held

## `TT_BIO_TRIATT_FUSE_QKV` — CLOSED, unreachable on shipping defaults

Found only by widening the lever census past `tenstorrent.py`. "ROOF Phase A: the qkv projection
moved inside this kernel", moving 603.9 -> 402.7 MB at boltz-2's 512 aa triangle attention, held
off as "a release-gated arm" — a condition rather than a refusal, so it looked like this row's
shape.

**Measured, then explained.** On a real Boltz-2 fold at 384 tokens the kernel's own counter reads
`FUSE_REJECTS = {'qkv_already_fused_with_gate': 560}` in BOTH arms — 560 of 560 calls decline, and
the CIF digest is identical across six legs.

**The source says why, exactly.** `_tri_att_fused_qkv_sdpa` runs only when `qkv is None` after
`_fused_qkvg`, and `_fused_qkvg` returns `(None, None)` only when the attention is `biased`, has
no `qkvg_weight`, or an L1 qkv config pre-empts it. `qkvg_weight` is built whenever
`not self.subtile and _dtype() == ttnn.bfloat16`. So on shipping defaults the broader qkv+gate
fusion always takes the call and this lever has nothing left to fold: a later fusion superseded
it.

**Its remaining reach is subtile attention, a non-bf16 operand dtype (bfp8), or an L1 qkv config.**
None is a shipping default, and a win reachable only inside another default-off regime does not
reach users — which is this row's whole charter. Closed.
