# The "+2.3 % regression" on main is not a regression

The current tree measures **14.8813 s** at 512 aa at a pinned 1350 MHz (`c10-bare-baseline`,
`bc66f7d6d`, 16 folds). Commit `0df13ad9` measured **14.554 s** pinned, same fixture, same config,
same timer boundary (`time.perf_counter` around `state.predict_one` in both harnesses). The
+0.327 s, +2.3 %, was flagged as a possible regression and handed to `c10-fixed-cost` to check.
It is neither necessary nor possible to spend a chip on it.

## The whole functional diff is two changes, and neither can fire

`git log 0df13ad9..origin/main -- tt_bio/` is five commits. Three touch only comments, citations
and file locations. The diff of `tt_bio/tenstorrent.py` is 33 lines, of which two are functional:

**1. `_SDPA_FUSED_LARGE_S` flipped from `False` to `True`** (`bb3bc4afe`). Its gate in
`_tri_att_sdpa_at` is

    if (_SDPA_FUSED_LARGE_S and q_len == k_len and q_len % SDPA_CHUNK_TILE == 0
            and q_len > _triatt_sdpa._Q_SPLIT_MAX_S):

and `_Q_SPLIT_MAX_S` is 1024 (`tt_bio/triatt_sdpa.py:88`). At 512 aa the triangle attention's
`q_len` is 512, which is not greater than 1024, so **the route is unreachable at this size by
construction** — it can neither cost nor save anything here. That is consistent with its own
landing note, which measured 1.1856x on a 1536 aa fold. The flag is correctly defaulted: a real
win above the cap, a no-op at 512 and 298 aa.

**2. `aiclk.engage()` was removed from `_open_and_init_device`.** `0df13ad9` carried the opt-in
clock hold in-tree; main does not. Both measurements ran at a verified, during-sampled 1350 MHz —
`0df13ad9` through its own hold, the baseline through `force_aiclk.py` — so neither fold's clock
depended on this call.

## So what is the 0.327 s?

Cross-run variance, and we measured how big that is. `../fixed_cost/clock_label_error.py` found
that a model fitted inside one clean interleaved session predicts folds from other sessions with a
**mean error of 0.364 s**. The gap between these two numbers is 0.327 s. It is *smaller than the
error of the comparison itself*, which is exactly why this campaign's rule is that a cross-run
comparison is not evidence.

**Nothing should be spent chasing it**, and `c10-fixed-cost` was told so. If anyone still wants the
answer, the only instrument that can give it is two checkouts alternating in one interleaved
session — which is a real experiment with a real cost, for a difference that is within noise.

This note changes no code and measures nothing. It reads two commits and one threshold.
