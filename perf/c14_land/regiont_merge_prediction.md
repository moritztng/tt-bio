# Pre-registered before the merged-tree census returned (c14-land-tail pass 7, 2026-09-18 21:1xZ)

Written before reading `perf/c14_bfp8/regiont_values_main.json`, so the reading below cannot be
fitted to the result.

`wk/bfp8-region-t-land` was cut at `56278dadb`. `origin/main` is `bd3a5a2df`, and two of the 21
commits in between are in region T's own path:

* `5af25ddb9` keys the SDPA L1-refusal memos by operand dtype. Before it, a bf16 refusal retired a
  chunk that bfp8 fits, so the memo leaked one-directionally against bfp8.
* `d0649566c` adds an upward k rung "gated on what fits at the operand dtype", i.e. a rung that
  exists under bfp8 and not under bf16.

Region T narrows triangle-attention operands to bfp8. Both commits make bfp8 reach k chunks bf16
cannot, so they do not merely coexist with the lever, they change what its ON arm is allowed to
pick. Textually the merge is clean (131+/39- against main, identical to the pre-merge delta), and
a clean textual merge over interacting semantics is exactly the case that needs execution rather
than a syntax check.

## What is predicted

1. The ON arm still serves the fused persistent-mask SDPA and the head-major qkv path with ZERO
   declines. A decline would mean the merge moved the lever off its kernel, which would make any
   later A/B a measurement of a fallback rather than of region T.
2. The BASE arm's 512 aa digest is `2f2f5faae481337e...`, which is what a pristine `origin/main`
   folds. If it is not, the merge changed values on the shipped path and nothing else in this file
   matters.
3. The ON arm's digest is `4bc28f5d918ab65f...`, the T-s0 digest the accuracy envelope scored and
   `bfp8-region-t-land` reproduced on a third card. If it has MOVED, the lever's structure is no
   longer the one carrying the 0.42886 A reading, and its accuracy arm has to be re-scored before
   any flip. Accuracy is inherited only while the structure is identical.

Prediction 3 is the one at risk, and it is at risk in a specific direction: a wider k rung under
bfp8 changes the online-softmax reduction order, which is not bit-exact by construction. So a moved
ON digest would not be a bug, it would be main's new rung firing inside the lever, and it would
still invalidate the inherited accuracy number.

## What this census is NOT

One block, one timed fold an arm, on a host that `host_quiet.py` refused (loadavg 3.27 over a 2.00
ceiling, `release_gate` live for c13-land-first). Every second it reports is void and must not be
quoted. Counters and digests do not move with the AICLK or with host load, which is why this run is
admissible at all.
