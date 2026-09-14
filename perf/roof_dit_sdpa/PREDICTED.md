# PREDICTED — the token DiT's 80.53 GB/fold, before the device is opened

Written before any device run on this branch. Sources read first, in this order: the eligibility
expression at `tt_bio/tenstorrent.py:7614`, the flag default at `:1336`, the atom-level attention at
`:7386`, and the sibling capture `wk/roof-diffusion-tagged-capture` (6cec86aa3).

## What the source already says

`_B2_TOKEN_DIT_SDPA = env_flag("BOLTZ2_TOKEN_DIT_SDPA", True)`. The flag defaults **on** at the tip.
`fc7fed56f` flipped it on 2026-09-11 19:43 UTC, four and a half hours after `graph_512_all.json.gz`
(the capture the 80.53 GB/fold row was ranked from) was taken at 15:04 UTC.

So the premise of this row — "the triangle attention has a fused SDPA and the token DiT does not" —
is false at the tip. Both of the brief's two candidate shapes already run a fused SDPA:

* token DiT `1x512x768`: `tenstorrent.py:7640`, gated on `token_dit_sdpa`.
* atom transformer `1x140x32x128`: `self._attention()` at `:7386`, which for any non-fp32 dtype calls
  `ttnn.transformer.scaled_dot_product_attention` **unconditionally** — no flag, no eligibility
  condition, it has never had one.

`sdpa_generic.plan_for_shape` does not refuse the token DiT's shape either. The sibling measured it
accepting: 19 of 20 chunk pairs, no padded mask, 128 of 130 cores, 114 KB of CB. It is a
transcription of the same wheel kernel, so routing through it deletes no byte.

## (a) How much of the 80.53 GB/fold is still on the table at the tip

**0.00 GB/fold.** The 8.39 MB probability matrix is never allocated: SDPA keeps it in a circular
buffer between the QK and PV matmuls, which is the whole point of the kernel. The eligibility
expression can only decline on `z is None`, `seq_mask is not None`, or the fp32 raw-matmul path, and
Boltz-2's token DiT hits none of those.

Prediction to be checked on the device, via `scripts/lever_census.py` on a real CLI fold (the
counter must be summed across the worker spawn, not read in the launcher):

    B2_TOKEN_DIT_SDPA   resolved True   served 4800   declined 0     512 aa, 200 sampling steps
                                        served 4800   declined 0     298 aa, 200 sampling steps

4800 = 24 token-DiT layers x 200 steps. The declined count is the number that matters: any non-zero
value is a size-conditioned eligibility hole and turns this row back into a live lever.

## (b) Op ratio on `DiffusionTransformerLayer|1x512x768`

**1.000x**, because there is no arm to turn on. The lever this row was written to build is already
the default. The value it delivered when it was flipped is on the record and is not re-claimed here:
1.0522x at the fold, 22.195 -> 21.095 s, `perf/b2x-integrate/README.md`.

## (c) Fold ratio

**1.000x.**

## The conversion, stated because the brief asks for it

The prize was never worth its byte count. 80.53 GB/fold against the measured 424.7 GB/s stream roof
is **0.190 s**, on a 17.340 s cell — 1.1 %. The budget table's deficit at that row is 1.448 s, so
even a perfect collection of every byte would have closed 13 % of it. That the shipped flag measured
1.100 s at the fold is therefore mostly *not* bytes: it is 4800 fewer program pairs and a softmax
that never leaves L1. Byte rankings rank; they do not price.

The remaining DRAM traffic at this exact site is the 8.389 MB rollout-invariant pair bias, re-read as
the SDPA mask on all 4800 calls: 40.3 GB/fold, 0.095 s at the stream roof. Halving it in bfp8 was
already measured and refused on accuracy (0.949x AND 1.496 A). Nothing else at the site is a round
trip; the mask is a genuine operand.

## Kill criteria, and which one this fires

Criterion 1: "Kill if the round trip is already collected at the tip: report the byte accounting and
STOP, which is a pass." That is the prediction. The device run exists to falsify it, by finding a
non-zero `declined` count at either size.
