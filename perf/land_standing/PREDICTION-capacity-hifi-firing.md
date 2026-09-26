# PREDICTION, registered before the counters were read — 2026-09-26 09:1xZ

This file is committed **before** `perf/land_standing/out/capcount/hifi_counters.jsonl` exists, so
the ordering is checkable with `git log` rather than on my word. This row has twice misread a zero
counter as "the lever did not fire" when it was actually "the wrong process", so the prediction is
registered first and the reading is graded against it.

## The question

Last pass's gate table said, honestly, that whether `TT_BIO_TRIATT_FUSED_HIFI` served a single call
inside the `capacity` arm was **unknown, not zero** — no counter was captured. `capacity` is the
only one of the ten arms whose inputs are not the 117 aa target:

    9j4c_abag   protenix-v2    1095 tokens   50 samples
    9ivj        opendde-abag    891 tokens    8 samples

## The predicate, read from source

`_tri_att_sdpa_hifi` declines in exactly two ways before reaching the inner routine:

    if ops.taping():                                    -> STATS["taped"] += 1,     return None
    if min(q.shape[2], k.shape[2]) < _TRIATT_FUSED_HIFI_MIN_S:  -> STATS["too_short"], return None

with `_TRIATT_FUSED_HIFI_MIN_S = 128` (`tenstorrent.py:2470`), and `_tri_att_sdpa_hifi_inner`
repeats the same MIN_S guard.

## What I predict, stated before looking

1. **`taped` is 0.** `capacity` is inference, not a taped training step; `ops.taping()` is false.
2. **`too_short` is 0 for the triangle-attention calls at these lengths.** 891 and 1095 are both
   far above 128. (`too_short` may still be non-zero from *other*, shorter attention sites in the
   same fold — that would not contradict this.)
3. **`served` > 0.** The lever is entered and `_tri_att_fused_large_s` returns a pick for at least
   some calls, recorded in `TRIATT_FUSED_HIFI_PICKS` keyed by `(q_len, k_len)`.

**If 1-3 hold, the ten-arm gate stops being only "the lever breaks nothing" and becomes "the lever
was exercised at 891 and 1095 tokens on two models, and every arm stayed green."** That is a
materially stronger claim than the one in the table today, and it is the reason this run is worth
18 minutes of card 3.

## What would falsify it, and what I would then conclude

- **`served` 0 with `too_short` > 0 at these lengths** — the predicate reads a different axis than
  I think it does, and my reading of `q.shape[2]` as the token axis is wrong.
- **All counters 0 and `picks` empty** — the counters are in the wrong process, NOT proof the lever
  is idle. The hook dumps one line per pid; a single all-zero line from the parent while the fold
  ran in a spawned worker is exactly what this row has been caught by before. The tell is the line
  count in the jsonl: one line means the worker never dumped.
- **`taped` > 0** — some part of `capacity` runs under a tape, which would be news.

## This is NOT a perf measurement

No AICLK is sampled and no timing is claimed from this run. A firing question is load-insensitive,
so a co-tenant on another card cannot change the answer. Any wall-time difference against the
earlier `capacity` runs is uncontrolled and is not offered as evidence of anything.
