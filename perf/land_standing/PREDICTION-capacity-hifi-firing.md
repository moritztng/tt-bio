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

---

# RESULT — the prediction above is FALSIFIED. Counted zero, and structurally so. 2026-09-26 09:34Z

`capacity` re-run with counters, rc=0, **GATE PASS**, both legs green again:

    capacity:9j4c_abag   7.23 GiB   50 CIFs  1 PAE  <=10.5 GiB   787s  PASS
    capacity:9ivj        7.00 GiB    8 CIFs  1 PAE  <=12.0 GiB   343s  PASS

Four counter lines, one per process:

    pid 2059377   hifi {served 0, declined 0, too_short 0, taped 0}   route {fused 1600, stock 0}   1 shape
    pid 2081499   hifi {served 0, declined 0, too_short 0, taped 0}   route {fused 1304, stock 0}   2 shapes
    pid 2059274   hifi all zero, route all zero   "no triangle-attention pick recorded in this process"
    pid 2081214   hifi all zero, route all zero   "no triangle-attention pick recorded in this process"

## This is a real zero, not the wrong-process artifact

That distinction is the whole reason the prediction was registered first. **Two of the four
processes recorded substantial real work** — 1600 and 1304 SDPA calls, all on the fused route, with
recorded chunk picks. The two that recorded nothing say so explicitly in their own note. So the
fold was instrumented, the counters were live in the processes that did the folding, and
`_tri_att_sdpa_hifi` was still **never entered**: `declined` and `too_short` are zero too, not just
`served`.

## Why — and the specific error in my prediction

The hifi arm is reached at `tenstorrent.py:9042`, inside this branch:

    def _attend_heads(q, k, v, bias, keep_heads=False, gate=None):
        if _FP32_SOFTMAX or self.fp32_softmax:          # <- tenstorrent.py:9032
            ...
            if _fused_hifi_on(self.fused_hifi):
                o = _tri_att_sdpa_hifi(...)

**The gate is `_FP32_SOFTMAX or self.fp32_softmax`.** Neither `protenix-v2` nor `opendde-abag`
sets `fp32_softmax` at its triangle-attention sites, so `_attend_heads` takes the other branch —
which goes to the fused SDPA directly and reads `sdpa_hifi`, never `fused_hifi`. That is exactly
what the counters show: 1600 and 1304 calls on the **fused** route, zero on stock, zero hifi.

I predicted `served > 0` and was wrong. The error is precise and worth naming: I took
`if att.biased or _FP32_SOFTMAX or att.fp32_softmax:` at `tenstorrent.py:2376` to be this branch's
condition, and reasoned that triangle attention is always biased so the branch is always taken.
**That line is not this branch.** It belongs to the gate-epilogue helper and its effect there is
the opposite — `att.biased` sends it to `_gate_reject("site")`. Two predicates mentioning
`fp32_softmax` a few thousand lines apart, and I matched the wrong one.

This was the fourth falsifier listed above ("the models may not enter the `fp32_softmax` branch at
all"), so the prediction failed in a way it had already written down rather than in a surprising
one.

## What this settles

**"Unknown, not zero" is now COUNTED ZERO, for a structural reason, at 891 and 1095 tokens.**
`TT_BIO_TRIATT_FUSED_HIFI` cannot reach `protenix-v2` or `opendde-abag` at all, at any length,
because their sites do not take the `fp32_softmax` route.

So the ten-arm gate says exactly what the conservative framing already said, and no more:
**the lever breaks nothing, on every arm including the two long ones — and not one arm validates
it.** The hoped-for upgrade to "exercised at real length" does not happen. The validation remains
the PepN 3B34 evidence on OpenFold3, which passes `fp32_softmax=True` at all four sites and
therefore does take this branch.

One consequence worth carrying forward: **the lever's blast radius is narrower than the model list
suggests.** It can only ever move a site that sets `fp32_softmax`. Of the sites enumerated earlier,
that is OpenFold3's four — and of those, `openfold3.trunk` is already pinned on in shipped main, so
the reachable surface is `openfold3.msa`, `openfold3.template` and `openfold3.confidence`.

Not a perf measurement: no AICLK sampled, no timing claimed. Peak DRAM was identical to the byte
against both earlier `capacity` runs (7.23 and 7.00 GiB), which is the controlled part; the wall
times (787s / 343s here) are uncontrolled and are not offered as evidence.
