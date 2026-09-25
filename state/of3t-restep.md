# of3t-restep — the exactness-ON step: its memory, its forward-half bound, and where it can run

**TASK TYPE:** VERIFY/BENCHMARK | **PLAYBOOKS loaded:** VERIFY/BENCHMARK | **memories read:**
`a-firing-question-is-load-insensitive-so-a-loud-box-does-not-block-it`,
`read source to choose what to MEASURE` (the 30 GB vs 249 GB OOM),
`a-seam-bracketed-dram-read-understates-the-in-step-peak`,
`a-measurement-precondition-the-arm-itself-cannot-satisfy`,
`a-tape-node-count-is-not-a-cost-proxy`, `heading-range-splice`.

Board for every number below: **pc card 0, p150a on custom 130-core firmware**, commit
`05a72f608` (merge of `origin/main` `c41bd779b`). Artifacts under `perf/of3t_restep/`.

STEP: the exactness-ON forward half at crop 384 is **<= 386 s**, and the backward never ran.
Run 1 (`out/step_exact_on_384.json`, `out/run1_exact_on_killed.log`) started 17:04:35Z and was
OOM-killed by the kernel at 17:11:36Z: **421 s wall**. Subtracting the artifact's own setup —
imports 5.8 s, `build_fold` 8.94 s, `prep_to_sampler` 14.17 s, `AdamW.__init__` 3.22 s, untaped
3-cycle prefix 3.11 s = 35.2 s — leaves **385.8 s** covering one taped trunk cycle, 4 diffusion
samples and 4 loss roots, all of which the log shows completing. The diffusion half of that is
**13.140 s** measured in-log (11.579 + 0.495 + 0.570 + 0.496), so taped trunk + losses is
**<= 372.7 s**. Against the banked forward half at `451ed56f4` — trunk 3.491 + diffusion 0.605 +
losses 1.823 = **5.919 s** — the exactness-ON forward half is **<= 65.2x**. The axis holds:
both are **one taped trunk cycle** (`step_rekey_b_384.json` records `cycles_pinned: 1` with
`trunk_nograd_prefix_s = 0.0`, and this run's 3-cycle untaped prefix is subtracted above), 4
diffusion samples, 4 loss roots. `of3t-exactscope`'s independently banked base forward of
**268.27 s** sits inside this bound, which is the corroboration that makes it worth quoting.
This is an **upper bound from two clock stamps, not a timed run** — see CLOCK. The full step is
bounded above by nothing, because the backward was killed; getting it needs a host this step
fits on, and pc is not one.

PARTITION: host **memory** by phase — a 5 Hz RSS sampler tagged with `fullstep.py`'s running
phase, crop 384, cycles 4, samples 4, exactness ON
(`out/rss_384_exacton.summary.json`). The six-part **seconds** split is the part this pass could
not reach, because no arm survived into the backward.

| phase | dwell s | RSS in GiB | RSS out GiB | delta GiB |
|---|---|---|---|---|
| imports | 5.8 | 0.014 | 0.431 | +0.43 |
| `get_device` | 0.0 | 0.459 | 0.459 | +0.00 |
| `capture_build_fold` (MSA resolve, featurizer) | 8.4 | 0.528 | 3.670 | **+3.14** |
| `capture_prep` (shipped fold to the sampler) | 8.9 | 3.670 | 4.605 | +0.94 |
| `AdamW.__init__` | 3.2 | 4.470 | 10.473 | **+6.00** |
| trunk untaped, 3 recycles | 3.1 | 10.636 | 10.636 | **+0.00** |
| trunk taped, 1 cycle, exact | 27.2 | 11.125 | 13.948 | **+2.82**, then breach |

**The largest single term in the step's host memory is the optimizer, and it is spent before the
first forward op runs.** `tt_bio/train/optim.py:164-165` keeps three host fp32 numpy copies per
weight — `master`, `exp_avg`, `exp_avg_sq` — over 381.3 M elements: 4.26 GiB of buffers, 6.00 GiB
measured including the download's transient. Documented at `optim.py:121` ("Masters and moments
are fp32 numpy on host"), so this is by design, not a defect. It is **31.6 %** of the 19.0 GiB
the kernel recorded. The untaped 3-cycle prefix retains **+0.00 GiB**, which is the control that
says the growth above it is the tape and the exactness rather than "the trunk ran".

EXACTPRICE: in **memory**, one in-flight exact softmax is **+2.85 GiB**, measured — RSS jumped
11.256 -> 14.106 GiB in under 2.4 s with `sm_inflight = 1`. R216 computed 3.377 GiB at full
`[384,4,384,384]` shape from `PEAKPROBE.json`'s 2.00x; the measured value sits just under, which
is what you expect when not every call is at full shape. **So R216's open question has an answer:
the exact softmax's float64 temporary is NOT the whole 19.0 GiB.** It is one ~2.85-3.38 GiB
transient riding on a **10.6 GiB floor that exists before the tape opens** (6.00 optimizer + 4.08
capture + 0.43 imports). `of3t-xsplit`'s chunking removes the transient and is worth doing, but
on these numbers it does not by itself make the exactness-ON step fit in 30 GB. In **seconds**,
the forward is bounded at <= 386 s (see STEP) and the backward's own price is the one number this
row still owes, at 0 arms surviving into it.

STAMP: every artifact carries `env.commit`, `env.branch`, `config.crop`, `config.cycles`,
`config.diffusion_samples` and `config.exact_training`, plus `peak_is_a_lower_bound` so a
self-killed profile cannot be read as a completed one.
`baseline_expiry.py perf/of3t_restep/out/rss_384_exacton.summary.json` →
**`LIVE: 05a72f608 is an ancestor of HEAD and nothing since touched the measured paths.`**

CLOCK: **no DURING-sampled AICLK exists for any of this row's own runs**, and I will not borrow
one. `fullstep.py` writes `env.aiclk_during` from its `during()` sampler at the end of `main()`;
all four attempts died before that line, so the field is absent from every artifact here. That is
exactly why STEP is stated as a bound from two wall-clock stamps and not as a step time. For the
board itself, `of3t-exactscope` banked **AICLK 1350 MHz DURING (median, n=195)** on this same pc
card 0 p150a for its exactness arms, so the board boosts under this workload and an 800 MHz
reading here would be an idle sample — but that is exactscope's measurement, not this row's, and
no number above is quoted as clean because of it. `host_quiet.py` also read **RED at loadavg1
3.33 against the 2.00 ceiling** for the profiled runs. Host memory is load-insensitive, so
PARTITION and EXACTPRICE's memory figures stand unaffected; no second in this document is a clean
headline.

MEMORY GUARD: pc has 30 GB and no swap, and the kernel OOM-killer picks by score, not by cause.
On 19:11:36 local it chose this row's process; a sibling row's pytest suite at 0.5 GB was an
equally plausible victim of the next one. So `rssprofile.py` watches system MemAvailable and
hard-exits itself under a floor, flushing the JSONL per sample so the profile outlives the exit.
Every peak it reports is a **lower bound** and the artifact says so in its own field. Three
profiled attempts self-killed at 13.6 / 14.1 / 13.9 GiB with MemAvailable at 16.77 / 16.45 /
15.43 GiB at their starts, against a step the kernel has already shown needs 19.0 GiB.

CEILING: **this row is `card=cpu` and its headline has nowhere to run today.** pc at 30 GB cannot
hold the step. qb2 has 249 GB (186 GB free) and card 0 is unheld, but **qb2 card 1 is still held
— pid 542435, checked on qb2 this pass** — so `tt-smi -r` on the 0/1 p300c board pair stays
unavailable and a failed open on card 0 has no recovery path, on a box that hard-hung at
09:03:31Z today on exactly that error. Amendment 11's condition is not met and I am not taking
that risk on three other campaigns' live work.

**One thing to flag rather than act on: the pid on qb2 card 1 has changed.** Amendment 11 records
`land-standing`'s arm as pid 466323; it is now **542435**. The card has stayed busy across an arm
rotation, which means "card 1 free" is likely to be a short window between arms rather than a
state that persists — whoever catches it should expect to move quickly, and per the brief I am
not clearing that cardblock myself.

LEASE: **released this pass.** `state/leases/pc-card0.json` still named `worker:of3t-restep` with
`released: null` and pid 3013249 from the 18:01Z profile run. Both conditions checked before
acting — pid dead, `fuser /dev/tenstorrent/0` empty — then renamed to
`pc-card0.json.released-2026-09-25-of3t-restep-self-release-pid-3013249-dead-node-free`. pc card 0
is free for `of3t-xsplit`.

DEFECT I CAUSED, reported rather than quietly repaired: **I overwrote this state doc without
reading it first.** `state/` is gitignored in `~/.coworker`, so the prior pass's version — the one
amendment 10 quotes at "~299 s forward half ... coarse from file write times" — is gone, and the
derivation with it. Copying the run log to `run1_exact_on_killed.log` also reset the mtime that
derivation was built on, so ~299 s is no longer reproducible from this row's surviving artifacts.
STEP above is a fresh bound computed from the artifact's own `started_utc` and the kernel's OOM
timestamp, which are both durable; it brackets the lost figure rather than restating it. The
general lesson is the one already in the fleet's memory as `heading-range-splice`: read a
gitignored deliverable before writing over it, because there is no restore.

VERDICT: PARTIAL — the memory decomposition lands and corrects R216, the forward half is bounded
at <= 65.2x, and the full exactness-ON step needs a host it has not been given.

1. **Carry forward:** the exact softmax is one ~2.85 GiB transient, not the 19.0 GiB. The
   optimizer's host fp32 master and moments are **6.00 GiB**, the largest single term, present
   before any forward op and untouchable by `of3t-xsplit`'s softmax work.
2. **A full exactness-ON step does not fit on pc at crop 384 and will not fit after the softmax
   fix alone.** It needs either a large-memory host with a recoverable card, or the optimizer's
   masters and moments moved off host — a change to `tt_bio/train/optim.py`, which this row does
   not make, by brief. That is a decision for Moritz with a number attached, and the number is
   6.00 GiB.
3. The exactness-OFF full step and both `bwprof.py` arms stay open for whoever gets qb2 card 0
   once card 1 frees. `of3t-exactscope`'s `PRICE.json` already banks noexact 25.24 s, softmax
   637.33 s and base forward 268.27 s at crop 384 / 1 cycle, so the rung still open is the one it
   could not take either: base's backward.
