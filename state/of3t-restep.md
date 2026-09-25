# of3t-restep — the exactness-ON step's memory, phase by phase

**TASK TYPE:** VERIFY/BENCHMARK. Board pc card 0, p150a, custom 130-core firmware.
Commit `05a72f608` (merge of `origin/main` `c41bd779b`). Artifacts in `perf/of3t_restep/`.

STEP: OWED. No exactness-ON step time exists; both attempts died on host memory before the backward.

**There is still no full exactness-ON step time, and this pass did not produce one either.**
Two attempts, both stopped by host memory before the backward:

- 19:04 local, detached, crop 384 / cycles 4 / samples 4: **OOM-killed by the kernel** entering
  the backward, `anon-rss:19917952kB` (19.0 GiB) of a 30 GB box with no swap, pid 1795534.
  Forward, all four diffusion samples and all four loss roots completed.
  Evidence `perf/of3t_restep/out/run1_oom_evidence.txt`, log `run1_exact_on_killed.log`.
- 17:54–18:02Z, three profiled attempts: **self-killed** at a MemAvailable floor, in the taped
  trunk cycle, at 13.6 / 14.1 / 13.9 GiB. The guard is deliberate — see MEMORY GUARD below.

So the headline the brief and AMENDMENT 2 ask for is **owed, not delivered**. What this pass
delivers instead is the thing R216 said `of3t-restep` owes first, and it changes who the
headline's blocker is.

PARTITION: host MEMORY by phase, not seconds -- the six-part SECONDS split is owed, because no arm reached the backward.

`perf/of3t_restep/out/rss_384_exacton.summary.json`, from a 5 Hz RSS sampler tagged with the
phase of `fullstep.py` that is running. crop 384, cycles 4, samples 4, exactness ON.

| phase | dwell s | RSS in | RSS out | Δ GiB |
|---|---|---|---|---|
| imports | 5.8 | 0.014 | 0.431 | +0.43 |
| `get_device` | 0.0 | 0.459 | 0.459 | +0.00 |
| `capture_build_fold` (MSA resolve, featurizer) | 8.4 | 0.528 | 3.670 | **+3.14** |
| `capture_prep` (shipped fold to the sampler) | 8.9 | 3.670 | 4.605 | +0.94 |
| `AdamW.__init__` | 3.2 | 4.470 | 10.473 | **+6.00** |
| `trunk_forward` untaped, 3 recycles | 3.1 | 10.636 | 10.636 | +0.00 |
| `trunk_forward` taped, 1 cycle, exact | 27.2 | 11.125 | 13.948 | **+2.82**, breached |

**The largest single term in the step's host memory is the optimizer, and it is spent before
the first forward op runs.** `tt_bio/train/optim.py:164-165` keeps three host fp32 numpy copies
per weight — `master`, `exp_avg`, `exp_avg_sq` — over 381.3 M elements: 4.26 GiB of buffers,
6.00 GiB measured including the download's transient. This is by design and documented at
`optim.py:121` ("Masters and moments are fp32 numpy on host"). It is not a bug and not the
exactness. It is 31.6 % of the 19.0 GiB the kernel recorded.

The untaped recycle prefix retains **nothing** (+0.00 GiB over three cycles), which is the
control that says the growth above it is not just "the trunk runs".

EXACTPRICE: in MEMORY, +2.85 GiB for one in-flight exact softmax against R216's computed 3.377 GiB. In SECONDS: owed.

Two independent readings, and they agree with R216's arithmetic:

- The taped cycle grew **+2.82 GiB over 27 s and 13 exact softmax calls**, sawtoothing.
- In the first profile, RSS jumped **11.256 → 14.106 GiB in under 2.4 s** with
  `sm_inflight = 1`: a **+2.85 GiB step for one in-flight exact softmax**. R216 computed
  3.377 GiB at full shape from `PEAKPROBE.json`'s 2.00x. Measured on device, same order,
  slightly under — consistent with not every call being at the full `[384,4,384,384]`.

**So R216's open question has an answer: the exact softmax's float64 temporary is NOT the whole
19.0 GiB.** It is one transient of ~2.85–3.38 GiB riding on a floor that is already 10.6 GiB
before the tape opens. `of3t-xsplit`'s chunking fix removes the transient, which is real and
worth having, but on these numbers it **does not on its own make the exactness-ON step fit in
30 GB**: the floor under it — 6.00 GiB optimizer + 4.08 GiB capture — is 10.6 GiB, and the
retained tape and the backward's own temporaries are still to be added on top.

The seconds half of EXACTPRICE is owed: no arm this pass reached the backward.

## MEMORY GUARD — why these runs stop themselves

pc has 30 GB and no swap. The kernel OOM-killer picks by score, not by cause: on 19:11:36 it
chose this row's process, but a sibling row's pytest suite at 0.5 GB RSS is an equally
plausible victim of the next one. So `rssprofile.py` samples system MemAvailable and hard-exits
itself under a floor, flushing the JSONL per sample so the profile survives the exit. Every
peak it reports is therefore a **lower bound**, and the artifact says so in its own
`peak_is_a_lower_bound` field.

The box was contended throughout: `host_quiet.py` read **RED, loadavg1 3.33 against a 2.00
ceiling**, and MemAvailable at the three starts was 16.77, 16.45 and 15.43 GiB against a step
that the kernel has already shown needs 19.0 GiB. **Memory is load-insensitive so the profile
stands; no second in any artifact from this pass is quotable as a timing.**

STAMP: env.commit in every artifact; baseline_expiry.py run against this row's own output and it reads LIVE.

Every artifact carries `env.commit`, `env.branch`, `config.cycles`, `config.crop`,
`config.diffusion_samples` and `config.exact_training`.
`baseline_expiry.py perf/of3t_restep/out/rss_384_exacton.summary.json` →
**`LIVE: 05a72f608 is an ancestor of HEAD and nothing since touched the measured paths.`**

## SCOPE of what was measured

`fullstep.py` at `--cycles 4 --samples 4` contains: the untaped 3-cycle recycle prefix, one
taped trunk cycle, the diffusion module on 4 noised structures inside the same tape, the
`af3_loss` heads on host, the backward over the whole tape, and the AdamW step. It **omits**
upstream's 48 diffusion samples and upstream's per-step U{1..4} recycle draw, which is pinned.
That is the scope of the memory profile above. `39.11 / 7.5 = 5.2x` does not appear in any
artifact from this row.

VERDICT: PARTIAL -- the memory decomposition lands and corrects R216; the step time is owed.

**PARTIAL — the memory decomposition lands, the step time is owed.**

1. **R216's framing needs correcting, and this is the finding to carry forward.** The exact
   softmax is one ~2.85 GiB transient, not the 19.0 GiB. The **optimizer's host fp32 master and
   moments are 6.00 GiB**, the largest single term, present before any forward op and
   unaffected by anything `of3t-xsplit` can do to the softmax.
2. **A full exactness-ON step does not fit on pc at crop 384 and will not fit after the softmax
   fix alone.** Getting one needs either a box with more host RAM, or the optimizer's masters
   and moments moved off host — which is a change to `tt_bio/train/optim.py` and is not this
   row's to make. Handing that to you rather than acting on it: **this row does not change an
   engine file, by brief.**
3. The exactness-OFF full step on main, and both `bwprof.py` arms, are **remaining work** for
   the next pass. They need a green `host_quiet`, which pc did not offer in this window.

## Remaining, for the next launch of this row

- `bwprof.py --arm base` and `--arm noexact`, interleaved, median ≥3, `host_quiet` green and
  recorded per repetition. Read `of3t-exactscope`'s `PRICE.json` first — it has already banked
  noexact 25.24 s / softmax 637.33 s / base forward 268.27 s at crop 384, 1 cycle, so the rung
  to take is the one it could not: base's backward.
- The exactness-OFF full `fullstep.py` step on main, median of 3 — it fits in memory, it only
  needs the box quiet.
