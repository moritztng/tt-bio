# What OpenFold3 training costs on Tenstorrent, and why

Owner: `of3t-orchestrator`. This is the campaign's answer document and the last EXIT item —
*"one honest paragraph saying what OF3T training costs and why."* Drafted at pass 464 against what
is established, with every still-missing number marked **[OWED]** rather than estimated. Nothing
here may be filled in from a proxy; each gap names the row that closes it.

## The short answer

**OpenFold3 trains correctly on Tenstorrent, and what it costs is dominated by the mechanism that
makes it correct — which is host float64 arithmetic, not device work.**

The gradients are upstream 0.4.3's gradients to within upstream's own bf16 error, on every section,
over the whole step; that is stamped, evidence-backed, and unchanged. The price is a host round
trip per softmax and per layer norm inside every training tape, and on the measurements this
campaign holds it is roughly a **25x** multiplier on the step. So the interesting number is not how
fast the device is. It is how much of a training step is not on the device at all.

## The arithmetic, with its scope on every line

**Nothing below is a chip-to-chip claim unless it says so**, and a ratio that beats the ~8.5x
compute / ~11x bandwidth silicon floor is a scope mismatch every time (K32).

| quantity | value | scope, and where it comes from |
|---|---|---|
| exact softmax **alone**, whole step | **612.09 s**, 25.25x the noexact step | crop 384, 1 trunk cycle, taped, pc card 0 p150a, AICLK 1350 DURING, quiet box (`of3t-exactscope`) |
| noexact step, quiet | **25.24 s** | same arm, same box, sole tenant |
| noexact step, loaded | **39.11 s** | same arm at loadavg 4.58 — a **1.55x** contention penalty that bounds every contended figure on that box |
| `exact_training`'s share of the backward | **95.2 %** | trunk-cycle A/B (`of3t-bwattrib`, R212) |
| backward, both exact ops | **570.196 s of 705.78 s (80.79 %)** | verb self-time, the one additive axis (`of3t-xcost`) |
| of which genuine host float64 arithmetic | **94.07 s** | measured on the production route |
| of which host tilize/untilize + buffer copy | **82.56 s** | host-only, no DMA in the number |
| **unattributed: DMA, blocking sync, or queue drain** | **168.51 s (23.87 %)** | by subtraction; three different defects — `of3t-xsplit` owns splitting it |
| forward half, exactness ON | **~299 s** against 5.5 s → **~54x** | coarse, from file write times, labelled coarse (`of3t-restep`) |
| full step, exactness ON | **[OWED]** | `of3t-restep`. The campaign's one missing number |
| full step, exactness OFF, current tree | **[OWED]** | `of3t-restep`; the stale 466.702 s is from `451ed56f4`, a tree where the feature did not exist |
| GPU gap | **58-67x** step-to-step against an H200 at matched crop 384 | **on the pre-exactness tree**, so it prices a configuration whose gradients do not clear the bar |

## Why it costs that, in one paragraph

Tenstorrent's fp32 is a few mantissa bits short of IEEE fp32. Nothing on-device reaches real fp32:
the device softmax reads 2.029e-02 against float64, `precise_config()` 1.646e-03, `_accurate_softmax`
5.156e-04, where a true fp32 softmax agrees with float64 to ~1e-7. Upstream trains fp32 on GPU where
fp32 means IEEE fp32, so a host round trip is not overshooting them — it is the only way to reach
what they already do. That is a silicon ceiling, not a configuration choice, and it is why no
on-device softmax configuration ever reached the accuracy bar.

## Why it cannot simply be turned off

Measured against the pre-registered bar, lower is better, 1.0 is the bar:

- both exact (shipped): **0.98226x** — the only configuration that clears it
- softmax only: **1.30379x**, and it *raises* the model's squared gradient error against float64 to
  **1.1760x** of shipping nothing, because part of the apparent gain is matching upstream's rounding
  rather than approaching the truth
- layer norm only: **1.28386x**
- neither: **1.4512x**

So the fidelity is fixed and only its price is variable. Both scopes are load-bearing.

## What has actually been recovered

- `of3t-zerosfill`: **1.0518x**, on main. A write-only fill running 424x off its roof — and its
  briefed share of 35.1 % was an async-dispatch artifact worth 1.04 s (K30).
- `of3t-tapedfwd`: **1.00508x**. The eleven fused forward kernels decline under taping by design.
- Refuted, and worth as much: the fused-backward-kernel programme the sprint was premised on.
  J1 NO-GO at a 9.40 s ceiling against a 46.4 s brief, J2 NO-GO on its free route, J3 dead at
  `sdpa_taped_calls = 0`, J4 small. The job list had been ranked by tape-node **count**, and count
  does not track seconds (R211: LayerNorm backward is 52.4 % of nodes and 2.6 % of seconds).

## The host requirement, which is a product fact and not a footnote

The shipped crop-384 step with the exactness on wants **more than 19.0 GiB of host RAM** and does
not fit on a 30 GB box: it was OOM-killed holding **81.4 %** of all resident memory there. It fits
comfortably on a 249 GB host. **Training OpenFold3 here is specified by host memory and host CPU as
much as by the accelerator**, which follows directly from the fidelity mechanism being host work.

## What this does NOT say

It does not say what a full exactness-ON step costs — that is **[OWED]** and is the one number
between this campaign and a defensible headline. It does not say the 6-7x software target is in
reach; R212 refutes that for the job list as it stands. And no speed number here may be quoted
without the exactness state, the scope and the board class beside it.
