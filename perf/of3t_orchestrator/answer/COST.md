# What OpenFold3 training costs on Tenstorrent, and why

Owner: `of3t-orchestrator`. The campaign's answer document and its last EXIT item. Every
still-missing number is marked **[OWED]** rather than estimated, and each gap names what closes it.
Restructured at pass 471 — it had grown by appending across seven passes into three overlapping
sections on host memory and two on the same gap, which is the failure this campaign keeps paying
for in its own state docs.

## The short answer

**OpenFold3 trains correctly on Tenstorrent, and what it costs is dominated by the mechanism that
makes it correct — host float64 arithmetic, not device work.**

The gradients are upstream 0.4.3's gradients to within upstream's own bf16 error, on every section,
over the whole step; stamped and unchanged. The price is a host round trip per softmax and per
layer norm inside every training tape, and on the measurements held today that is roughly a **25x**
multiplier. **So the interesting number is not how fast the device is. It is how much of a training
step is not on the device at all.**

## Is it correct? The gate says yes, and it can fail

The gradient gate was written *before* the levers it grades and came back **green** on the tree
carrying the landed engine change: **16 of 16 cases** at `rel_l2 <= 1.0e-02` and `cos >= 0.9999`,
against a float64 reference itself validated by central finite differences to **3.02e-10 –
3.63e-10**, with the bf16 quantisation floor measured at **2.76e-03** against the **2.8e-3**
predicted from the mantissa before any run.

All three controls fired: the float32 arm collapses the error (so the formula is imprecise rather
than wrong), LoRA's frozen base receives no gradient, and the deliberately broken arm was **refused
by measuring** rather than by raising before it measured. That last property is why this is
evidence rather than decoration, and it is the one such harnesses usually lack.

**That gate covers the code shipping today, exactly.** It ran on `f0e8f3b71`, and the five
gradient-bearing files are **byte-identical by blob hash** between that tree and current main.

**One caveat, stated because it is the kind that gets lost.** A separate and stronger claim — the
whole-model **per-parameter** score, 4,152 tensors with 21 of 21 sections inside a 3x bar — was
stamped on an **earlier** tree, and **1,032 insertions / 297 deletions** in the gradient path sit
between it and main. A per-op gate and a per-parameter model score are different instruments, so
the byte-identity above does not extend to it. Re-scoring on current main is owed.

## What it costs, with the scope on every line

**Nothing below is a chip-to-chip claim unless it says so**, and a ratio that beats the ~8.5x
compute / ~11x bandwidth silicon floor is a scope mismatch every time.

| quantity | value | scope, and where it comes from |
|---|---|---|
| exact softmax **alone**, whole step | **637.33 s**, 25.25x the noexact step | crop 384, 1 trunk cycle, taped, pc card 0 p150a, AICLK 1350 DURING, quiet box (`of3t-exactscope`) |
| noexact step, quiet | **25.24 s** | same arm, same box, sole tenant |
| noexact step, loaded | **39.11 s** | same arm at loadavg 4.58 — a **1.55x** contention penalty bounding every contended figure on that box |
| `exact_training`'s share of the backward | **95.2 %** | trunk-cycle A/B (`of3t-bwattrib`) |
| backward, both exact ops | **570.196 s of 705.78 s (80.79 %)** | verb self-time, the one additive axis (`of3t-xcost`) |
| of which genuine host float64 arithmetic | **94.07 s** | measured on the production route |
| of which host tilize/untilize + buffer copy | **82.56 s** | host-only, no DMA in the number |
| **real transfer, at the achievable roof** | **165.99 s of a 168.51 s residual** | ATTRIBUTED (`of3t-xsplit`, GO): only **2.54 s** is queue drain — `from_torch`'s is **zero by measurement**. The transfer half runs at the board's own achievable host-DMA rate measured in the same process, **1.265 GB/s live against 1.187 GB/s isolated**, so it is a **roof, not a defect**. **Not recoverable by removing overhead** |
| forward half, exactness ON | **375.908 s** against 25.563 s → **14.705x** | a BOUND from two durable clock stamps, not a timed run; matched one-taped-cycle axis, crop 384 (`of3t-restep`). Supersedes a ~299 s / ~54x derivation that row withdrew as unreproducible |
| full step, exactness ON | **[OWED]** | `of3t-stepqb2`. The campaign's one missing number |
| full step, exactness OFF, current tree | **39.886 s** steady, 77.532 s cold | `of3t-stepqb2`, n=3, spread 0.865 s, AICLK 1350 MHz DURING. **Disagrees with the carried 466.702 s by 11.7x**, which staleness does not explain — both are no-exactness arms. Scope / cold-vs-steady / a real gain are all open; the row is bounding it |
| GPU gap | **58-67x** step-to-step against an H200 at matched crop 384 — **DO NOT RE-PUBLISH UNTIL RE-DERIVED** | Two separate problems, and the second is newer and larger. **(a)** it prices the pre-exactness tree, a configuration whose gradients do not clear the bar and which we do not ship. **(b)** its numerator is **466.702 s**, off `451ed56f4` (*"a steady training step is 507.02 s"*), and the current tree's no-exactness step is **39.886 s** — so the numerator is a figure nothing on today's tree reproduces, and the error runs in the direction that flatters us. The accuracy caveat this row used to carry described (a) only and so read as if the seconds were sound |

**And when the shipped number arrives it will not be a chip-to-chip comparison.** The 58-67x above
is a device-to-device ratio for a configuration **we do not ship** — the pre-exactness tree, whose
gradients do not clear the bar. The configuration we *do* ship puts ~80 % of its backward in host
float64, so its ratio against a GPU measures **our host CPU and PCIe link against NVIDIA's device
implementation**, not one accelerator against another. Both numbers are legitimate; they answer
different questions, and neither should be quoted as the other. The axis belongs in the sentence.

**No clean second exists yet, and that is not a quibble.** Not one exactness-ON run has produced a
DURING-sampled AICLK — all four died before `fullstep.py` writes `env.aiclk_during`, and
`host_quiet` was RED at loadavg1 3.33 on the one that got furthest. So every figure above from this
campaign's own OF3T arms is a bound, an A/B between arms of identical scope, or a named quiet run.
The standing rule here is that a number without a DURING clock is not a measurement. Host memory is
load-insensitive, so the decomposition below is unaffected — that split is stated rather than
glossed.

**A narrower transfer is not available either.** Halving the bytes would have been bit-exact only
if the score block's fp32 words carried ≤ 8 explicit mantissa bits. Censused on the real tensor at
real sites during a real backward — 324 crossings, **73,383,542,784 elements**, three sites, every
block index, no synthetic data — **65.69 % of words carry a set bit below the bf16 boundary**, the
mean is **8.47** set mantissa bits, and **20.76 % of the mass sits on bit 0 alone**. So a bf16
crossing would be a rounding, and this campaign does not trade fidelity for bytes.

## Why it costs that

Tenstorrent's fp32 is a few mantissa bits short of IEEE fp32. Nothing on-device reaches real fp32:
the device softmax reads **2.029e-02** against float64, `precise_config()` **1.646e-03**,
`_accurate_softmax` **5.156e-04**, where a true fp32 softmax agrees with float64 to ~**1e-7**.
Upstream trains fp32 on GPU where fp32 means IEEE fp32, so a host round trip is not overshooting
them — it is the only way to reach what they already do. A silicon ceiling, not a configuration
choice, and it is why no on-device softmax configuration ever cleared the accuracy bar.

## Why it cannot simply be turned off

Against the pre-registered bar, lower is better, 1.0 is the bar:

- **both exact (shipped): 0.98226x** — the only configuration that clears it
- softmax only: **1.30379x**, and it *raises* the squared gradient error against float64 to
  **1.1760x** of shipping nothing, because part of the apparent gain is matching upstream's
  rounding rather than approaching the truth
- layer norm only: **1.28386x**
- neither: **1.4512x**

The fidelity is fixed; only its price is variable. Both scopes are load-bearing.

## What has been recovered

- `of3t-zerosfill`: **1.0518x**, on main. A write-only fill running 424x off its roof — and its
  briefed share of 35.1 % was an async-dispatch artifact worth **1.04 s**.
- `of3t-tapedfwd`: **1.00508x**. The eleven fused forward kernels decline under taping by design.
- `of3t-xsplit`: **1.048x** — **32.55 s off the backward, 4.61 %**, bit-identical. Moving the
  TILE/ROW_MAJOR conversion from host to device. **Not a bandwidth saving**: both routes move the
  same 0.906 GB over PCIe, and the crossing stays host-DMA-bound at 1.19-2.25 GB/s either way. It
  relocates the conversion from a host untilize at **12.8 GB/s** to a device `to_layout` at
  **202.7 GB/s** — the same bytes on a machine 15.8x faster for them, because the card's DRAM roof
  sits far above the PCIe roof that binds the crossing.
- **Refuted, and worth as much**: the fused-backward-kernel programme the sprint was premised on.
  J1 NO-GO at a 9.40 s ceiling against a 46.4 s brief, J2 NO-GO on its free route, J3 dead at
  `sdpa_taped_calls = 0`, J4 small. The job list had been ranked by tape-node **count**, and count
  does not track seconds — LayerNorm backward is 52.4 % of nodes and 2.6 % of backward seconds.

## What it needs to run

**Host memory, measured.** A 5 Hz RSS sampler tagged with the running phase, crop 384, exactness ON
(`perf/of3t_restep/out/rss_384_exacton.summary.json`):

| phase | ΔGiB |
|---|---|
| imports | +0.43 |
| capture: MSA resolve + featurizer | **+3.14** |
| capture: fold to the sampler | +0.94 |
| **`AdamW.__init__`** | **+6.00** |
| untaped trunk, 3 recycles | **+0.00** |
| taped trunk, 1 cycle, exact | **+2.82** |

**The largest single term is the optimizer, spent before the first forward op runs — and a direct
census puts it higher than the sampler did.** `tt_bio/train/optim.py` builds **five** host fp32
dicts, not three; the three resident before any compute are `master`, `init_master` and
`init_device`, and at crop 384 they total **7.105 GiB** (3,152 parameters, 381.3 M elements,
4.000-4.003 bytes/element/dict). **The 6.00 GiB in the phase table above is a 5 Hz sampler's phase
boundary and is 1.10 GiB low** — a sampled trajectory bounds a phase, it does not census one.

**4.263 GiB of it goes, with the gradients bit-identical.** `init_master` and `init_device` were
built from the same unwritten value three lines apart and held **identical bytes**; and
`np.zeros_like` is `empty_like` plus `copyto(0)`, so it **writes every page** — the moments were
fully resident from construction rather than a cheap untouched mapping. Materialising them at first
use, inside `step()`, gives **construct 7.105 → 2.842 GiB (−60.0 %)** and **after one step
7.105 → 5.684 GiB**. Proven bit-identical across six arms and six steps by raw-byte comparison of
masters, both moments, device weights and every report — 0 differences.

**Which of those numbers is on main today: the 7.105 GiB one.** The fix is release-gated on
`wk/of3t-optorder` and unmerged, so a user running main gets the full figure. The 2.842 / 5.684
pair describes the configuration that would ship if it lands, and `docs/training.md` carries
those two numbers on the same branch, so the doc and the code move together. The untaped recycle prefix retains
**nothing**, which is the control that makes the rest of the table readable. One in-flight exact
softmax costs **+2.85 GiB** measured against 3.377 GiB predicted — real, understood, and a
transient riding on a **10.6 GiB floor** rather than the floor itself.

**So the step wants more than 19.0 GiB of host RAM**: ~6 GiB of optimizer state before any compute,
~4 GiB of capture, then the tape. It was OOM-killed on a 30 GB box holding **81.4 %** of all
resident memory there, and fits comfortably on a 249 GB host. **Training OpenFold3 here is
specified by host memory and host CPU as much as by the accelerator**, which follows directly from
the fidelity mechanism being host work.

**The PCIe link is well below capability, and the two hosts differ by 2x on it.** pc's Blackhole
negotiates **Gen4 x8** against a **Gen5 x16** capability (~15.8 GB/s per direction against ~63);
all four qb2 cards negotiate **Gen4 x4** against Gen4 x8 (~7.9 against 15.8). Since the exactness
is a host round trip per softmax and per layer norm, **its floor is that link** — so a host-bound
figure from one host is not comparable to one from the other, independently of board class, and the
two caveats run in opposite directions. It does **not** explain the crossing cost: those measure
**1.19-2.25 GB/s**, roughly 7x below even pc's degraded link, so raw bandwidth is not what binds
them.

**And a 30 GB box is not merely tight, it is not repeatable.** pc's available memory at run start
swings from **~16.5 to ~26 GiB** with the resident agent population — 4.33 GiB of co-tenants at the
OOM, ~14 GiB at both later attempts. A run that fits at one hour does not at another, independently
of the ceiling, and a timing taken there is conditioned on a number nobody controls.

**The capability limit is 512 tokens, it is measured, and it is undocumented.** Every figure here
is at **crop 384**. The largest crop that runs is **512**: `of3t-crop768` built the missing 544
fixture and ran the ladder — **544, 576, 640 and 768 all refuse**, so there is no rung between 512
and 640 that clears (measured 2026-09-21; nothing since has touched the device-side wall, and
`of3t-cropwall`'s DRAM-padding fix that might raise it is release-gated and unmerged).

**`git grep` over `origin/main -- README.md docs/` finds no crop or token limit stated anywhere.**
Upstream ships four stage configs and this campaign has only ever run the smallest end to end. That
is a capability a user meets and it belongs in the docs.

## What is missing, why, and the one decision with a number on it

The full exactness-ON step time is **[OWED]**, and the reason is fleet capacity rather than
engineering:

- **pc (30.5 GB, 1 card)** cannot reliably *dispatch* a card row — the scheduler reserves half the
  box, **15,617 MB**, against ~15,700 MB available, a **93 MB** margin — and cannot *hold* the step.
- **qb2 (249 GB, 4 cards)** has one free card, half a p300c board pair whose sibling carries another
  campaign's live arm. While that sibling is in use a pair reset is unavailable, so a failed device
  open has **no recovery path** — on a box that hard-hung on exactly that failure the same day.
  Taking it risks three other campaigns' running work, not just this measurement.
- **qb1** is unreachable.

**So the missing number is one card-hour away, not one fix away.** `of3t-stepqb2` is written and
gated on qb2's card pair becoming recoverable.

**The one decision, with its number:** the 6.00 GiB of AdamW host state could move to device.
**Decided against** (`state/ask-of3t-optmem-decision.md`) — device DRAM is where this fleet already
runs out, and trading a well-understood host ceiling for that failure mode is the wrong direction.
The better and unasked question is **ordering**: `AdamW.__init__` runs *before* the trunk forward
and its state is only touched in the optimizer step, so constructing it after the capture may be
worth 6.00 GiB at the peak with no redesign.

## What this does NOT say

It does not say what a full exactness-ON step costs — **[OWED]**, and it is the one number between
this campaign and a defensible headline. It does not say the 6-7x software target is in reach; the
job list as it stands refutes that. And no speed number here may be quoted without the exactness
state, the scope and the board class beside it.
