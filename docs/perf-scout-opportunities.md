# TT perf-acceleration scout — ranked opportunities (tt-bio + tt-atom)

Scout pass, 2026-07-12, qb1 physical card 2 (Blackhole P150). Proposals only — no
runtime code changed. Every expected-speedup figure is an **estimate** grounded in the
stated mechanism/precedent unless it cites a measured number; each proposal names the
bench that would confirm it. Numbers I measured this pass are marked *(measured)*.

## Context: what is already won or dead (do not re-propose)

Single-card **warm compute** on the tt-bio shared trunk is exhausted. Prior scouts +
memory measured and rejected: triangle-mul / transition / TriangleAttention /
OuterProductMean / PairWeightedAveraging dispatch-collapse, Protenix confidence,
BoltzGen diffusion trace, Protenix device-resident diffusion, atom-attention KV-pack,
trimul megakernel, largeseq single-card levers, orb `--fast`. The one shared-trunk win
(FC1+SwiGLU epilogue fusion, ~1.18x/1.14x at N=512/1024) is landed and build-gated.
Boltz-2 / Protenix-v2 / ESMFold2 all already run device-resident recycling trunks with
one-time weight caches; ESMC-6B multicard regression is fixed on main.

tt-atom warm compute is likewise mature: SO(2)-as-GEMM, Wigner-as-sparse-MAC, analytic
on-device reverse pass, fused kernels (`fused_rotate` 13.9x isolated, `fused_gate`,
`fused_ln_bw`), disjoint-union batching (~13x), trace for MD/relax (2.14x/step),
multicard 3.95x@4. Dead: bf8 **weights** (data-movement bound, not flop), lowering math
fidelity (HiFi4==LoFi), bf8 one-hot scatter (non-bit-exact, no speed).

The remaining ROI is **not** deeper single-card trunk kernels. It is: precision on the
one trunk that was left bf16, cold-start amortization, single-job multi-card sharding,
and flipping-on tt-atom wins that already exist but ship OFF.

---

## Ranked proposals

### 1. ESMFold2 `--fast` trunk in bf8 (bandwidth)  — TOP PICK
**Model:** ESMFold2. **Type:** compute/bandwidth. **Effort:** LOW–MEDIUM. **Gate:** accuracy.

ESMFold2's 48-block trunk matmul weights stay **bf16 even on `--fast`**
(`tt_bio/esmfold2.py:39` `_DTYPE = ttnn.bfloat16`, never reassigned; the fast branch at
`:80-89` only toggles bf16-vs-fp32 *accumulation*, not weight dtype). Boltz-2 and
Protenix-v2 already drop their trunk weights to **bf8** on `--fast` via the shared
`tenstorrent._dtype()` (`tenstorrent.py:69`, ~20 `dtype=_dtype()` sites), holding trunk
PCC ~0.99 (`protenix.py:834-843`). ESMFold2's own trunk never routes through that helper.

The trunk is the dominant warm stage — *(measured)* trunk 1.0 s of a 2.1 s L=76 fold
(48%); at N=512/1024 the trunk is 5–20 s dominated by trimul + transition, and the prior
scout showed those are **DRAM-bandwidth bound** (transition FC1 activation 1–4 GiB/block;
trimul win came from channel movement). Halving the large trunk weight reads is exactly
the lever bf8 provides.

- **Expected:** ~1.1–1.25x on the trunk stage at N≥512 (Boltz-2/Protenix precedent),
  ~1.1–1.2x end-to-end warm; larger at 1024 than at small N. Confirm with an A/B on the
  48-block trunk at N=512/1024 (reuse `scripts/kernel_scout_next_bench.py` harness).
- **Effort:** route ESMFold2 weight loads through a fast-mode-aware dtype instead of the
  `_DTYPE` constant; keep activations/accumulation as today. Mechanism already exists.
- **Gate (blocking):** `scripts/release_gate.py --model esmfold2` must clear 4.0 Å / 0.65
  on the multi-target set; bf8 trunk could hurt distogram/pLDDT. Keep on branch until parity.
- **Risk:** if ESMFold2 trunk turns out dispatch- not bandwidth-bound at the target size,
  win shrinks — the A/B is the cheap way to find out before committing.

### 2. Persistent cross-process kernel cache (cold-start)
**Scope:** all tt-bio models + CLI. **Type:** cold-start/throughput. **Effort:** LOW. **Gate:** none (lossless).

*(measured)* Cold ESMFold2 process = **135 s** (weight load + first-compile of every
shape bucket) vs **2.1 s** warm — a ~64x first-call cost. There is **no on-disk kernel
cache** anywhere in the repo (repo-wide grep for `TT_METAL_KERNEL_CACHE` /
`PERSISTENT_KERNEL` = zero hits); program cache is in-process only
(`tenstorrent.py:293`). Every fresh `predict --devices` worker (`main.py:907-921`, spawn)
and every restarted serve worker recompiles from scratch.

- **Expected:** eliminate the *compile* fraction of cold-start (tens of seconds) for the
  2nd+ process on the same host/shapes. Does not help the very first compile or weight I/O.
- **Effort:** point tt-metal's persistent kernel cache at a shared dir in the spawned
  worker env; verify the installed ttnn honors it; add a cache-version guard on ttnn bump.
- **Applicability caveat:** the serve/controller worker is long-lived (residency already
  closed — see #4), so this mainly helps **ad-hoc CLI**, **fleet scale-up / restarts**,
  and the first job after a deploy. Real, but narrower than the 64x headline suggests.
- **Gate:** none — kernel cache is bit-identical. Lowest-risk item on the list.

### 3. tt-atom: ship the measured-but-OFF wins by default
**Model:** tt-atom (UMA/eSEN). **Type:** bandwidth + dispatch/residency. **Effort:** LOW. **Gate:** light parity (bf8 edge).

Three landed, measured wins default **OFF**, gated behind a source-ttnn build:
`TT_ATOM_DEVICE_EDE` (moves the largest per-step host cost — radial-MLP fwd+bw over E
edges — onto the device/trace, `device.py:device_ede`), `TT_ATOM_BF8_EDGE` (edge-activation
bandwidth win — note bf8 *weights* is the dead end, bf8 *edges* is the win,
`device.py:bf8_edge`), `TT_ATOM_FUSED_LNBW` (fused radial LN backward, `so2.py:38`).

- **Expected:** each is already-measured positive; combined they cut per-step host +
  bandwidth cost meaningfully for MD/relax. Re-measure the stack on the current canon
  master to get the combined number, then flip defaults with capability detection (mirror
  ESMFold2's `fuse_swiglu` build-probe so ttnn-0.68 users keep the safe path).
- **Effort:** flip defaults + add build/capability probe. No new kernel work.
- **Gate:** BF8_EDGE changes numerics slightly → run the force/energy parity check;
  DEVICE_EDE / FUSED_LNBW should be bit-comparable.

### 4. Single-job multi-card sharding for large folds (tensor/model parallel)
**Scope:** tt-bio large single jobs. **Type:** multicard latency. **Effort:** HIGH. **Gate:** parity + real multi-card.

`--devices` is strictly **data-parallel — one whole target per card**
(`main.py:1842-1845`, one spawn per card pinned to one physical device). A single N=1024
fold, or **ESMC-6B on one long sequence**, uses exactly **one** card while the other three
sit idle (`esmc.py:1037-1062` shards the sequence *set*, never one sequence). Nothing splits
one input across cards. This caps the **latency of the largest single jobs** — the case a
user waiting on one big structure actually feels.

- **Expected:** up to ~2–3x latency on large single folds by sharding the pairformer /
  triangle ops across cards; real ceiling set by inter-card comms on the trimul/transition
  matmuls.
- **Effort:** HIGH — mesh-device, sharded matmuls, cross-card reduction. Recommend a
  **scoping spike first**: micro-bench a sharded trimul + transition at N=1024 across 2/4
  cards to measure comms overhead before committing. Orthogonal to the largeseq single-card
  dead end (that was one card).
- **Gate:** bit/PCC parity vs single-card + on-hardware multi-card run.

### 5. tt-atom: trace-replay for repeated single-shot / screening
**Model:** tt-atom. **Type:** dispatch/residency. **Effort:** MEDIUM. **Gate:** parity.

Trace gives 2.14x/step and 2.33x per FIRE relaxation, but only **fixed-topology MD/relax
loops** use it; one-shot `calculate()` and screening `evaluate_batch` get no trace
(`trace.py:3-9`). For small graphs the device fwd+bwd is ~96% of a call and
**host-dispatch-bound**, so a persisted trace replayed across many **same-topology**
single-shots (high-throughput screening of similar systems) is unexploited.

- **Expected:** up to ~2x on repeated identical-topology inference (screening).
- **Effort:** MEDIUM — persist/replay a trace keyed on topology; fall back on topology change.
- **Applicability:** screening workloads only; single arbitrary structures don't benefit.

### 6. tt-bio: model-affinity pinning to stop mixed-model reload ping-pong
**Scope:** fleet under mixed-model load. **Type:** throughput. **Effort:** LOW–MEDIUM. **Gate:** none.

Within a worker, weight residency is already closed (model built once, reused across all
same-model jobs; scheduler has model affinity — `worker.py:612-618`, `distributed.py:221-231`).
The residual cost: under **mixed-model** load a worker can lease a job for a *different*
model when its own model has no waiting work, paying a full `reset()` + `load_model()`
(`distributed.py:228-232`). No per-worker model-pinning knob exists.

- **Expected:** removes sporadic full weight-reload stalls (tens of s) under mixed load;
  no effect on single-model workloads.
- **Effort:** optional per-worker model pin, or raise the model-switch penalty in the
  scheduler rank tuple. **Gate:** none. Lower priority — only bites mixed-model fleets.

---

## Top pick

**#1 — ESMFold2 `--fast` bf8 trunk.** Clearest mechanism (the trunk that was left bf16
while its two siblings already went bf8), targets the dominant warm stage on every
ESMFold2 fold, bounded low–medium effort with the precision path already built in
`tenstorrent._dtype()`, and a proven Boltz-2/Protenix precedent. It needs an accuracy gate,
so it stays on branch until `release_gate.py` passes — but it is the highest expected
warm-latency win per unit effort on the list.

If a lossless, zero-gate win is wanted first, **#2 (persistent kernel cache)** and
**#3 (tt-atom flip-on wins)** are the low-risk quick hits.
