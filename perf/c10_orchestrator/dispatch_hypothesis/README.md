# What the 3.952 s is probably made of, and the lever that already exists for it

The clock-immune term of the 512 aa fold is 3.952 s (`../fixed_cost/`). Clock-immune is not the
same as host CPU, so the next question is what is inside it. Three numbers from the existing
corpus narrow it a long way, and none of them needed a chip.

**The fold issues 465,664 top-level `ttnn` calls at 512 aa** (`perf/roof_launch/op_census_512.json`).
Against 3.952 s that is **8.49 us per call**, or 13.8 us per call that actually launches a program.
Eight microseconds is the right order for Python-side dispatch on this stack, not for anything the
device does.

**The device's own launch cost is only 0.487 s of the fold** (`perf/roof_launch/LAUNCH_FLOOR.md`,
measured per op class under trace replay on Blackhole: 4.13 to 54.96 us fixed depending on class,
not the 20.6 us Wormhole constant). That is **12 % of the clock-immune term**. So roughly seven
eighths of the 3.952 s is not device launch.

**A third of the calls are pure host bookkeeping.** The census's top entry is `ttnn.deallocate` at
122,112 calls, and `ttnn.Tensor.__getitem__` appears 12,776 times beside exactly 12,776
`ttnn.slice` calls because it *is* those calls. The launch-floor row found 150,160 of the 465,664
launch nothing at all.

## The lever is already written and has never been priced

`tt_bio/tenstorrent.py` contains a ttnn trace of the per-step DiT device stream — `_capture_diff_trace`
and `forward_traced`, opt-in through `Boltz.__init__(diffusion_trace=True)`. Its own comment states
the case: the diffusion loop is shape-stable across sampling steps, so one captured graph replays
every step and **collapses the per-step host dispatch**, and it is *"lossless by construction — the
replayed graph is the exact captured program with new input buffer contents, so the output is
bit-identical to the untraced forward"*. ESMC, Protenix and RFdiffusion3 all use trace replay in
this same repo.

Boltz-2 does not. Every measurement in this project's corpus records `diffusion_trace=False`, and
`perf/b2x-flag-levers/ab_flag_levers.py` labels it *"the published 23.504 s cell's protocol"* — it
was treated as part of the protocol, never as a lever. No corpus entry prices it.

The diffusion sampler is worth attacking: on the older tree that flag sweep measured
`sampler_s` 7.957 s of a 25.054 s fold at 39.62 ms per step over 200 steps, about a third of the
fold, against `trunk_s` 13.863 s and `confidence_s` 1.463 s.

## Why this is worth a chip before any kernel work

A lever with **zero accuracy spend** (bit-identical by construction, so bit-exactness is free to
check rather than something to trade away), **already implemented**, attacking a term worth
**3.952 s of a 14.881 s fold**. If trace replay removes even half the dispatch cost of the
diffusion loop, that is a larger single step than anything in the corpus, and it costs one
interleaved A/B session to find out.

## What would refute it

- The clock-immune term is mostly featurization, MSA handling and CIF writing rather than dispatch,
  in which case tracing the diffusion loop moves almost nothing. The fixed-cost row's 298 aa arm
  starts to separate these, since featurization scales with the target and dispatch scales with
  call count.
- The Boltz-2 fold path's diffusion loop is not shape-stable the way BoltzGen's is, so capture
  fails or has to re-capture per step.
- The 1 GiB trace region does not fit beside the 512 aa working set, or it pushes a size users have
  today into OOM. That is a hard stop, not a trade.

Numbers here are arithmetic on committed artifacts. Nothing in this note was measured on a device
by its author, and the 8.49 us per call is a ratio of two numbers from different trees: the call
census is from the roof campaign's capture and the 3.952 s from commit `0df13ad9`.
