# The two measurements this row could not make, written to be handed to a chip

`b2z2-bh-tile-census` held no card. Everything in `REPORT.md` is arithmetic over committed
Blackhole artifacts, and it is enough to redirect the wave, but two of its claims rest on a fit
rather than on a measurement. Both experiments below are cheap, neither changes model code, and
neither needs more than one qb2 Blackhole chip for well under an hour. pc's card miscomputes
silently, so neither may run there.

## E1 — the diffusion step has never had its stall accumulators read (30 min, 1 chip)

**Why.** `b2z-kernel-cycle-census` turned on `--enable-sum-profiling` for the PairformerLayer only.
The DiffusionStep census (`ops_perf_step_qb2c0.csv.gz`, 1066 programs) has `DEVICE COMPUTE CB WAIT
FRONT` blank in every row, so **the sampler's input-tile wait is unmeasured on any architecture.**
The campaign has been assuming the trunk's 57.0 % transfers to it. This row's count says it should
not: the sampler moves 2,340.6 MB of DRAM reads per step against the trunk's 4,710.8 MB per block,
in a step that is now 26.40-26.60 ms against the block's 36.34 ms, so its delivery pressure is
about half the trunk's.

**Do.** Exactly what `b2z-kernel-cycle-census` did for the block, pointed at the step. On qb2,
source build at `/home/ttuser/tt-metal-b2z` (tt-metal `1452925b` = the shipped ttnn 0.68.0 tag,
`ENABLE_TRACY=ON`), env recipe `/home/ttuser/scratch/b2z_env.sh`:

* `python -m tracy ... --enable-sum-profiling`, venv `bin` on `PATH` or the child dies with
  `ModuleNotFoundError: loguru` and the parent misreports it as "not a Tracy build".
* the legacy post-processor needs the pandas-3 patch already applied in that tree
  (`tools/tracy/process_device_log.py:143`, two `df.iloc[:, n] =` -> `df.isetitem(n, ...)`).
* 3 fenced repetitions of one DiffusionStep at 512 aa on the cell's fixture, same as the block.

**Predicted, before it runs.** Input-tile wait **25-40 %** of TRISC1 on the step, against 57.0 %
on the block. If it comes back near 57 % the sampler is delivery-bound too and the L1-residency
lever applies to both tracks. If it comes back near 25 % the sampler is dispatch- and
latency-bound, its 1066 programs at 20.7 us mean kernel are the problem, and byte levers aimed at
it are mispriced.

## E2 — the two delivery bandwidths are fitted, not measured (20 min, 1 chip)

**Why.** `REPORT.md` fits the per-site wait to bytes delivered and gets **352.2 GB/s** for
DRAM-interleaved reads and **651.9 GB/s** for L1-interleaved reads (R2 0.79 over 40 sites). The
DRAM figure is 79 % of the measured 444.9 GB/s roof and is believable; **the L1-interleaved figure
has no independent measurement behind it at all**, and the 1.188x this row prices for moving the
trunk's DRAM operands into L1 is exactly the ratio of those two numbers.

**Do.** A reader-only microbenchmark, no model: on 110 cores, stream N tiles per core out of a
DRAM-interleaved tensor into a CB and drop them, then repeat with the identical tensor in
`L1_INTERLEAVED`. Take the rate as a **slope over N** so program setup cancels, min of 3, n>=5,
interleaved A/B in one process. `perf/b2z2_datum_rate/` on `wk/b2z2-datum-rate-floor` is the
working pattern for a `ttnn.generic_op` harness with a barrier-only control; reuse it rather than
writing a third one. Sweep the tile stride too: the trunk reads a pair tensor whose tiles are
32 rows apart in a `[1, 512, 512, 128]` layout, and a contiguous sweep will flatter the interface.

**Predicted.** DRAM-interleaved read 330-380 GB/s on 110 cores; L1-interleaved 600-900 GB/s.
If L1-interleaved comes back under 450 GB/s, the residency lever is worth well under 1.188x and
this row's recommendation weakens accordingly. If it comes back above 1 TB/s, it is worth more.

## What neither experiment needs

No model code changes, no parity arm, no benchlocked fold. Both are instrument runs. E1 reuses a
recipe that has already run once on this exact box; E2 reuses a harness that has already run 26
arms on two chips.
