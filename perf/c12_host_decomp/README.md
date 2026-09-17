# c12-host-decomp — what F is made of

`c10-fixed-cost` fitted `T = F + C/f` over four pinned AICLK arms and got **F = 3.9830 s at
512 aa** and **1.9500 s at 298 aa**, with work terms 14665.0 and 10403.4 Mcycles. F is 26.8 % of
the 14.881 s fold and it does not move across a 1.69x clock range. This directory decomposes it
into named line items and says, per item, whether that item's own seconds moved when the clock
moved.

## Why two clocks and not just a profiler

A host block that is really a synchronisation wait on the device looks exactly like host time to
any Python profiler, and it moves when the clock moves. So every item is measured at 1350 and at
800 MHz and split by the same two-parameter form the fold itself obeys:

    F_i = 2.454545 * t_i(1350) - 1.454545 * t_i(800)
    C_i = 1963.636 * (t_i(800) - t_i(1350))        Mcycles

The split is linear, so if the items partition the fold then `sum(F_i) = F_fold` as an identity.
Closure is then arithmetic on measured numbers and any gap is a gap in the partition. An item is
called CLOCK-IMMUNE, CLOCK-SCALED, MIXED or UNRESOLVED against its **own** rep spread, never a
fixed percentage.

## Files

| file | what |
|---|---|
| `decomp.py` | the capture. One process and one device context per size, arms interleaved inside each clock, clock order reversed on odd reps, FORCE_AICLK before every label, ~1 kHz during-fold clock and board power. |
| `fit.py` | the two-clock split, closure, per-item 512/298 scaling, perturbation column. |
| `controls.py` | CPU known-answer checks, including negative controls that break what the checks read. |
| `evidence.py` | holder census, clock sampler, coverage scored against each fold's own target. Node-parameterised from `perf/c10_bare_baseline/control.py`. |
| `force_aiclk.py` | ARC FORCE_AICLK over tt-kmd's SMC queue, verbatim from `c10-fixed-cost`. |
| `criterion.json`, `prediction.json` | the protocol and the predictions, both recorded before any fold. |

The region tree is **imported** from `perf/b2x_host_residual/host_residual.py`, not copied, so
this row and the `b2z2-host-residual-round2` census cannot drift apart. `decomp.py` adds the spans
that instrument did not need: `Boltz2.forward`'s own body, the trunk's static build, the device
pair-assembly stages, and the program-cache clear that runs on every forward.

## Running it

`run.sh` is the entry point. Do not call `decomp.py` directly: the guards below live in `run.sh`,
and without them a dead chip costs a whole pass rather than a minute.

    perf/c12_host_decomp/controls.py               # no device needed
    perf/c12_host_decomp/dispatch_probe.py --node <n>   # optional, run.sh does it for you
    bash perf/c12_host_decomp/run.sh <name> <node> 512 298

`run.sh` refuses to measure unless four things hold, in this order, before it takes `benchlock` or
opens anything:

| check | why | on failure |
|---|---|---|
| `tt_aiclk` readable on the node | a card that is off the bus | exit 1 |
| `pair_idle.py --card <n>` | a p300c's two chips share one board power budget, so a fold on the sibling moves this node's timing while never touching the lock file. `benchlock` cannot see it: its contract is mutual exclusion among `benchlock` callers | exit 75, retry later |
| `dispatch_probe.py --node <n>` | whether the chip can run a program at all, in a child under a timeout. tt_bio's own bring-up probe is the right test but runs inline in the process that then holds the device, so it has no deadline | exit 1, the node needs a reset |
| `benchlock.sh` | the box-wide timed-run mutex | exit 75 |

`pair_idle.py` is read out of `origin/wk/c12-orchestrator` rather than copied, so this row and the
orchestrator's cannot drift apart. `PROBE_TIMEOUT_S` defaults to 240.

No production code is changed: every patch is a timing wrapper installed in the capture process,
and `decomp.py` refuses to run if the production diff against `criterion.json`'s base is non-empty.
It also refuses if the load it cannot account for exceeds 1.0 — every foreign device holder is
priced off its own `/proc` ticks and the remainder is what gates, because an absolute load average
becomes unpassable rather than stricter once a wedged holder parks a core on the box.

## What it closes against

The measured non-device remainder, **1.6489-1.6720 s**, not F. `c12-profiled-fold` measured the
device term in situ at 13.2090 s of 14.8810 s, and F = 3.9830 s exceeds that remainder by 2.31 s
of clock-immune device cost. Closing this table against F would mean padding it with 2.3 s of work
that is not host work. `fit.py` carries both, with the remainder primary.

## First cut, and what it still owes

`host_firstcut.py` combines two committed on-card measurements into a first cut with no device:
`b2x_host_residual`'s closed 128 aa and 512 aa host trees from this same card, classified leaf by
leaf into host-today, moved-to-device-since, device, and mixed. It is not this row's table, carries
no clock arm, and says so. What it found: every host item that can be named is 0.0000 s reducible
against the 0.8490 s bar, and the whole question collapses onto 0.3997-0.7738 s inside
`Boltz2.forward`'s own body, which no committed instrument has ever bracketed. So run the
`TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE` A/B and the `forward` brackets FIRST, ahead of the two-clock
arms.
