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

    bash perf/c12_host_decomp/controls.py          # no device needed
    bash perf/c12_host_decomp/run.sh <name> <node> 512 298

`run.sh` refuses a node whose `tt_aiclk` is unreadable, wraps every timed run in `benchlock.sh`,
and writes `runs/<name>/table.json`. No production code is changed: every patch is a timing
wrapper installed in the capture process, and `decomp.py` refuses to run if the production diff
against `criterion.json`'s base is non-empty.
