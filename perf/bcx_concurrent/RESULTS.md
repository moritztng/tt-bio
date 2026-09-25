# How many BindCraft 2 arms one QuietBox runs at once

Driver: `perf/bcx_concurrent/sweep.py`. Each arm is `perf/bcx_tracewire/round_ab.py --arm eager
--extra-msa`: BindCraft 2's own campaign driving the shipped device predictor with the extra-MSA
stack on card. Same seed, same settings, same rounds at every N, so the work per arm is identical
and only the number of neighbours changes. Host tt-quietbox2, tree 4f3d2c212 (= wk/bcx-tracewire
f8b9070b2, which carries bcx-extramsa). **The extra-MSA switch was ON in every window here.**

A round is the interval between the campaign's own consecutive `sequence_gradients` calls, so it
carries the round's host work, not only the card's.

**What the arm is.** `round_ab.py:110` pins `design_models=["model_1_ptm"]` and
`validation_model=monomer`: one monomer design model. The shipped `examples/pdl1.json` resolves
five `multimer_v3` models (`state/bcx/MODELPOOL.md`), and its round is about five times this one
(measured below). Every ratio on this page is a ratio of identical arms, so it transfers; the
absolute seconds are the one-model arm's, and a shipped trajectory/hour figure is not this number
divided into 3600.

## The answer depends on what else the box is doing

| run | ambient (threads of other rows) | N | cards | round s/arm | rounds/h box | per-arm | box |
|-----|------|---|-------|-------------|--------------|---------|-----|
| A 08:23Z | 10.13 | 1 | 3     | 10.628 | 338.7 | 1.000 | 1.00 |
| A 08:26Z | 10.19 | 2 | 3,2   | 13.702 | 525.6 | **1.289** | **1.55** |
| B 08:48Z | ~4.8  | 2 | 3,0   |  9.981 | 721.6 | **1.046** | **1.91** |
| C 08:51Z | 4.75  | 1 | 3     |  9.538 | 377.4 | 1.000 | 1.00 |

Every window held 1350 MHz median on every card in use, sampled at 1 Hz from the class node
(`/sys/class/tenstorrent/tenstorrent!N/tt_aiclk`); the lowest single sample anywhere was 1312. No
number here is a degraded-clock artifact.

**On a box that is otherwise free, a second chip is nearly free: 4.6 % per arm, 1.91x per box.**
On a box already running ~10 threads of other work, the same second chip costs 28.9 % per arm and
buys 1.55x. Two arms contend with the rest of the box, not meaningfully with each other.

Ambient is measured, not assumed: run A's two windows sit behind 10.13 and 10.19 threads of
foreign load (box busy 11.91 and 13.68, minus the arms' own 1.78 and 3.49), so the 1.289x is one
added arm and nothing else. An earlier pass on this row read 1.069x because its N=1 control ran on
a box at 15.33 of 16 threads: the control was already paying the contention the measurement was
meant to expose. **On a shared box the control's ambient is part of the measurement.**

## Where the contention is: the host's memory system, not its cores

Per-round medians inside run A's windows, from the arms' own counters:

| | N=1 card 3 | N=2 card 3 | N=2 card 2 |
|---|---|---|---|
| round s            | 10.628 | 13.895 | 13.509 |
| taped (fwd device) | 2.180  | 2.186  | 2.194  |
| backward           | 5.237  | 6.046  | 6.310  |
| extra-MSA          | 0.662  | 0.728  | 0.788  |
| host trunk CPU     | 6.994  | 7.651  | 7.819  |
| rest of the round  | 2.549  | 4.935  | 4.217  |

- **Not the clock.** 1350 MHz median on both cards at both N.
- **Not the device.** Forward device time is 2.180 s at N=1 and 2.186 s at N=2, 0.3 % apart. Two
  cards running their own forward do not slow each other.
- **Not host cores.** Each arm keeps its full 1.78 cores and box busy rises by exactly one arm's
  worth, with 2.3 threads still spare. Core starvation is sub-additive; this is additive.
- **The host's memory system is what is left.** 2.39 s of the 3.27 s the round grows sits outside
  every device call, in a part of the round that nearly doubles, 2.549 -> 4.935 s. The arms hold
  their cores and each core gets less done per second. qb2 is a Ryzen 9700X: one CCD, 8 cores,
  32 MB of L3 shared by everything on the box, and the AF2 host side streams through it.

That is also why the quiet box scales: at run B the total demand was about 8.8 of 16 threads, at
run A about 13.7. The cost appears as the box approaches saturation, not per added arm.

## The shipped five-model program says the same thing, on a run we never touched

`bcx-multimer`'s device arm ran the shipped pool (five `multimer_v3`) on card 0 throughout, one
trajectory, 125 gradient steps, its per-step times in `pool_selections.jsonl`. Comparing only
steps inside its **anneal** stage, so the stage is not the variable:

- 23 anneal steps while the box was shared: **54.22 s** median
- 22 anneal steps of the same stage once it quieted: **35.30 s** median

A 1.54x swing on the same trajectory, same card, same stage, from box load alone. It also fixes
the shipped round at roughly 5x the one-model arm used here (54.22 s against 10.628 s at
comparable ambient), which is what the five design models cost.

## Against the fleet figure

`state/bcx/THROUGHPUT.md` gives 3.28 trajectories/hour, from 6,591 chip-s per trajectory on one
chip times six usable Blackhole chips fleet-wide. The suspect step was the multiplication: the
worry was that a chip cannot cost the same when six run at once, since `bcx-seam` had the round at
79.2 % host work.

**Measured, the multiplication is close to right, on one condition.** Post-extramsa the round is
55.8 % device, an arm draws 1.9 cores of 16, and a second chip on an idle box costs 4.6 %. The
host-contention fear does not survive the extra-MSA swap. What does survive is that the box must
be free: the same second chip costs 28.9 % when ~10 threads of other work are already resident, so
a fleet figure computed from chips alone silently assumes nothing else runs on those hosts.

## Scope and what remains open

Blackhole only, on qb2. The 128 Osaka chips are Wormhole and nothing here transfers to them.

N=3 was attempted at 08:48Z on cards 3, 0 and 2 and the card-2 arm was refused at device open:
`bcx-multimer` released card 0 at 08:48:53 and took card 2 seconds later, and `tt_bio`'s
`CardSetLease` correctly refused a concurrent open after waiting 120 s. The cards 3 and 0 arms of
that attempt completed and are the run-B row above. qb2 has four chips, card 1 is `of3t-infab`'s
hard pin, and the third free chip has had a holder for this row's whole life, so N=3 needs a
window where one BCX row is between device sessions.

Rows live on the box during these windows: `bcx-multimer`'s device arm (card 0, then card 2), an
of3t/land gate on card 1, the `bcx-predictor` CPU reference arm at ~2.5 cores, and the mgx
BoltzGen CPU reference. The device rows were correctness runs rather than timing runs, so this
sweep did not corrupt a timing measurement; it did slow them by the cores it took, which is the
54.22 s column above.
