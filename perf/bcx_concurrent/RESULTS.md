# How many BindCraft 2 trajectories one QuietBox runs at once

Driver: `perf/bcx_concurrent/sweep.py`. Each arm is `perf/bcx_tracewire/round_ab.py --arm eager
--extra-msa`, BindCraft 2's own campaign driving the shipped device predictor with the extra-MSA
stack on card. Same seed, same settings, same rounds at every N, so the work per arm is identical
and only the number of neighbours changes. Host tt-quietbox2, tree 4f3d2c212 (= wk/bcx-tracewire
f8b9070b2, which carries bcx-extramsa). The extra-MSA switch was ON for every window here.

A round is the interval between the campaign's own consecutive `sequence_gradients` calls, so it
carries the round's host work too, not just the card's.

## The numbers

Run A, 2026-09-25 08:23-08:31Z, `sweep.py --cards 3,2 --n 1,2 --rounds 14`:

| N | cards | round s/arm | rounds/h box | per-arm vs N=1 | box vs N=1 | AICLK med (min) | box busy thr | arm cores |
|---|-------|-------------|--------------|----------------|------------|-----------------|--------------|-----------|
| 1 | 3     | 10.628      | 338.7        | 1.000          | 1.00       | 1350 (1325)     | 11.91        | 1.78      |
| 2 | 3,2   | 13.702      | 525.6        | 1.289          | 1.55       | 1350 (1325/1331)| 13.68        | 1.78/1.71 |

At 125 gradient rounds per trajectory: **2.71 trajectories/h at N=1, 4.20 at N=2**.

Both windows held the full 1350 MHz Blackhole clock, sampled at 1 Hz from the class node
(`/sys/class/tenstorrent/tenstorrent!N/tt_aiclk`), so neither number is a degraded-clock artifact.

## The control has to sit at the same ambient as the measurement

An earlier pass on this row read 1.069x per arm and 1.87x per box from cards 3 and 0. That pass's
N=1 control ran on a box at 15.33 of 16 busy threads, so the control was already paying most of
the contention the N=2 number was supposed to expose. Run A's two windows sit behind the same
~10 threads of other rows (box busy 11.91 -> 13.68, a delta of 1.77 that is exactly the one arm
added), and the cost triples to 1.289x. **On a shared box the ambient of the control is part of
the measurement.** Run A is the number to use.

## Where the contention is: the host's memory system, not its cores

Per-round medians inside each window, from the arms' own counters:

| | N=1 card 3 | N=2 card 3 | N=2 card 2 |
|---|---|---|---|
| round s            | 10.628 | 13.895 | 13.509 |
| taped (fwd device) | 2.180  | 2.186  | 2.194  |
| backward           | 5.237  | 6.046  | 6.310  |
| extra-MSA          | 0.662  | 0.728  | 0.788  |
| host trunk CPU     | 6.994  | 7.651  | 7.819  |
| rest of the round  | 2.549  | 4.935  | 4.217  |

Four candidates, three of them ruled out by these windows:

- **Not the clock.** 1350 MHz median on both cards at both N, minimum 1325.
- **Not the device.** The forward device time is 2.180 s at N=1 and 2.186 s at N=2, a 0.3 %
  difference. Two cards running their own forward do not slow each other.
- **Not host cores.** Each arm keeps its full 1.78 cores at N=2, and box busy rises by exactly one
  arm's worth. Core starvation is sub-additive; this is additive, with 2.3 threads still spare.
- **The host's memory system is what is left.** 2.39 s of the 3.27 s the round grows is in the
  part of the round outside any device call, and that part nearly doubles, 2.549 s -> 4.935 s. The
  arms hold their cores and each core gets less done per second. qb2 is a Ryzen 9700X: one CCD,
  8 cores, 32 MB of L3 shared by everything on the box, and the AF2 host side streams through it.

That distinction decides the fix. More host cores would not buy much; memory bandwidth, or moving
more of the remaining host work onto the card the way bcx-extramsa did, would.

## Against the fleet figure

`state/bcx/THROUGHPUT.md` has 3.28 trajectories/hour per box, from 6,591 chip-s per trajectory on
one chip multiplied by six usable chips. Both of its inputs have since moved: the round is
10.6 s rather than 52.7 s with extra-MSA on the card, and qb2 has three usable chips, not six,
because card 1 is a hard pin. Two chips of this box already do 4.20 trajectories/h, above the
figure the arithmetic gave for six.

The multiplication's assumption is the part that does not survive. A second chip does not come
free: it costs 28.9 % on both arms, so two chips buy 1.55x, not 2x.

## Scope

Blackhole only, on qb2. The 128 Osaka chips are Wormhole and nothing here transfers to them.
Rows live on the box during Run A: bcx-multimer's device arm on card 0 (pool_full, a correctness
run), an of3t/land gate on card 1, the bcx-predictor CPU reference arm at ~2.5 cores, and the mgx
BoltzGen CPU reference. The two device rows were correctness runs, not timing runs, so this sweep
did not corrupt a timing measurement; it did slow them by the cores it took.
