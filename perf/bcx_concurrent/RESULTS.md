# How many BindCraft 2 trajectories one QuietBox runs at once

Measured on tt-quietbox2 (Ryzen 9700X, 8 cores / 16 threads, four Blackhole cards) on
2026-09-25, on `wk/bcx-concurrent` = `wk/bcx-tracewire` f8b9070b2, which contains bcx-extramsa.
**The extra-MSA switch was ON** (`round_ab.py --extra-msa`), so this measures the program the
campaign is moving to, not the one it is leaving.

Each arm is one BindCraft 2 campaign driving the shipped device predictor: same seed (100), same
settings, same bucket, same number of rounds at every N. A round is the interval between the
campaign`s own consecutive `sequence_gradients` calls, so it is the loop`s own round with all of
its host work inside it, not a harness step.

    perf/bcx_concurrent/sweep.py --cards 3,0 --n 2 --rounds 16

| N | cards | round s/arm (median) | per-arm vs N=1 | rounds/h box | box vs N=1 | AICLK median (min) | box busy threads | loadavg |
|---|-------|----------------------|----------------|--------------|------------|--------------------|------------------|---------|
| 1 | 3     | 12.397               | 1.000          | 290.4        | 1.00       | 1350 (1312)        | 13.77            | 22.1    |
| 2 | 3, 0  | 13.254               | 1.069          | 543.3        | **1.87**   | 1350 (1306/1325)   | 13.97            | 23.6    |

Both windows ran at 1350 MHz, the full Blackhole clock, sampled every second from the class node
during the window. Round counts are the 12-14 rounds inside the window where every arm was past
its compile rounds and none had finished.

The N=1 row is the control taken at 07:00Z, right after the N=2 window, at a matched box load. An
earlier N=1 at 06:50Z on a busier box (15.33 threads) read 12.975 s, so ambient load from other
rows moves a round by about 5 %, and the N=2 window sits inside that bracket.

## Where the contention is, and it is not the host

Each arm draws **1.67 cores** of the box`s 16 hardware threads while its round runs. Two arms
together draw 3.3. The host share that made this question urgent was measured before the
extra-MSA stack moved onto the card: bcx-seam had a round at 79.2 % host, and on that program
four arms on eight cores really would have fought. On the post-extramsa program a round is
55.8 % device and its host side is nearly single-threaded, so two arms fit in the idle part of a
box that was already 14 of 16 threads busy with unrelated work. The 6.9 % per-arm cost at N=2 is
in the same range as the ambient drift, and no arm lost clock.

## Trajectories per hour

At 125 gradient rounds per trajectory, the basis `state/bcx/THROUGHPUT.md` uses:

    N=1   12.397 s x 125 = 1550 s   2.32 trajectories/h on one chip
    N=2   13.254 s x 125 = 1657 s   2.17 per arm, 4.35 trajectories/h on the box

THROUGHPUT.md`s 3.28 trajectories/h came from 6,591 chip-s per trajectory (52.7 s/round) times
six chips, on the pre-extramsa program. Two chips of this box already beat it. The part of that
figure this row was asked to test is the assumption that a chip costs the same whether one or
several are running, and at N=2 that assumption holds to within 6.9 %.

N=3 is unmeasured. qb2 has four cards, card 1 is another campaign`s hard pin, and cards 0 and 2
were both held by other BCX rows doing device work for this whole pass.
