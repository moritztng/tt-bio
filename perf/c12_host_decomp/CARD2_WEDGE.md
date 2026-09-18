# 2026-09-17 — qb2 card 2 cannot dispatch a 32x32 add. No fold was taken.

Board `…410D` (nodes 2 and 3) has both chips stuck. This is the reason `runs/sweep3` carries no
rows, and it is a hardware finding, not a scheduling excuse.

## What happened

`decomp.py --size 512 --node 2` opened `/dev/tenstorrent/2` at 15:38:01Z and never reached a fold.
At 15:48:16Z, ten minutes in, `py-spy dump` put the main thread here:

    _assert_local_dispatch (tt_bio/tenstorrent.py:5007)   <- ttnn.synchronize_device(dev)
    _open_and_init_device  (tt_bio/tenstorrent.py:5422)
    get_device             (tt_bio/tenstorrent.py:5051)
    build_fold             (tt_baseline.py:361)

Line 5007 is the `synchronize_device` of tt_bio's own bring-up probe: `ttnn.add` on a single
`[32,32]` bf16 tile. The op was issued and never completed. Beside it:

| signal | reading |
|---|---|
| node 2 board power | 32-35 W across four reads 2 min apart (a real 1350 MHz fold draws 77.4 W, `c10-fixed-cost`) |
| node 2 AICLK | 1350 MHz, unforced by this run: the clock is pinned per label and no label started |
| process CPU | one thread at 99.7 %, main thread `idle` (blocked), RSS 1.6 GB |
| kernel compile | none: no `riscv`/`sfpi` child, nothing written under `~/.cache/tt-metal-cache` or `/tmp/ttnn` in the preceding 3 min |
| dmesg, bus 03:00.0 | nothing at all since boot |

Host spinning, device idle, no compile, no kernel error. The chip accepted the program and did not
run it.

## Why it is the board and not this row

- Node 3, the other chip of `…410D`, has carried a wedged holder since 14:37Z
  (`hall-capacity-800aa` pid 1526759, its own state doc calls it a wedged orphan).
- Node 2 itself worked earlier the same day: `c12-compose-fold` held it and released cleanly at
  14:28Z. So the chip degraded between 14:28 and 15:38.
- The 15:47:50Z event in `dmesg` is bus `01:00.0` coming back up (`FW log forwarding enabled`,
  `ASIC entering A0 state`), which is the *other* board, `…4103`, being reset by someone else. It
  is 10 min after this wedge started, so it is not the cause. It did clear the card-0 orphan:
  node 0's `tt_heartbeat` reads 568 against 199642 on the three chips that did not reboot.
- Both `…410D` chips idle at **1350 MHz**, where a healthy idle chip on this box reads 800. The
  governor is not the one the orchestrator described at 11:3xZ.

`state/qb2-board410d-down` was lifted at 11:3xZ on the evidence that the board had not caused the
recent reboots. It was not lifted on evidence that the board can dispatch a program, and a
32x32 add is the cheapest test of exactly that.

## What was done about it, and what was not

`SIGTERM` was sent to pid 1571080 at 15:49:14Z. It did not clear: the thread is blocked below
Python, so the handler that releases the clock and the device never runs. It was **not** SIGKILLed
(standing rule, and it would not free the chip anyway). The outer `run.sh` and `benchlock.sh` pids
were terminated explicitly so the mutex released, and the stale `state/benchlock` holder line for
the dead pid 1570807 was cleared by hand.

So this row now holds a wedged chip on node 2 and says so. Recovery is `tt-smi -r 2`, which on a
p300c is **board-pair granular**: it would reset nodes 2 and 3 together, clearing this wedge and
`hall-capacity`'s node-3 orphan in one go and touching neither chip of the release gate's board.
Nothing of value is running on either chip of `…410D`. This row's brief says explicitly not to
take that reset but to say that its card needs one, so: **node 2 needs a board-pair reset of
`…410D`, and the reset destroys nothing.**
