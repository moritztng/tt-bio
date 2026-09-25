# Killing a wedged fold that holds a device fd hard-resets qb2

Three times on 2026-09-22. The third one I caused, with a harness I wrote this pass, and it is
what turned a hypothesis into a reproducer.

## The chain

    fold wedges  ->  harness timeout kills the process group  ->  card left un-reinitialisable
      ->  next device open logs "Failed to set initial power state: -5"  ->  host hard-hangs

`-5` is EIO. The kernel log does not end with a shutdown sequence; it stops mid-session.

## The three instances

    boot -3   died 2026-09-22 14:36:52Z
    boot -2   died 2026-09-22 17:20:15Z   size-ladder killed openfold3-768-rep0, wedged 30 min
    boot -1   died 2026-09-22 20:45:36Z   narrowq_quiet.sh killed rf3-896 warmup, wedged 300 s

Boots -2 and -1 both log the error against **the same PCI address as the card the killed fold was
on**, `0000:01:00.0` = card 0.

## Instance three, timed to the second

    20:39:17Z  narrowq_quiet.sh takes benchlock, pre-flight clean: loadavg 0.06, cards 0 and 1
               idle, no foreign fold. Starts the rf3 896 aa warm-up on card 0.
    20:44:17Z  300 s --fold-timeout-s fires. fold_ab_flip kills the process group.
    20:44:19Z  the harness moves straight to the 768 aa cell and opens card 0 again.
    20:45:21Z  kernel: tenstorrent 0000:01:00.0: Failed to set initial power state: -5
    20:45:21Z  ..repeats ten times over fifteen seconds..
    20:45:36Z  log ends mid-session. No shutdown. Hard hang.
    20:48:15Z  boot 0.

## What my harness did wrong, specifically

`narrowq_retake.sh`, the version from 02:10Z, ran `~/.local/bin/tt-smi -r "$CARD"` after a failed
attempt, with a comment saying exactly why: "`tt-smi -r` between attempts clears the card the
killed fold left dirty". I rewrote it as `narrowq_quiet.sh` to add a refusing pre-flight, dropped
the retry loop as unnecessary, and **dropped the reset with it**. Sixty-two seconds after the next
device open, the host was gone. The reset was not incidental to the retry; it was the thing
standing between a killed fold and a dead host.

## The same hole is in the release gate, and it is unfixed

`scripts/release_gate.py` `_run_fold` (line 1629) kills a timed-out fold via `_kill_group`
(line 1393), which escalates SIGTERM then SIGKILL, and then **returns to the next leg with no
card reset**. That is boot -2's death. `_kill_group`'s docstring is careful about orphaned
device-holding children and says nothing about the card the killed child leaves behind, because
until today the failure looked like a wedged card rather than a dead host.

SIGINT before SIGTERM is not on its own the repair. A wedged fold spins inside ttnn C++, where a
Python-level signal handler does not run between bytecodes, so the polite signal is likely to be
ignored by exactly the process that needs it. **The repair is to reset the card after killing
anything that held its fds, before the next open.**

## The other half: rf3 at 896 aa wedges reproducibly

Twice on this card. `narrowq_retake.sh`'s own comment records the first: "an rf3 896 aa leg stopped
at 'trunk 1/10' with its device child spinning at 100 % CPU and no progress for 11 minutes, on a
card that had just folded the same shape in 101.7 s." The second is 20:39Z above, on an empty box
at loadavg 0.06. 896 aa is a size users can submit, so this is worth a row of its own; it is not
a property of the narrow-q lever, which was not even enabled on the warm-up leg.
