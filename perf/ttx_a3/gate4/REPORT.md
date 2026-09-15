# Re-gate attempt 4: the gate moved off qb2, and the card pin did not hold

Status: **HOLD, unchanged.** The default stays OFF and the branch stays unmerged. This pass did not
produce a gate verdict. It established where the gate can run at all, and it found one defect in
the move that has to be fixed before the next launch.

The lever itself is not in question and was not re-measured: GO on the evidence in `../gate/REPORT.md`
and `../gate2/REPORT.md`, and the branch is already merged up to `origin/main` with the flag default
reading `True` (`tt_bio/tenstorrent.py:1827`, verified by import on qb1's checkout of the branch tip).

## Blocker 1 (weights outage): not retired, not reached

`793a2ebaa` and `095ae476c` are on the branch through the `5aef1d601` merge, so the pins are present
in the tree. Whether esmfold2 and esmc-6b actually load is a question only the capacity and perf arms
answer, and no arm got that far. Carry it forward as unverified rather than fixed.

## Blocker 2 (qb2): worse than "still resetting" — qb2 is gone

qb2 is not reachable at all. `ping -c2 100.105.31.41` is 100 % loss and `ssh` connect-times-out;
it is not answering on the tailscale address the ssh config pins it to. This is the silent host
hang, not the watchdog reset the last three passes worked around, and it matches QBROOT's closed
verdict that the root cause is a per-card PCIe link failure whose next lever needs hands on the
box. No amount of gate-side resumability gets past a box that is not there.

So the gate has to move, and this pass moved it.

## Where the gate can run

- **pc** is out for the accuracy arms. Its single p150a is the card root-caused in
  `pc-card0-512aa-fold-nondeterminism`: it miscomputes some matmuls at a low, location-keyed rate at
  every size, so a bit-exact or hash-equality result measured there means nothing. It also runs a
  custom 130-core firmware, so its timings are not the p150a the baselines were recorded on.
- **qb1** is the answer. Four Blackhole p150a cards, idle (no worker processes, no card holders),
  `tt-bio-dev/env` imports ttnn and torch 2.11.0, 204 GB free. `docs/size_ladder_baseline.d/` carries
  p150a cells for 8 of the 9 ladder models, the exception being `protenix-v1`, which is p300c-only and
  will read as a coverage gap on qb1 for reasons that predate this lever.

Two hardcodings had to come out to get there, both now fixed and pushed: `gate_drive.sh` passed a
literal `tt-quietbox2:$CARD` to `capacity_gate.py` and `full_parity_gate.py` while every other path
was already parameterized, and `resume_after_boot.sh` held gate3/card0/qb2 as constants, which would
have had cron resume a different run on a different card than the driver this pass started, against
a progress file the done-guard never reads.

## The defect this pass found: the card pin did not take

The driver was launched with `GATE_CARD=3`, which sets both `TT_VISIBLE_DEVICES=3` and
`TT_BIO_LEASE_CARDS=3`. The fold that came up under it was holding `/dev/tenstorrent/0`:

    lsof -t /dev/tenstorrent/0  ->  15281
    /proc/15281/environ         ->  TT_VISIBLE_DEVICES=3
                                    TT_BIO_LEASE_CARDS=3
                                    TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge

and the first arm's own traceback names node 0 too, not node 3:

    Failed to allocate TLB window. Look at /sys/kernel/debug/tenstorrent/0/mappings

Two separate things are wrong here and they should not be conflated:

1. **A grant of card 3 opened node 0.** `TT_VISIBLE_DEVICES` is a UMD logical id and not a device
   node, which is already known; what is new is that on this box the two do not agree, so a gate
   that believes it is on card 3 is in fact on the card `fleet.sh`'s `pick_card()` hands out first.
   That is a collision waiting to happen, not a cosmetic mismatch.
2. **The lease did not refuse it.** `TT_BIO_LEASE_CARDS=3` is supposed to make any open of another
   card fail at the device open. The process opened node 0 and ran. Whatever the lease compared, it
   was not the node that got opened, so on this host the lease is not the backstop it is documented
   to be. This wants its own look before any gate is trusted to stay inside its grant here.

`neut768-off1` failed separately, `rc=1` in 14 s, on `tt_tlb_alloc failed with error code -12` for a
2 MB TLB window: ENOMEM against a card whose TLBs another process held. `lsof` did show a holder on
node 3 in the second before launch, so the box was not as idle as the earlier sweep said. The
following arm opened fine, which makes this transient contention rather than a broken card, but it
is the second reason not to relaunch until the pin is trusted.

Everything this pass started is reaped. All four card nodes read free by explicit-pid `lsof`, no
driver is running, `perf/ttx_a3/gate4/PAUSE` is in place so the cron relaunch stands down, and the
owner file is removed.

## What the next pass does, in order

1. Establish the real logical-id -> `/dev/tenstorrent/N` mapping on qb1 and fix `run_arm` to pin by
   whatever the open actually honours, then prove it: launch one arm, `lsof` the node, and refuse to
   continue unless the node is the granted one.
2. Work out why `TT_BIO_LEASE_CARDS` did not refuse the mismatched open. Until then a gate here is
   one bad pin away from taking a card the fleet has promised to someone else.
3. Only then relaunch the driver, remove `PAUSE`, and install the `@reboot` + `*/10` cron.

## Reused, not rebuilt

`perf/ttx_a3/gate_drive.sh` and `resume_after_boot.sh` (the resumable driver and its relaunch, both
now host-parameterized rather than replaced), `perf/ttx_a3/fold_parity_a3.py` for the neutrality
folds, and the banked 1536 aa ratio and below-cap digests, which this pass did not re-measure.
