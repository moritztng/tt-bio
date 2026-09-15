# Re-gate attempt 4: qb2 flaps on a minutes cycle, and a qb1 move exposed a bad card pin

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

## Blocker 2 (qb2): flapping on a minutes cycle, and I misread it once

Corrected mid-pass, because the first reading was wrong and the wrong reading is the interesting
part. qb2 answered neither ping nor ssh at 19:46Z and 19:48Z, and I wrote that up as the box being
gone for good. It answered at 19:53Z with an uptime of 6 minutes, so those two probes landed inside
a reboot window, not on a dead box. One minute later, at 19:55Z, ssh timed out again.

So qb2 is not gone and it is not stable either: reachable and unreachable inside a ten-minute span,
which is the reset cadence getting worse rather than a new failure. **Two unreachable probes a
couple of minutes apart cannot tell a dead box from a rebooting one — only an uptime reading after
it answers can**, and on this box that distinction decides whether you take it out of fleet
dispatch. I had `state/qb2-ready` staged for removal, which would have stopped every card dispatch
to qb2 fleet-wide, and the ping guard in front of that rename is the only reason it did not happen.

The gate is genuinely making progress there on its own, which the "gone" reading would have thrown
away: `gate3/progress` has the neutrality arms advancing across reboots, with
`resume_after_boot relaunching (uptime 150s)` doing exactly its job at 19:35Z. `neut1024-on1`
recorded `rc=124`, the 500 s arm timeout, and `neut1024-off2` started at 19:43Z.

The practical verdict for this gate is unchanged, though the reasoning is not: a box that reboots
inside a ten-minute window cannot finish a capacity roster or a 44-leg parity arm, and only the
arms short enough to fit one boot will ever record an rc.

## Where the gate can run

- **pc** is out for the accuracy arms. Its single p150a is the card root-caused in
  `pc-card0-512aa-fold-nondeterminism`: it miscomputes some matmuls at a low, location-keyed rate at
  every size, so a bit-exact or hash-equality result measured there means nothing. It also runs a
  custom 130-core firmware, so its timings are not the p150a the baselines were recorded on.
- **qb1** is the better host for the long arms. Four Blackhole p150a cards, idle (no worker processes, no card holders),
  `tt-bio-dev/env` imports ttnn and torch 2.11.0, 204 GB free. `docs/size_ladder_baseline.d/` carries
  p150a cells for 8 of the 9 ladder models, the exception being `protenix-v1`, which is p300c-only and
  will read as a coverage gap on qb1 for reasons that predate this lever.

Two hardcodings had to come out to get there, both now fixed and pushed: `gate_drive.sh` passed a
literal `tt-quietbox2:$CARD` to `capacity_gate.py` and `full_parity_gate.py` while every other path
was already parameterized, and `resume_after_boot.sh` held gate3/card0/qb2 as constants, which would
have had cron resume a different run on a different card than the driver this pass started, against
a progress file the done-guard never reads.

## The defect: not the pin, the reaper

The first version of this report said the card pin had failed, because the driver was launched with
`GATE_CARD=3` and the fold that came up under it held `/dev/tenstorrent/0`. That conclusion was
wrong, and the way it was wrong is the finding.

`TT_VISIBLE_DEVICES` takes a **UMD logical id**, and UMD numbers chips by sorting the PCI devices on
BDF. The kernel driver numbers `/dev/tenstorrent/N` in probe order. On qb1 the two disagree, and the
disagreement is a rotation:

    UMD 0 = 0000:01 = node 1        UMD 2 = 0000:42 = node 3
    UMD 1 = 0000:41 = node 2        UMD 3 = 0000:c1 = node 0

Measured, not inferred: each `TT_VISIBLE_DEVICES=K` in turn was given a bare `ttnn.open_device`, and
`lsof` read back which node it took. All four opened exactly one node, in that rotation. So the pin
held perfectly — a grant of UMD card 3 opened UMD card 3 — and `TT_BIO_LEASE_CARDS=3` was right not
to refuse it. Both of the things the first write-up called broken were working.

What is broken is the driver's own bookkeeping. `reap_card()` and `parity_card_holders()` both index
the node path with the UMD id:

    lsof -t "/dev/tenstorrent/$CARD"

On qb2 the two numberings happened to coincide, so this read correctly for three passes. On qb1 it
reads a card that belongs to someone else, and `reap_card` does not merely read: it sends SIGTERM and
then SIGKILL to every pid it finds. A gate granted UMD 3 would have reaped node 3, which is UMD 2 —
killing a co-tenant's job after every single arm, and logging it as its own leaked holder. The gate
never got far enough to do it, because `neut768-off1` failed first.

`gate_drive.sh` now resolves the UMD id to a node by sorting the sysfs `PCI_SLOT_NAME` entries the
same way UMD does, logs the resolution as its first progress line, and refuses to start if it cannot
resolve. The helper reproduces the measured mapping on qb1 exactly.

**The general shape, which is what makes it worth recording: a card grant and a device node are
different namespaces, and every safety check that crosses between them has to convert.** Comparing
the two integers is not a check, it is a coincidence that held on one box. Anything that reaps,
kills, or asserts "this card is free" by `lsof`-ing a node path indexed with a grant number is
wrong wherever the numberings differ, and nothing warns you.

`neut768-off1`'s own failure was separate and transient: `rc=1` in 14 s on `tt_tlb_alloc failed with
error code -12` for a 2 MB TLB window, ENOMEM against a card whose TLBs were momentarily held. The
next arm opened the same card cleanly 14 seconds later.

## What the next pass does, in order

1. Done: the mapping is established and `card_node()` converts, so the reaper can no longer reach
   another card.
2. Relaunch the driver on qb1, clear `PAUSE`, install the `@reboot` + `*/10` cron.
3. Leave gate3 running on qb2 exactly as it is. It advances across reboots on its own and costs
   nothing to let continue.

## Reused, not rebuilt

`perf/ttx_a3/gate_drive.sh` and `resume_after_boot.sh` (the resumable driver and its relaunch, both
now host-parameterized rather than replaced), `perf/ttx_a3/fold_parity_a3.py` for the neutrality
folds, and the banked 1536 aa ratio and below-cap digests, which this pass did not re-measure.
