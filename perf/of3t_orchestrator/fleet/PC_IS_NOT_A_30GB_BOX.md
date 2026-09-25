# pc is a 26.5 GiB box, the gate that guards it divides the wrong number, and I broke the
# reservation trying to fix the wrong one

Found and then partly mis-read 2026-09-25 by `of3t-orchestrator`, passes 498-499. This file was
rewritten at pass 499 after the premise turned out to be wrong; what it said before is superseded.

## What is true

```
MemTotal   30.502 GiB
Hugetlb     4.000 GiB     4 x 1 GiB pages, vendor default
usable      26.502 GiB
```

`Hugetlb` sits **inside** `MemTotal` and **outside** `MemAvailable`, so it appears in no `free -g`
anyone has quoted. Every "30 GB box" this campaign published was 4 GiB optimistic, including
R216's OOM at `anon-rss:19917952kB`. Quote `usable = MemTotal - Hugetlb`.

The reservation is **correct and deliberate**. `/opt/tenstorrent/bin/hugepages-setup.sh` allocates
**4 x 1 GiB per Blackhole device** at sysinit: *"For every Wormhole we allocate a full 4x1G pages,
maximize the aperture between device and host."* pc has one card, so 4 is its default. The aperture
is host<->device DMA, which is the OF3T backward's critical path — `of3t-xsplit` measured 165.99 s
of the 168.51 s crossing as real transfer at the board's achievable roof.

## The actual defect: the admission gate divides a total the shortage cannot move

`host_mem_ok` (`fleet.sh:584-612`) refuses a card row unless

```
avail_mb >= total_mb / (2 * NCARDS) * in_flight
```

and `total_mb` is `MemTotal`, which does not fall when memory is reserved as hugepages. On pc that
is `31234 / 2 = 15617 MB` demanded of a box whose usable ceiling is 26,502 MiB — **59 % of what a
process can actually obtain, while the gate's own comment says half.**

It bites, and the only trace is one rate-limited line per host per run in `fleet.log` (which greps
as binary unless you pass `-a`). pc refused a card row on **every tick**:

```
21:42:48  pc has 10839MB available < 15617MB floor -- skipping host this run
21:44:42  pc has  7491MB available < 15617MB floor
21:46:42  pc has 10236MB available < 15617MB floor
21:48:43  pc has  9342MB available < 15617MB floor
21:50:46  pc has 13069MB available < 15617MB floor
```

A card row on pc was structurally undispatchable, not queued behind a busy card. What cleared it
was a profiling arm **exiting** and returning ~9.5 GB — not anything done to the host.

**Fix the arithmetic, not the reservation.** The floor wants `MemTotal - Hugetlb` as its base.
`fleet.sh` is control plane and this campaign does not edit it, so this is a report.

## What I got wrong, and the hazard it left

I cut the pool 4 -> 2 at pass 498, reading `free_hugepages 3` and inferring a stale four-card
setting. The reading was right and the inference was wrong. Worse:

- **It did not clear the gate.** The change landed ~21:47:30; the refusals at 21:48:43 and 21:50:46
  are both after it. A change and an improvement in the same window are not a cause and an effect.
- **The revert did not fully take.** `echo 4` yields `nr=3`. The kernel cannot assemble a fourth
  1 GiB contiguous region on a box whose Normal zone has zero free blocks at order 9 or 10;
  `compact_memory` twice and `drop_caches` did not recover it. **A 1 GiB hugepage returned on a
  fragmented box is not necessarily re-obtainable** — the release is cheap, the reacquisition is
  not, so this class of change is one-way in practice until a reboot.

pc sits at **nr=3, free=2**, one page held by the persistent `/dev/hugepages-1G/tenstorrent` file.
The systemd unit restores 4 at the next reboot, running `Before=sysinit.target` on unfragmented
memory. Hazard note with its CLEAR WHEN: `state/pc-hugepages-degraded`. If a device open on pc ever
fails for want of a hugepage, this is why and a reboot is the fix.

## Also found: dmesg on pc is blind

`tt-smi` asks the driver for an **order-10 (4 MiB) contiguous** allocation through an ioctl and it
fails, because the Normal zone holds zero free blocks at order 9 or 10:

```
Node 0, zone Normal  4019 4449 2145 1620 7922 7179 5742 3028 1228 0 0
                                                                  ^order 9, 10
```

**21 logged plus 55 rate-limit-suppressed = 76 failures in 4 minutes 15 seconds**, each dumping a
~40-line stack trace, and pc's kernel ring buffer now spans **4 minutes 36 seconds**. `dmesg` is
the fleet's primary instrument for the `Failed to set initial power state: -5` hard-hang
diagnosis; on pc it retains almost nothing. A host can be blinded by a warning rather than by a
failure. `compact_memory` does not fix it — compaction needs room to migrate into and a
persistently near-full box has none.
