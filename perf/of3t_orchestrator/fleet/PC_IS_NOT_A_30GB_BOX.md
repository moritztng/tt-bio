# pc is a 26.5 GiB box, and the campaign has been doing its memory arithmetic against 30.5

Found 2026-09-25 by `of3t-orchestrator` (pass 498) while checking why pc keeps running out of
memory under the exactness-ON training step. Everything below is read from `/proc/meminfo` and
`/sys/kernel/mm/hugepages/`, no device opened.

## The reservation

```
MemTotal   30.502 GiB
Hugetlb     4.000 GiB     4 x 1 GiB pages in the 1 GB pool
usable      26.502 GiB
```

`Hugetlb` sits **inside** `MemTotal` and **outside** `MemAvailable`. A run that reasons "30 GB box,
19 GiB peak, should fit" is reasoning against a ceiling 4 GiB larger than the one it will hit. That
includes R216's OOM at `anon-rss:19917952kB` — reported against "a 30 GB box with no swap", and
actually a 26.5 GiB one.

## Three of the four pages were never used

With a device open on card 0 and a taped training arm running:

```
nr_hugepages   4
free_hugepages 3
/dev/hugepages-1G/  one file: tenstorrent, 1073741824 B, created 2026-09-23 11:46
```

**One page per card, and pc has one card.** The pool is sized for a four-card QuietBox. That is
direct evidence, not inference: three of four pages read free *while* the device was open.

## What was changed, and how to undo it

```sh
echo 2 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
```

`4 -> 2`: **2.0 GiB returned to the normal allocator**, one spare page kept over the one observed
in use, so a close/open cycle that briefly wants two still gets them. Verified after: `nr=2
free=1`, the in-use `tenstorrent` page untouched, `Hugetlb 2.000 GiB`, usable **28.502 GiB**, and
the live arm on card 0 came through it healthy (same pid, running, RSS still climbing normally).

Reverse it with `echo 4 | sudo tee ...` — it is one command and takes effect immediately. Shrinking
only ever reclaims FREE pages, so this cannot take a page out from under an open device; the risk
it does carry is the other way round, a *future* open that wants more pages than the pool holds.
If a device open on pc ever fails for want of a hugepage, raise the number first and ask second.

## A second, unrelated defect found on the way: dmesg on pc is blind

`tt-smi` asks the TT driver for an **order-10 (4 MiB) contiguous** allocation through an ioctl and
it fails, because pc's Normal zone holds **zero free blocks at order 9 or order 10**:

```
Node 0, zone Normal  4019 4449 2145 1620 7922 7179 5742 3028 1228 0 0
                                                                  ^order 9, 10
```

Each failure dumps a ~40-line stack trace. **21 logged plus 55 rate-limit-suppressed = 76 failures
in 4 minutes 15 seconds**, and the kernel ring buffer on pc now spans **4 minutes 36 seconds
total**. That matters beyond this campaign: `dmesg` is the fleet's primary instrument for the
`Failed to set initial power state: -5` hard-hang diagnosis, and on pc it now retains almost no
history at all.

`echo 1 > /proc/sys/vm/compact_memory` does **not** fix it — run and re-read, orders 9 and 10 stay
at 0. Compaction needs free space to migrate into and the box does not have it. So this is the
campaign's own blocker wearing a different hat: pc runs persistently near-full, and the first thing
that breaks is not a big allocation but a contiguous one.

These failures **predate** the hugepage change (first at 21:45:08, the change at ~21:47:30). Stated
because the timestamps sit a minute apart in the same log and the wrong causal reading is the easy
one.
