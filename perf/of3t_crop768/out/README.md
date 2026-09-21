# of3t-crop768 rung artifacts

One JSON per rung, written incrementally, so a file with no `backward.ok` is a run still in
flight rather than a run that produced nothing. `env.drop_dead_values` names the arm from
commit `51be36ad8` on; before it the arm is the branch default, `DROP_DEAD_VALUES = True`.

Complete as of 2026-09-21T15:00Z, all on tt-quietbox (qb1), Blackhole p150a, card
34,225,520,128 B. Clock is in every artifact, median 1350 MHz sampled DURING the work.

| file | tokens | arm | forward peak | backward peak | live DRAM allocs | backward |
|---|---|---|---|---|---|---|
| `split_384_off.json` | 384 | off | 4,794,512,384 B | 17,924,142,080 B | 4870 | completed, 1025.95 s |
| `split_384.json` | 384 | on | 4,794,512,384 B | 12,686,603,264 B | 4509 | completed, 1220.49 s |
| `split_448_on.json` | 448 | on | 6,035,735,552 B | 17,300,685,824 B | 4832 | completed, 1591.22 s |
| `split_480.json` | 480 | off | 11,040,498,688 B | 27,815,601,152 B | 4114 | refused |
| `split_480_on.json` | 480 | on | 10,155,762,688 B | 19,731,309,568 B | 4107 | completed, 842.13 s |
| `split_512_on.json` | 512 | on | 7,487,050,752 B | 23,299,281,920 B | 5934 | completed, 4433.81 s |
| `split_768.json` | 768 | off | 15,197,644,800 B | 34,219,288,576 B | 6420 | refused |
| `split_768_on.json` | 768 | on | 15,197,644,800 B | 34,219,681,792 B | 7463 | refused |

768 refuses on both arms with the card full, 393,216 B apart, so the dead-value release does
not reach it. Its forward completes. 512 runs with the lever where the baseline arm refused,
and it is the largest crop measured to run. 480 without the lever is refused on fragmentation,
not capacity: 221,184,000 B per bank against `free: 810099520 B, largest free block:
200752064 B`, missing the block by 20,431,936 B per bank while 6,480,796,160 B is free.

The two units disagree and the count changes sign: the lever takes 29.22 %, 29.06 % and
0.00115 % off the backward high-water at 384, 480 and 768, and moves the allocation count by
-7.41 %, -0.17 % and **+16.2 %**. Only the byte figure explains which crops pass.

384 off was re-measured here on qb1 and returns the same 17,924,142,080 B over 4870 allocations
as the qb2 p300c rung, so the board class does not move this number.

## Gradient

| file | what |
|---|---|
| `grad_a_{on,off}.json` | of3t-equivalence's float64 `TriangleMultiplicationIncoming` A/B, both arms PASS, identical |
| `grad_probe_{on,off}.json` | pin counts on that module: 0 parents released on either arm |
| `lever_gradcheck_{on,off}.json` | `hallgrad/gradcheck.py` both arms PASS; `calls_with_reads` 0, so the lever is inert in it |

Both float64 harnesses agree across arms and neither exercises the parent-release path. Read
the probe JSON before quoting the A/B as a control on the lever.

## In flight

`576:on` on card 1 (pid 354733, started 14:50Z), with `640:on` chained behind it on pid 4165188,
detached under setsid and rooted in this worktree. The 384-512 fit, exponent 2.0800, predicts
576 at 85.6 % of the card and 640 at 106.6 %, frontier 620.5 tokens.
