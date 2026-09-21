# of3t-crop768 rung artifacts

One JSON per rung, written incrementally, so a file with no `backward.ok` is a run still in
flight rather than a run that produced nothing. `env.drop_dead_values` names the arm from
commit `51be36ad8` on; before it the arm is the branch default, `DROP_DEAD_VALUES = True`.

Complete as of 2026-09-21T12:50Z, all on tt-quietbox (qb1), Blackhole p150a:

| file | tokens | arm | forward peak | backward peak | live DRAM allocs | backward |
|---|---|---|---|---|---|---|
| `split_384.json` | 384 | on | 4,794,512,384 B | 12,686,603,264 B | 4509 | completed, 1220.49 s |
| `split_480.json` | 480 | off | 11,040,498,688 B | 27,815,601,152 B | 4114 | refused |
| `split_480_on.json` | 480 | on | 10,155,762,688 B | 19,731,309,568 B | 4107 | completed, 842.13 s |

480 runs with the lever and is refused without it. The refusal is a fragmentation refusal, not
a size one: the allocator asks for 221,184,000 B per bank against `free: 810099520 B, largest
free block: 200752064 B`, so it misses the block by 20,431,936 B per bank while 6,480,796,160 B
is free on the device.

The lever takes 29.06 % off the 480 backward high-water and 0.17 % off the allocation count, so
the two units disagree by a factor of 170 and only the byte figure explains the pass.

In flight, chained under setsid so they outlive the launch that started them: `split_768.json`
(768 off, card 0) with 768 on and 384 off behind it; 448, 512, 576 and 640 with the lever on,
on card 1.
