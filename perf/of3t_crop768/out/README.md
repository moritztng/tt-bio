# of3t-crop768 rung artifacts

One JSON per rung, written incrementally, so a file with no `backward.ok` is a run still in
flight rather than a run that produced nothing. `env.drop_dead_values` names the arm from
commit `51be36ad8` on; before it the arm is the branch default, `DROP_DEAD_VALUES = True`.

Complete as of 2026-09-21T12:40Z:

| file | tokens | arm | board | forward peak | backward peak | live DRAM allocs | backward |
|---|---|---|---|---|---|---|---|
| `split_384.json` | 384 | on | qb1 p150a | 4,794,512,384 B | 12,686,603,264 B | 4509 | completed |
| `split_480.json` | 480 | off | qb1 p150a | 11,040,498,688 B | 27,815,601,152 B | 4114 | refused |

The 480 refusal is a fragmentation refusal, not a size one: the allocator asks for
221,184,000 B per bank against `free: 810099520 B, largest free block: 200752064 B`, so it
misses the block by 20,431,936 B per bank while 6,480,796,160 B is free on the device.

In flight, chained under setsid so they outlive the launch that started them: `split_768.json`
(768 off, card 0), then 768 on and 384 off behind it; `split_480_on.json` (card 1), then 448,
512, 576 and 640 with the lever on.
