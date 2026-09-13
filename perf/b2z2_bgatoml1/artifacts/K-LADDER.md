# The K ladder, one line per measured point, Wormhole whglx 8x9

K is the atom-window count the gate sees: `ceil((target_atoms + 14*design_len)/448)*448 / 32`, so
it moves in steps of 14 and **no K exists between 280 and 294**. `live` and `budget` are the
gate's own numbers, read off its per-call trace. Budget is `0.5 * 1,395,424 * 72 = 50,235,264 B`,
so the gate's admit limit is K <= 360.

| K | fixture | design_len | live B | % of budget | census | outcome |
|---|---|---|---|---|---|---|
| 140 | bg400  |  80 | 19,496,960 | 38.8 % | `l1 3000` | clean, bit-exact, A/A exact |
| 182 | bg580  |  80 | 25,346,048 | 50.5 % | `l1 3000` | clean |
| 238 | bg768  |  80 | 33,144,832 | 66.0 % | `l1 3000` | clean, bit-exact vs base |
| 266 | bg768  | 150 | 37,044,224 | 73.7 % | `l1 3000` | clean |
| 280 | bg768  | 180 | 38,993,920 | 77.6 % | `l1 3000` | **clean — last K that works** |
| 294 | bg768  | 210 | 40,943,616 | 81.5 % | `l1 1`    | **CRASH** |
| 294 | bg1024 |  80 | 40,943,616 | 81.5 % | `l1 1`    | **CRASH**, identical addresses |
| 364 | bg1300 |  80 | 50,692,096 | 100.9 % | `dram 3000` | declines, completes in 511.8 s |

Both K = 294 crashes throw the same thing at the same addresses from different targets and
different program ids (791 and 999):

    Statically allocated circular buffers in program N clash with L1 buffers on core range
    [(x=0,y=0) - (x=7,y=7)]. L1 buffer allocated at 1026048 and static circular buffer region
    ends at 1041696

Per core: L1 top 1,499,136 B, L1 buffers hold the top 473,088 B, the op's static CBs need
1,041,696 B from the bottom, overflow 15,648 B = **1.04 % of L1**. The op runs on **8x8 = 64
cores**; the gate divides by **8x9 = 72**.
