# generic_op, measured in situ and scored against roofs from the same session

The campaign's largest op, 3,920 calls over 6 sites, priced against a roof this row measured on the
part it runs on. Reproduce:

    python3 perf/c12_genop_rate/insitu_sites.py            # per-site in-fold seconds
    python3 perf/c12_genop_rate/cores.py                   # core count per op class
    TT_VISIBLE_DEVICES=3 ... python3 perf/c12_genop_rate/roofs.py   # the roof session
    python3 perf/c12_genop_rate/headroom.py                # the join
    python3 perf/c12_genop_rate/budget.py                  # what it does to 12.5 s and 10.0 s

`PREREG.md` holds the predictions, written and committed before `roofs.py` opened a card.

## Clock

AICLK forced to 1350 MHz through tt-kmd's ARC queue (message 0x33) and sampled at 1 kHz from a
separate process across the whole session: 405,561 samples after the force settled (55 ms), min =
max = 1350 MHz, zero read errors, no gap over 60 ms, 440.0 s of span covering all three roof runs.
**The chip read 800 MHz before the force** (`clk.txt`, `"before": 800`), which is the artifact the
clock discipline exists for: unforced, every number below would have been ~1.62x slow.

## Sites: the 14 programs in a layer, from the executed graph

A PairformerLayer issues exactly 14 `generic_op` programs and an MSALayer the same 14, so
264 x 14 + 16 x 14 = 3,920, the campaign's own count. They are identified by
`COMPUTE KERNEL SOURCE` in the ops report, not by a census key label:

| # | site | kernel | fidelity |
|---|---|---|---|
| 0, 4 | trimul in-projection | `minimal_matmul/device/kernels/compute.cpp` | HiFi4 |
| 1, 2, 5, 6 | trimul reblock, gated | `kernels/reblock_permute_gated/` | HiFi4 |
| 3, 7 | trimul reblock, back | `kernels/reblock_permute/` | HiFi2 |
| 8, 11 | tri-att in-projection | `minimal_matmul` | HiFi4 |
| 9, 12 | tri-att fused SDPA | `kernels/triatt_sdpa/compute/sdpa.cpp` | HiFi2 |
| 10, 13 | tri-att out-projection | `minimal_matmul` | HiFi4 |

## Rate, in situ

`DEVICE KERNEL DURATION` medians over the 13 PairformerLayer and 11 MSALayer reps that
`c12-profiled-fold` grabbed out of a live fold, weighted by the fold's own call counts.

| site | calls | ms/call | fold s | Mc | TFLOP/s | GB/s | cores |
|---|---|---|---|---|---|---|---|
| trimul in-proj | 560 | 1.4926 | 0.8392 | 1132.9 | 28.77 | 269.9 | 110/110 |
| reblock gated | 1120 | 0.6448 | 0.7225 | 975.4 | 0 | 312.3 | 110/110 |
| tri-att SDPA | 560 | 1.2622 | 0.7067 | 954.0 | 54.44 | 214.3 | 110/110 |
| tri-att in-proj | 560 | 1.0733 | 0.6012 | 811.6 | 34.01 | 328.4 | 110/110 |
| reblock back | 560 | 0.4941 | 0.2775 | 374.6 | 0 | 271.8 | 110/110 |
| tri-att out-proj | 560 | 0.4056 | 0.2272 | 306.7 | 21.18 | 165.5 | 110/110 |
| **total** | **3,920** | | **3.3743** | **4555.3** | **26.02** | **270.6** | |

Cross-check: `c12-profiled-fold` composes `GenericOpDeviceOperation` at 3.3830 s by a different
weighting of the same legs. 3.3743 against 3.3830 is 0.26 %.

The in-situ rate is **faster than every standalone arm ever measured on these kernels**:
`rate_ab_512_qb2c3.json` has `ship_in` at 1.2495 ms against 1.0733 in situ, `ship_sdpa` 1.4667
against 1.2622, `ship_out` 0.4759 against 0.4056, a consistent 1.16-1.18x. A standalone arm is a
lower bound on the fold's rate here, not an upper one, so "headroom" read off one is not headroom.

## Roofs, measured on qb2 card 3 in one interleaved session

| arm | min ms | rate | A/A floor |
|---|---|---|---|
| bw_add8192 (2R+1W, 402.7 MB) | 0.9241 | **435.7 GB/s** | 0.32 % |
| bw_clone (1R+1W, 201.3 MB) | 0.5119 | **393.3 GB/s** | 0.34 % |
| cube4096 HiFi4 | 1.2090 | **113.68 TFLOP/s** | 0.09 % |
| cube4096 HiFi2 | 1.2191 | 112.74 TFLOP/s | |
| cube4096 LoFi | 0.6512 | 211.06 TFLOP/s | |

Both quoted roofs are bracketed by two independent prior readings of the same arm: 435.7 GB/s
against the census's 442.877 GB/s on qb2 node 1 (0.984x) and pc's 415.8 GB/s (1.048x); 113.68
TFLOP/s against 108.543 in an earlier card-3 session (1.047x) and the 115.685 the brief quotes
(0.983x). **Machine balance is 260.9 FLOP/byte** at HiFi4 on the 2R+1W roof.

The 2048^3 known-answer control did NOT land in its predicted 0.150-0.160 ms band. It reads 0.1695
ms min / 0.1690 A/A, stable to 1.7 % across three sessions of 9, 9 and 40 reps, against the b2z
proof's 0.1537 and `c12-profiled-fold`'s 0.1520. It is not jitter and it is not a config miss
(0.2154 ms at HiFi4 + fp32_dest_acc is a different arm again): the proof ran on the
`/home/ttuser/tt-metal-b2z` **source build** with the card pair, this session runs the
`ttnn==0.68.0` **wheel** on one card, and the pair is unavailable because `c12-compose-fold` holds
card 2. So there is a **1.115x cross-build offset on the one arm measured on both builds**, and the
site times come from the source build while the roofs come from the wheel. Carried explicitly
below rather than papered over.

## Headroom: all six sites are bandwidth-bound

Traffic floor uses the roof whose read:write mix matches the site (a matmul is 2R+1W, a reblock is
1R+1W; the two roofs differ by 1.11x, so one number cannot serve both). Arithmetic floor uses the
cube roof at the fidelity the ops report says the site runs at.

| site | FLOP/byte | traffic ms | arith ms | floor ms | in situ ms | ratio | % of roof |
|---|---|---|---|---|---|---|---|
| trimul in-proj | 106.6 | 0.9246 | 0.3778 | 0.9246 | 1.4926 | 1.614 | 61.9 % |
| reblock gated | 0.0 | 0.5119 | 0 | 0.5119 | 0.6448 | 1.260 | **79.4 %** |
| tri-att SDPA | 254.0 | 0.6209 | 0.6095 | 0.6209 | 1.2622 | 2.033 | 49.2 % |
| tri-att in-proj | 103.6 | 0.8090 | 0.3211 | 0.8090 | 1.0733 | 1.327 | 75.4 % |
| reblock back | 0.0 | 0.3414 | 0 | 0.3414 | 0.4941 | 1.447 | 69.1 % |
| tri-att out-proj | 127.9 | 0.1541 | 0.0756 | 0.1541 | 0.4056 | 2.632 | **38.0 %** |

**Sites where arithmetic binds: none.** Every one sits below the 260.9 FLOP/byte machine balance,
the two reblocks at zero. Floor 2.1693 s / 2928.6 Mc against 3.3743 s / 4555.3 Mc in situ, 1.555x.

That refutes the premise this row was opened on. The brief's "26.00-28.67 TFLOP/s against 115.685
TFLOP/s" is a comparison between an op at 0-254 FLOP/byte and a dense cube at 4096^3, and it cannot
be closed at any engineering effort: an op that moves 402.9 MB per call to do 42.95 GFLOP has no
route to a cube's rate. The tri-att SDPA is the one site near balance (traffic 0.6209 vs arithmetic
0.6095 ms, 1.019x), and that is exactly where the `max(traffic, arithmetic)` construction is least
informative, because a balanced op reaches neither roof.

## Prize

| target | floor | prize | Mc |
|---|---|---|---|
| every site at its own measured roof | 2.1693 s | 1.2050 s | 1626.7 |
| 90 % of roof | 2.4104 s | 0.9639 s | 1301.3 |
| 79.4 %, the best any of the six reaches today | 2.7325 s | 0.6418 s | 866.5 |

The third row is the only one with an existence proof on this part: `reblock_permute_gated` does
reach 79.4 % of its shape-matched roof, so bringing the other five to the same efficiency is at
least a describable target. The first row is arithmetic, not an engineering target.

## What that does to 12.5 s and 10.0 s

`budget.py`, both accountings, so the conclusion does not depend on this row re-pricing anyone
else's. `fused-eltwise` (0.1321 s) is excluded from both because its accuracy failed.

| generic_op ceiling | rest of book | total | fold |
|---|---|---|---|
| 1.2050 s | 0.6137 s as the campaign states it | 1.8187 s / 2455.2 Mc | 13.062 s |
| 0.6418 s | 0.6137 s | 1.2555 s / 1695.0 Mc | 13.625 s |
| 1.2050 s | 0.3866 s re-based by `c12-profiled-fold` | 1.5916 s / 2148.6 Mc | 13.289 s |
| 0.6418 s | 0.3866 s | 1.0284 s / 1388.4 Mc | 13.853 s |

12.5 s needs 2.3810 s. The loosest of the four is short by **0.5623 s / 759.2 Mc**, the tightest by
1.3526 s / 1826.0 Mc. Apply the 1.115x cross-build offset to the streaming roof in the direction
that helps and the f = 1.00 ceiling becomes 1.4289 s, which with the campaign's own 0.6137 s book is
2.0426 s, still 0.3384 s short. 10.0 s needs 4.8810 s, more than double the largest ceiling here.

## Mechanisms, and why five of six are closed

1. **Core count.** All 182 + 154 `generic_op` programs in both legs run **110 of 110** cores. There
   is no occupancy to recover, and the core-count sweep already found the whole grid fastest at all
   four shapes. This is also the measurement that separates this row from the two refuted grid
   levers, which measured 103.8/110 coverage on the `ttnn` op ladder (prize 0.000 s) and `core_grid`
   on `ttnn` hot-path sites (0.156 s in fold): neither touched a `generic_op` program.
2. **The 64-of-110 pin handed over by `c12-matmul-key-attribution` is real, and it is not in this
   op.** `cores.py` finds the 64-core cluster in `MatmulDeviceOperation`: 58 programs carrying
   23.662 ms across the leg's 13 PairformerLayer calls, and 110 programs carrying 22.795 ms across
   the MSALayer leg's 11. That is ~1.820 and ~2.072 ms per call, so ~0.5137 s / 693.5 Mc of fold in
   the trunk (an estimate, because 58 does not divide by 13 -- the leg window clips matmul programs
   at its boundary where it clips no `generic_op` program). At the 110/64 = 1.72x occupancy the lead
   assumes, that prices at 0.2150 s / 290 Mc, under this row's kill criterion, and it belongs to the
   `matmul` class. `_triangle_mul_program_config` is a `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig`
   (`tenstorrent.py:3967`); no `generic_op` site takes one.
3. **DRAM bank/page walk.** Measured at 512 aa: gated stride 0.964x, rotate 1.030x; back stride
   1.019x, rotate 1.018x. No walk ships and the bank model picks the device's winner at 6 of 12.
4. **Folding the qkv projection into SDPA** (`TT_BIO_TRIATT_FUSE_QKV`): already fused at this shape.
   `rate_ab_512_qb2c3.json` records `fuse_rejects {qkv_already_fused_with_gate: 48}` and
   `sdpa_route_counts {fused: 60, stock: 0}`; the arm read 3.7753 ms against a 3.7870 ms A/A.
5. **Trimul operand L1 residency**: <= 0.1066 s / 143.8 Mc, and the 67.1 MB chunk does not fit L1 at
   512 aa (`TRIANGLE_MULT_L1_MAX_SEQ = 352`).

## The one mechanism left, priced and not built

The two reblocks are **1.0000 s / 1350 Mc of zero-FLOP fold time** whose input is a tensor another
`generic_op` program wrote microseconds earlier. Chasing their *rate* is worth at most 0.2356 s
(they are already at 79.4 % and 69.1 % of their shape-matched roof). **Deleting** them is worth up
to the full 1.0000 s, and it is the only mechanism in this op that goes below the traffic floor,
which is the floor that binds all six sites.

The shape of it: have `minimal_matmul`'s writer emit the trimul in-projection already reblocked,
with the gate applied in the epilogue, so the projection's DRAM write and the reblock's read of the
same tensor both disappear. Bounded above by 1.0000 s / 1350 Mc, because a fused producer cannot
beat not running the reblock at all. Not bounded below by anything yet: it needs a kernel that does
not exist, in a tt-metal source build, plus an accuracy pass on the gate epilogue. That is a build
row, not a knob, and this row does not open it on a prediction.

Two instrument defects found on the way, both handed to the parent:
- The site times and the roofs come from different tt-metal builds (1.115x on the one shared arm).
  Closing that needs board 410D as a pair, which no C12 row can get while every other row is pinned
  to card 2 -- the same structural blocker `c12-profiled-fold` reported.
- `perf/roof_shape/shape_roofs.py:177` calls `bw_add8192` "the" DRAM roof. It is the 2R+1W roof;
  the 1R+1W roof is 1.108x lower on this part (393.3 vs 435.7 GB/s), and four of this op's six
  sites would be mis-scored by up to 11 % if scored against the wrong one.
