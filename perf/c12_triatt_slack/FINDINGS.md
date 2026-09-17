# The 0.7343 s of `generic_op` slack no row owns, attributed

Four sites, 1.8126 s in situ against a 1.0782 s floor. No device, no card, no new measurement.
Reproduce with `python3 perf/c12_triatt_slack/slack.py`; output pinned in `SLACK.txt`.

Clock: every number is at a during-sampled 1350 MHz. The in-situ per-site times come from
`c12-profiled-fold`'s ops reports via `perf/c12_genop_rate/insitu_sites.json`; the roofs from
`c12-genericop-rate`'s own qb2 card-3 session (`clk.txt`: min = max = 1350 across 405,561 samples,
the chip read 800 before the force). The reblock width ablation is tt-quietbox2 card 3, the same
part.

## The table

| site | calls | ms/call | MB/call | floor ms | % of roof | brief said | mechanism |
|---|---|---|---|---|---|---|---|
| triatt_sdpa | 560 | 1.2622 | 270.53 | 0.6209 | 49.19 % | 49.19 % | packer passes over the score matrix bind, not traffic |
| triatt_in | 560 | 1.0733 | 352.39 | 0.8087 | 75.35 % | 75.37 % | 328.3 GB/s against a roof no matmul dataflow here has reached |
| triatt_out | 560 | 0.4056 | **134.25** | **0.3081** | **75.96 %** | 37.99 % | byte count was halved; same rate as triatt_in |
| reblock_back | 560 | 0.4941 | 134.22 | 0.3413 | 69.07 % | 69.10 % | width-independent transaction issue, 72-99 % of its slack |

In situ 1.8126 s / 2446.9 Mc. Floor as the brief states it 1.0782 s, slack 0.7343 s / 991.3 Mc.
Floor with the byte model fixed 1.1642 s, slack 0.6483 s / 875.2 Mc. With the packer term carried,
0.5731 s at the clock-scaled end of its band and 0.4251 s at the Wormhole-measured end.

## 1. `triatt_out`'s byte count is exactly half its traffic

Five of the six sites' charged bytes reconcile to a 2R+1W or 1R+1W count on the executed shape to
within 0.02 %. `triatt_out` is charged 0.5000x. The audit already found it and printed the number:
`repair_B` = 0.0188 TB, labelled "DROPPED by the published rule (L1 pre-allocated destination)".
`charges()` returns false from `moves()` for an op whose destination was reserved in L1, so that
call is charged nothing, not even its 67.1 MB DRAM read. `insitu_sites.py:47` takes `B` and drops
`repair_B`.

The split is exact, not approximate: charged W is 280.0 whole pair-tensor writes and `repair_B` is
280.0 whole dropped reads, over 560 calls. 264 Pairformer + 16 MSA layers = 280, so one of the two
tri-attention directions writes L1 and the other DRAM, in every layer of both types.

**That gives a natural A/B with no device.** The two out-proj programs differ by 67.1 MB of DRAM
write. A DRAM-bytes floor predicts a 0.1540 ms gap between them. The legs measure 0.0161 ms:
0.0145 ms in the Pairformer leg and 0.0178 ms in the MSA leg, agreeing independently, 10.5 % of
prediction. The write costs the same wherever it lands, so the floor that fits this site is moved
bytes, and moved bytes are 134.25 MB either way.

Corrected, `triatt_out` is at **75.96 %** of its roof, not 37.99 %, and **0.0860 s of the unowned
0.7343 s was never slack.**

## 2. The two `minimal_matmul` sites are one rate, and there is no roof to score it against

`triatt_out` streams 331.0 GB/s, `triatt_in` 328.3 GB/s: 0.8 % apart, at shapes 2.62x apart in
bytes, with 1 output buffer against 5. That pair refutes both candidate mechanisms at once, the
brief's own leading hypothesis for `triatt_out` ("a small per-call cost that does not shrink with
the work") and the multi-destination page cost the 5-way N split would suggest.

What is left is a flat 1.32x under `bw_add8192`. That arm is `ttnn.add` on a big DRAM tensor
(`roofs.py:119`): unicast, page-sized, no relay. `minimal_matmul`'s in0 is a daisy-chain relay.
`tt_bio/kernels/triatt/dm_in0_sender.cpp:275-296` is, once per k block per hop, a semaphore wait on
the downstream core, a whole-block unicast to it, a Blackhole-only `noc_async_writes_flushed()`,
and a remote semaphore set. Every cube arm in the roof session is arithmetic-bound (cube4096 at
113.68 TFLOP/s moves 83.3 GB/s; cube2048_dflt at 101.38 TFLOP/s moves 148.5 GB/s), so the session
measured **no streaming roof for a block-relayed matmul dataflow at all**. 1.32x against an
eltwise roof is a missing roof, not a measured deficit. Same defect class as the dense-cube roof
that self-refuted `c12-matmul-key-attribution`'s 78.24 TFLOP/s ceiling, on the byte side instead of
the FLOP side.

## 3. `triatt_sdpa`: the binding resource is the packer, and the floor does not carry it

`sdpa_standard` makes three full-size PACK passes over the score matrix per k chunk:
`compute_common.hpp:1899` the QK^T write, `:1990` the mask add, `:2016` the sub-exp. At 512 aa the
score matrix is 2048 x 16 x 16 = 524,288 tiles, 16x the op's own output, 4766.3 tiles per core over
110 cores.

The only packer rate this campaign owns is 71.3 ns/tile, measured on Wormhole, dose-responsive and
linear to 4 % (memory `packer-tile-passes-third-roof-neither-byte-nor-flop`). **No Blackhole packer
rate has been measured**, so it is carried as a band: 52.8 ns/tile scaling by the clock ratio,
71.3 ns unscaled.

| term | ms/call | % of the op |
|---|---|---|
| traffic floor | 0.6209 | 49.2 % |
| arithmetic floor | 0.6095 | 48.3 % |
| **packer passes** | **0.755-1.020** | **59.8-80.8 %** |

The packer term is the largest of the three at both ends of the band, and it is the one term
`max(traffic, arithmetic)` omits. So `triatt_sdpa`'s 0.3590 s is not slack against bandwidth: no
bandwidth lever can reach it. The route is deleting a score-matrix pass, and the campaign has
already measured the cheap version of that on this exact kernel: batching the mask add's per-tile
`acquire_dst()` gave **2.60 % of the op, bit-exact**. Deleting a pass outright needs fusion across
a rounding point.

## 4. `reblock_back`: width-independent transaction issue, and the route it would need

Zero FLOP, so no overlap and no packer story (one pass, 2.9 % of the op). The same-part width
ablation splits it: `back / 512 / 128 / DRAM->DRAM` reads 0.83470 ms at bf16 and 1.48126 ms at
fp32, **1.7746x**, spread 0.0006 (`perf/ttx_reblock_bfp8/byte_sensitivity_qb2c3.json`). Fitting
`t(w) = a + b·w` puts 77.5 % in the byte term and **22.5 % in a width-independent term**.

That arm runs 1.689x the in-fold time, so its *time* does not transfer. Its *fraction* brackets
from one side and the DRAM floor from the other: in fold the non-byte residual is at most 30.9 %.
So the transaction term is 22.5-30.9 % of the site, 0.0624-0.0855 s of fold, i.e. **72-99 % of
`reblock_back`'s entire 0.0863 s slack**.

The kernel states its own invariant at `reader_reblock_permute_back.cpp:30`: "the gather is 64
transactions per output tile whatever the kernel structure". That is 19,065 32-byte L1-to-L1 reads
per core per call, which prices the transaction at 5.84-8.01 ns, 7.9-10.8 cycles at 1350 MHz.

**The route it would need:** a wider gather packet, which requires the 32 source rows of an output
tile to be contiguous in the producer's layout. The producer is `ttnn.matmul` inside
`MatmulDeviceOperation`, which is why this site has no wheel route. The only owner of a
producer-layout change is `c12-reblock-delete`. Handed over, not priced here.

## Handed to other rows

1. **`c12-genericop-rate` / orchestrator.** `insitu_sites.py:47` must read
   `MB["triatt_out"] = (0.0376 + 0.0188)e6 / 560`. Every figure downstream of `headroom.json` that
   quotes 37.99 % or a 2.63x ratio for this site is wrong by 2.00x in bytes.
2. **`c12-reblock-delete`.** `reblock_gated` is charged 3 tensor-units a call (134.2 MB read,
   67.1 MB written), a 2:1 mix, but `headroom.py:33` roofs it against `bw_clone`, the 1R+1W arm.
   At its matching roof it is at **71.66 %**, not 79.39 %. That retires the campaign's
   existence-proof efficiency: the best any of the six reaches is now `triatt_out` at 75.96 %, with
   `triatt_in` at 75.35 %, and the ceiling the brief handed this row (0.4544 s / 613.5 Mc at
   79.39 %) does not exist.
3. **`c12-reblock-delete`.** `reblock_back`'s route, above.

## The one device arm that would change a decision

Measure the Blackhole packer rate on `triatt_sdpa` with the linear-pass control the Wormhole row
established: run the mask add TWICE (`add_block_inplace` with `pop_in1` false is CB-neutral by
construction) and take the difference, rather than removing a pass, which can starve a downstream
`cb_wait_front` and wedge the card. Assert the doubled arm's output differs from baseline before
timing. One chip, three arms, ~10 minutes. It collapses a 0.755-1.020 ms band that currently spans
whether this site has 0.28 s of slack or none.
