# The 2.34x holds at 1.95x. Its error bar is one convention in the byte term, and that one is now closed.

`ws:b2z-redteam-arithmetic`, the adversary row of the RADICAL 2x campaign. No card. Every number
here is re-derived from committed artifacts: the op-cost curve on `wk/b2x-op-cost-curve`, the
byte captures on `wk/b2x-diffusion-layer-bytes`, the roof measurements in
`perf/bioir_roofline/`, and the tt-metal device-profiler trace of a real `PairformerLayer` on
`wk/b2z-grid-utilization`.

**ARCH caveat, stated once and it applies to every number below.** The profiler trace is
**pc card 0, p150a Blackhole, 13x10 = 130 cores, ttnn 0.72.0-dev (Tracy build), stock SDPA**
(`TT_BIO_TRIATT_PERSISTENT_MASK=0`). The campaign's cell is **qb2 p300c, 11x10 = 110 cores,
ttnn 0.68.0, fused SDPA**. The transferable quantity is the *fraction* of the block that no roof
explains; the seconds are that fraction projected onto the cell's 41.4152 ms x 280 calls.

## The verdict

| | model | deficit | unexplained |
|---|---|---|---|
| campaign, as published | 17.668 ms | **2.23x** on this part (2.34x on qb2) | 55.1 % of the block |
| corrected, every term | **20.224 ms** | **1.95x** | **48.6 % of the block** |

`DEFICIT-SECONDS: -0.75 s`. The campaign's 6.6 s is **5.6 s**. It is still the largest single
thing in the fold and it is still not explained by any roof.

## Term by term

Measured block on this part: **39.3568 ms over 274 dispatched device programs** (median of 3
repeats, spread 0.44 %).

| correction | effect on the model |
|---|---|
| op count 428 -> 274 | **-0.979 ms** (deficit grows) |
| bytes 6.651 -> 8.177 GB | **+3.429 ms** (deficit shrinks) |
| sum of the two terms -> per-op `max` of them | -0.914 ms |
| add the arithmetic term, at each op's own math fidelity | +1.020 ms |

### 1. `t_fixed = 6.36 us` is not a fit intercept, and the brief's hypothesis about it is refuted

The brief asked whether 6.36 us is the intercept of a least-squares fit dominated by large points.
It is not an intercept at all. `op_cost_curve_512_qb2c0.json` carries three least-squares fits and
their intercepts are **5.636 / 4.666 / 3.859 us**; the harness's own `block_model` block uses the
`le_8MB` fit and reports a **2.77x** deficit. The campaign instead reads 6.36 us straight off the
flat end of the curve (the four smallest points are 6.360 / 6.365 / 6.380 / 6.465 us across an 8x
byte range) and 445.0 GB/s off the asymptote. Both are measured, not fitted, and 6.36 us is the
**most generous** per-op floor available in that file. Re-fitting makes the deficit bigger, not
smaller.

What is true about `t_fixed` is different and worse: **it is one op class under the most
favourable dispatch condition that exists**, R identical `ttnn.add` programs replayed back to back
on the same buffers. The real block's per-op floor is not one number. Two `1x1x32x32` bfloat16
adds — one tile, 2 KB — cost **449 and 448 us** each, stable to 0.9 % over three repeats, 71x the
floor. Two `ReshapeViewDeviceOperation` on 0.5 MB cost **215 us** each. Together that is
1.33 ms/block, **0.37 s/fold**, in four programs that move almost nothing and compute nothing.
Nobody in the swarm is on them.

### 2. The op count counts Python calls. 154 of the 428 launch no program.

`attribution_512_qb2c1.json`'s own census of the block:

    ttnn.deallocate 154   ttnn.linear 109   ttnn.multiply_ 44   ttnn.layer_norm 41
    ttnn.allocate_tensor_on_device 20   ttnn.generic_op 16   ...

`ttnn.deallocate` enqueues no device work — the same file says so. **428 - 154 = 274, which is
exactly the number of dispatched programs the device profiler counts** in the same block on a
different part and a different tt-metal. The fixed term is 1.743 ms, not 2.722 ms.

### 3. The byte count is 23 % low, and the reason is documented in the instrument that produced it

`b2x-diffusion-layer-bytes` states its `real` rule is a "lower bound inside fused generic_op
kernels". It is. `ttnn.generic_op` is handed `[in0, in1, *outs]` (`tt_bio/mm_generic.py:359`), so
**every tensor after the first two in the profiler's input list is an output the kernel writes**.
The block's 14 generic programs write two and four output tensors each; the lower-bound rule
charges one.

Counting every operand at its padded size, with each output charged one write:
**8.177 GB**, against the campaign's 6.651 GB. The convention is not a judgement call — it is
readable in our own kernel's call site, and it is the single largest correction in this pass.

Cross-check that it is the right convention: under it **no op in the block exceeds the DRAM roof
by more than 27 %**, and under the lower-bound convention the whole-block count lands at 6.566 GB,
1.3 % from the campaign's 6.651 GB, which is what a copy of the same rule should do.

### 4. Six programs run *above* the roof the campaign says nothing in the fold reaches

At the source-verified byte convention, six of the fused trimul/triatt generic programs move DRAM
bytes at **495-565 GB/s**, 11-27 % above the 445 GB/s `BW_eff` every op in this campaign is priced
against, and 13-29 % above this part's own measured 434-438 GB/s streaming roof. They are
1-read-2-write and 1-read-4-write kernels; `BW_eff` was measured with a 2-read-1-write
`ttnn.add`. **A roof measured with one read/write mix is not the roof for another.** The campaign
sentence "nothing in this fold is bandwidth-bound" is false for its own largest op class.

### 5. The model has no arithmetic term, and 84 % of the block's programs run at the slowest fidelity

`t = n*t_fixed + bytes/BW` prices zero FLOPs. **230 of the 274 programs run `HiFi4`** — 84 % of
the programs, 28.838 ms, 73.3 % of the block's device time; with the HiFi2 programs added,
93.5 % of the block runs at a fidelity whose measured roof is at or below 121.78 TFLOP/s. The
campaign's 85.96 TFLOP/s compute roof is exactly the HiFi4 8192-cube figure in
`roofs_p300c_qb2_card2.json` (LoFi 150.34, HiFi2 121.78, HiFi4 85.96), so it is the *right* roof
for this block. `b2z-arch-deficit`'s "the campaign's compute roof is ~2x too low, 178.6 TFLOP/s at
ttnn defaults" does not apply here without saying which fidelity it measured: at HiFi4 this part
does not do 178.6.

Charging each op its own arithmetic at its own recorded fidelity adds **1.020 ms** to the model.
The 149 GFLOP inside the fused generic kernels that the profiler's shapes do not expose changes
nothing: those ops are byte-bound by a factor of 2 already, so their compute term never binds.

### 6. "Uniform 2-3x across all four sub-units" is not what the source says, and per class it is a 6x spread

`b2x_op_cost/FINDINGS.md` reads **1.90 / 1.64 / 3.21 / 1.88** — three byte-heavy sub-units tight at
1.6-1.9x and one outlier. Per dispatched-program class, against the corrected per-op floor:

| class | n | ms/block | deficit |
|---|---|---|---|
| GenericOp (fused trimul/triatt) | 14 | 11.672 | **1.11x** |
| Concat | 1 | 0.328 | 1.24x |
| Slice | 33 | 0.580 | 1.88x |
| BinaryNg | 54 | 5.494 | 1.98x |
| Matmul | 113 | 8.303 | 2.69x |
| LayerNorm | 41 | 3.783 | 2.97x |
| Transpose | 8 | 1.866 | 2.97x |
| SDPA (stock) | 2 | 6.531 | 4.08x |
| ReshapeView | 3 | 0.442 | 5.26x |

**The block's largest class is within 11 % of its floor.** The fused trimul/triatt kernels are
essentially done. A megakernel that fuses them further is fighting for 11 %, not for 2.34x. The
deficit lives in `Matmul` (113 programs, 2.69x), `LayerNorm` (2.97x) and the attention kernel, and
`b2z-custom-sdpa` has already named the mechanism for the last one: packer tile passes, a third
roof that is in neither term of this model and that scales with tile count rather than with bytes
or FLOPs.

### 7. The additive model's own error bar, since the brief asked for it

Two independent checks.

* **Against the op it was calibrated on.** `6.36 us + bytes/445 GB/s` applied back to the
  `ttnn.add` curve over-predicts by **up to 35 %** between 1 and 8 MB moved (13.43 vs 9.93 us at
  3.1 MB) and is exact above 25 MB. The model is a sum where the hardware does a max.
* **Against convention choice.** Holding the measurement fixed and varying only defensible
  conventions for the three terms at the source-verified byte count, the deficit spans
  **1.87x to 2.36x**. The published 2.23x sits inside that band, near the top. That is a ±12 %
  bar, not the 1.8x the brief set as the disqualifying threshold.

### 8. Serialisation, which the model assumes for free, is nearly free

Summed `OP TO OP LATENCY` across the 274 programs is **1.379 ms**, 3.5 % of the block. The
additive-over-ops assumption costs almost nothing, and the gaps themselves are 0.386 s/fold of
real dead time that no lever in the campaign is pointed at.

## What the swarm should change

1. **Stop pricing levers against 6.651 GB.** The block moves 8.177 GB. Every "% of the byte roof"
   in this campaign is 23 % pessimistic for the pairformer, which is why byte-deleting levers kept
   under-delivering: there were more bytes than the ledger said and they were already moving near
   the roof.
2. **The fused trimul/triatt generic ops are at 1.11x of their floor.** Re-point anything aimed at
   fusing them further.
3. **113 matmul programs at 2.69x and 41 layer norms at 2.97x are where the block's recoverable
   time is**, together 12.1 ms of the 39.4 ms block.
4. **Four programs cost 1.33 ms/block (0.37 s/fold) for no bytes and no FLOPs** — two single-tile
   adds at 449 us and two 0.5 MB reshapes at 215 us. Cheapest lever found in this pass.
5. **`b2z-arch-deficit` should state the math fidelity of its 178.6 TFLOP/s.** This block is
   HiFi4 and its roof is 85.96.
