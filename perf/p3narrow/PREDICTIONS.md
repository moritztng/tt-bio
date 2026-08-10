## Predictions (before measuring)

Committed and pushed as `perf/p3narrow/PREDICTIONS.md` before the device was opened; that file is the
tamper-evident copy and this section is the same text.

**P1 — Deliverable 1, the bit-exact arm at the PWA z->bias site, in a live fold.** `out_block_h`=5
with `in0_block_w` left at 1, applied at `tenstorrent.py` `PairWeightedAveraging.__call__`'s per-head
`ttnn.linear(z, z_weight[:, i:i+1])`. In-fold, synced per call, I predict the baseline per-call time
lands within 15 % of P2's card-0 figure of 438.46 us (i.e. 373-504 us) and the tuned arm gives
**1.10-1.20x**, so the site wall over its own call count falls by **9-14 ms/fold**. P2 priced this
half at 11.8 ms/fold. **Wrong if** the in-fold per-call delta is under 3 % of the baseline per-call
figure, which is P2's own kill test transplanted into the fold, or if `torch.equal` is not True.

**P2 — Deliverable 1 at the template z projection.** Same knob at `protenix.py:306`
(`_KeyedWeights._lin`), which the template embedder calls at `[1,298,320,256] x [256,64]`. I predict
**1.18-1.30x** (P2 measured 1.24x) and a site wall falling by **3.0-4.5 ms/fold** at 40 calls/fold.
P2 priced it 3.7. **Wrong if** under 3 % of the baseline per-call figure, or not `torch.equal`.

**P3 — the fold wall will NOT resolve deliverable 1, and that is the honest finding.** 15.5 ms on a
~31.9 s fold is **0.049 % of the fold wall**, and this harness's fold-to-fold spread is ~1 % of the
fold wall (~300 ms). I predict the 3-fold median moves by **less than 200 ms in either direction**
and that the sign is not stable across arms. The resolvable production number is therefore the
**site wall measured in place inside the live fold** — every call at that site, synced on both sides,
summed over the fold — and the stage wall it sits in, not the fold wall. **Wrong if** the fold wall
moves by more than 200 ms in either direction: that would mean the change does something beyond the
two sites and I have not accounted for it.

**P4 — in0's buffer type at both sites is DRAM-interleaved, so P2's 1.93x/2.02x transfers.** The pair
tensor is 48.82 MB and the MSA `z` is the same class; neither fits an L1 operand. I predict the
live-fold op capture shows `buf=DRAM` for in0 at both sites, and therefore that `in0_block_w`=8 is on
the 1.50-1.60x side of P2's sign flip, not the 0.72x side. **Wrong if** either capture shows
`buf=L1`, in which case P2's own mechanism says `in0_block_w`=8 will **cost** ~28 % and deliverable 2
is dead as written.

**P5 — Deliverable 2 parity at the fold's own shape.** `in0_block_w`=8 is **not** bit-exact. I
predict `torch.equal` False at both sites, PCC above 0.99998 and RMSD within 2x of P2's 2.021e-02
(@PWA) and 2.120e-02 (@template), i.e. 1.0e-02 to 4.3e-02. **Wrong if** `torch.equal` returns True
(P2's parity verdict would then be wrong), or if PCC falls below 0.9999 (a worse parity class than
`_PAIR_PROJ_BW`=16, which already ships, so the precedent argument would not hold).

**P6 — Deliverable 3, candidate 1: an L1 output for a pair-track projection is REFUSED at the real
call site.** The projection writes `[1,298,320,256]` bf16 = **48.82 MB**, which is 455 kB/core across
110 banks on top of the program's circular buffers and everything the live block already holds.
trimul-rescore measured every L1-output configuration above ~24 MB refused on a **quiet** card with
nothing else resident. I predict the allocation throws (`Out of Memory` from the bank manager, or
"Statically allocated circular buffers ... exceed L1") inside a real Pairformer block, and that free
L1 at the projection call site measures **below 500 kB/core**. **Wrong if** it allocates and runs, in
which case I measure it and price the win.

**P7 — Deliverable 3, candidate 2: the three output projections cannot fuse, and the fusion that
does exist is qkv+g+bias.** Read out of the graph: the trimul output projection (x2 per layer) and
`gate_and_project`'s `x_out` each consume a different upstream tensor and run sequentially, so there
is nothing to fuse — I predict this is a structural refusal, not a ttnn one. What does share an
input is `TriangleAttention`: `qkv` (nt=24), `g` (nt=8) and `triangle_bias` (nt=1) are three matmuls
on the same layer-normed `x`, adjacent in the graph. Fused they are one nt=33 output. I predict the
fused matmul beats the three separate calls by **1.10-1.35x** on device time, and that slicing the
fused `[1,298,320,1056]` back into its three consumers costs **400-700 us** (201 MB moved at
350-400 GB/s), so **the net is negative**: fused+slice within 10 % of separate, or worse. **Wrong if**
fused+slice beats the three separate calls by more than 3 % of the separate figure.

**P8 — roofs on card 1 land within 5 % of P2's card-0 figures.** Read 376.8 GB/s, unary write
264.1 GB/s, matmul-writer write roof 213.7 GB/s at nt=128, square bf16 HiFi4 compute 141.65 TFLOP/s.
I am measuring all four on card 1 myself and inheriting none. **Wrong if** any differs by more than
10 % of the card-0 figure, which would say the two cards are not the same part for this purpose and
no qb1 figure can be quoted without its card.

**P9 — Deliverable 4, the inventory.** Read out of a live fold's op capture, I predict **4 to 10**
distinct `ttnn.linear(core_grid=...)` shape classes with a DRAM-interleaved in0 beyond the two named,
and that the total unpriced `in0_block_w` headroom across them is **30-120 ms/fold**. **Wrong if**
the capture finds fewer than 3 or more than 20 such classes.

**Priced in advance, so the ranking can be wrong too.** I expect to bank ~15 ms/fold bit-exact and
~60 ms/fold release-gated, to kill both of the rank-1 candidates with evidence, and for the
inventory to be the largest thing I actually deliver. If that ordering comes out inverted I will say
so, as P2 did.

**C5 interaction, order assumed.** `protenix-trunk--p3-msa-untilize` owns the hoist that takes
`protenix.py:306` from 40 calls/fold to 10 (the template embedder recomputes `_lin(zn, ...)` once per
template, and `zn` does not depend on the template index). **I assume my change lands first and the
hoist second.** In that order my @306 figures are the 40-call ones (3.7 ms/fold bit-exact, 9.6
release-gated) and they fall to about a quarter of that after the hoist (0.9 and 2.4). The pair
must not be added at face value: bit-exact combined is ~12.7 ms/fold, not 15.5, and the
release-gated pair is ~17 ms/fold from these two sites, not 24.4.

---
