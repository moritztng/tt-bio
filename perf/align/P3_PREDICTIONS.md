# p3-align-widen — the free fill, and the one guard between us and 342 ms/fold

Phase 3, OPTIMISE. Protenix-v2, trunk only, 298 aa (token axis 298 logical, 320 padded; c_z=256).
qb2 card 0, board 007 chip 0, ttnn 0.68.0 wheel. **Ratios only, never campaign absolutes**
(charter §4.8). Parent: `protenix-trunk--p2-alignment`, which measured the blast radius (398.9
ms/fold), the mechanism (NCRISC `reader_bmm_tile_layout_in0_sender_padding`), and the two blockers.

## Predictions (before measuring)

Committed and pushed to `wk/protenix-trunk--p3-align-widen` before the device was opened.

The one thing my parent did not look for: **ttnn 0.68 ships `ttnn.fill_implicit_tile_padding`**
(`dir(ttnn)` on the production wheel). P2 priced the fill at 22.51 us/call from a full-tensor
`multiply_`; an op that writes only the padding lanes should be an order cheaper, and if it is, the
net moves off 153.3 ms/fold towards the 342.0 gross. Every prediction below is written against that.

**P1 — `fill_implicit_tile_padding(x, 0.0)` costs 2-10 us/call** on the contraction's own operand,
`[1, 32, 298, 298]` bf16 L1, against P2's 22.51 us/call `multiply_` floor. Reason: the last-dim
padding lanes are 22 of 320 columns in the last tile-column only, ~2.2 % of the 6.554 MB the mask
multiply rewrites. **Wrong if** it exceeds 15 us/call, or if it refuses a 4-D bf16 L1 tensor.

**P2 — the fill has to land after the permute, not at the mask site.** At `tenstorrent.py:1347`
`a_chunk` is `[1, L, L, C]`, so only dim2 carries tile padding and dim1 carries none; the contracted
axis is created by `_transform_chunk`'s permute. I predict the mask route zeroes the wrong axis and a
mask-on arm is **not** sufficient on its own, while a fill placed immediately before the contraction
is. **Wrong if** the mask-on arm is bit-exact against the aligned arm at the fold's own shape.

**P3 — one fill does not serve 16 contraction calls per block.** Each of the `n_pairs` chunk
iterations builds a fresh `a_chunk` out of a fresh `minimal_matmul`, so the fill is per contraction:
8 fills per trimul, 16 per block, 8384 per fold. The one placement that would amortise is zeroing
`x_norm_in` once per trimul, and it cannot work, because `layer_norm` writes `beta` into the padded
rows on every block (P2 E7). **Wrong if** the padding of a permuted chunk is already zero, i.e. if
`ttnn.permute` zero-fills what it does not write.

**P4 — the net.** At P1's midpoint (6 us/call) the contraction row nets `(40.79 - 6) x 8384` =
**291.7 ms/fold**, against P2's 153.3 at the `multiply_` price. Stated band: **250-320 ms/fold** for
the contraction row alone. **Wrong if** outside that band.

**P5 — the widen is a metadata change once the guard is relaxed.** `[1,32,298,298] ->
[1,32,320,320]` on a TILE-layout tensor whose padded shape is already `[1,32,320,320]`: same buffer
address, same page count, no copy, `torch.equal` True on the 298x298 valid region. **Wrong if** the
buffer address moves or the valid region changes by one element.

**P6 — the guard is not the only obstacle in the C++.** `reshape_common.cpp:50` is the first
`TT_FATAL` the call hits, but the reshape op then has to decide the case is a view rather than a
data movement. I predict a working widen needs the volume guard relaxed **and** the view-eligibility
path to accept a logical shape that grows into existing padding. **Wrong if** relaxing line 50 alone
produces a correct, copy-free widen.

**P7 — the alignment does not survive the block.** P2 traced it out of the source; I predict a live
block trace agrees, and the alignment dies at `_trimul_out_proj`, which writes back into `z` at
logical 298. **Wrong if** any pair-shaped tensor reaching the next block's `layer_norm` is logically
320.

**P8 — a 0.68.0 source build is not reachable in this pass and is not the right route anyway.** The
three built tt-metal trees on this host are v0.73.0-dev, v0.74.0-dev and one unbuilt; a cold build at
the 0.68 tag is hours. I predict the guard relaxation is provable on a v0.73-dev incremental build,
that the fold is **not** runnable against it (that build's python bindings are 3.12, the tt-bio env
is 3.10), and therefore that **no fold-level number through the widen exists at the end of this
pass.** I will say so rather than project one.

Instruments: `perf/align/a_probe*.py` (inherited from P2 unmodified) plus `perf/align/p3_probe.py`
(new). Wall timing with `ttnn.synchronize_device` on both sides of every region (charter §4.4).
`TT_METAL_WATCHER` is not used (T4 rejected it).
