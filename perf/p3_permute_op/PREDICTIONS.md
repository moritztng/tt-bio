# P3 / p3-permute-op — predictions, written before the device was opened

Phase 3, protenix-v2 trunk, 298 aa, qb2 card 2 (board 005, chip 2), board mate chip 3 held idle.
Production wheel is ttnn 0.68.0; the only build carrying `reblock_permute` is
`/home/ttuser/tt-metal-fused`, `v0.74.0-dev20260620-7-g0cea99ed1fa`. Ratios only (charter 4.8).

## Deliverable 1 — the DRAM->DRAM permutes T5 priced at 147.4 ms/fold

D1-P0 (op class). T5 numbers them `tenstorrent.py:1570/:1715`; the ledger numbers the same rows
`1295/1448 TriAtt`, `permute (1,0,2)`, 48.8 MB read + 48.8 MB write, ~14 % of the write roof in both
documents. Predict they are the **TriangleAttention `ending` pair-tensor transpose**
`ttnn.permute(x, (1,0,2))` on a 3-D `[S,S,c]` tensor, and therefore a **different op class** from
P4 E0`s 4-D channel move `permute(0,2,3,1)` that staged 4.39x. WRONG if the sites are 4-D channel
moves, or if the two `(1,0,2)` sites are not the ones carrying the 147.4.

D1-P1 (already partly harvested). W6 landed `_transpose_memory_config` (commit `cbf070de`), which
sends this permute`s **destination** to L1 when 2.5x the padded volume fits the grid`s unreserved L1.
At 298 aa the padded pair tensor is 320x320x256 bf16 = 52.4 MB and 2.5x is 131 MB against roughly
110 cores x ~1.2 MB. Predict the check **passes and production already writes to L1 on this card**,
so the ledger`s 147.4 ms/fold is a pre-W6 figure and the standing prize is smaller than the brief`s
~114. WRONG if `_transpose_memory_config` returns DRAM at this shape on this card.

D1-P2 (the source half is still open). The staging lever P4 found was about the **source**, not the
destination: a DRAM source cost 513.50 us against 134.91 us staged. Predict
`clone(DRAM->L1)` + `permute(L1->L1)` beats the production `permute(DRAM->L1)` by **>= 1.3x**.
WRONG under 1.1x.

D1-P3 (the repriced prize). Predict the deliverable-1 recovery at 298 aa is **20-70 ms/fold**, not
~114, because W6 already took the destination half. WRONG outside 15-90.

D1-P4 (parity). `torch.equal` True on every staged arm. A permute is a pure index move.

## Deliverable 2 — the custom op

D2-P0 (route). Predict a usable tt-metal **source build already exists** on this host and carries the
op, so no multi-hour build is needed to *measure* it; but it sits at `v0.74.0-dev`, six minors ahead
of the 0.68.0 production wheel, so pointing tt-bio at it is a **dependency major bump** and
release-gated. Predict **no route to the production wheel completes in this pass** and the honest
deliverable is a costed plan plus a ratio, not a delivered ms/fold. WRONG if a 0.68.0-matching source
build with the op turns up on this host.

D2-P1 (`generic_op`). Predict `ttnn.generic_op` on 0.68.0 cannot express this move without shipping
the same three kernels anyway, so it is not a cheaper route. WRONG if 0.68.0 exposes a
`generic_op` that takes user kernel paths and runs them.

D2-P2 (fold A/B). Predict a whole protenix fold does **not** run unmodified on the v0.74 ttnn (API
drift over six minors), so the A/B available this pass is a **block-wall or op-wall standin** on the
fused build, not a whole-fold wall. WRONG if `tt_bio` imports and folds on the fused build unchanged.

D2-P3 (the shipped gate). Predict the DRAM-only gate is a buffer-type or `N` test in the op`s
host-side entry (`reblock_permute.cpp` or its device operation), and that flipping it to admit the
L1 path at N=320 is a source edit on the fused build, not a tt-bio-side argument. WRONG if the gate
turns out to live in tt-bio.

## What is NOT predicted, because it is settled

P4 closed the kernel-structure question with a counting argument plus a decisive measurement: 64 NOC
transactions per source tile is a floor over all kernel structures, and bf16 -> fp32 left the permute
at 123.61 us against 123.68 while a clone control rose 1.36x on the same doubled bytes. I am not
looking for a better kernel structure and I make no prediction about one.
