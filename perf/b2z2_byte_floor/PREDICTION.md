# b2z2-trunk-byte-floor — PREDICTION, pre-registered

Written 2026-09-12 before any measurement of my own. What exists at this point: the committed
census artifacts I am about to re-read (`b2z2-bh-tile-census`'s `tile_census.json`, carried into
`perf/b2z2_redteam2/src/`), the code of `PairformerLayer`/`TriangleAttention`/`TriangleMultiplication`
in `tt_bio/tenstorrent.py`, and the three source state docs the brief names. No device opened, no
counter run, no arm measured.

## The quantity

The block moves **8.0493 GB** per call (4,710.8 MB read + 3,338.5 MB written, BH, 512 aa, per
program, each operand tensor counted once). The brief asks how much of that is a byte read twice
where once would do, and what deleting it is worth in the second currency.

## P1 — removable bytes, as a fraction of 8.0493 GB

**Point estimate 6.5 %. Band 3 % - 12 %.**

Decomposed, from the code walk and before I count anything:

* **R1, the triangle attention's normed pair tensor is read three times.** `TriangleAttention`
  norms `x` into `x_norm` (67.1 MB at 512 aa, bf16) and then hands the SAME tensor to three
  separate projections: `_pair_proj_linear(x_norm, bias_weight)` 128->32, `qkv_heads(x_norm,
  qkv_weight)` 128->384, `gate_proj(x_norm, g_weight)` 128->128. Three full reads of one tensor,
  two of them redundant if the projections share a read. Two triangle attentions per block:
  **2 x 2 x 67.1 = 268.4 MB = 3.33 %.** This is the same defect shape as `proj_z` in
  `wk/b2z2-msa-layer-census@dfb8b9e08`, one level up.
* **R2, the layernorm round trip.** 7 layernorms per block write 402.7 MB that a projection
  immediately reads back. A norm fused into its consumer's reader deletes the write and one read.
  For the two triangle attentions the norm has three consumers so it cannot simply vanish; for the
  trimul input norm and the transition norm it has one. Call it **1-3 %** reachable, and the rest
  a fusion problem rather than a redundancy.
* **R3, the writes.** 3,338.5 MB, 41.5 % of the traffic, unexamined by any row. My prior is that
  most of it is genuine (five residual adds and two trimul products are real results), and that the
  removable part is the intermediates that go to DRAM and come straight back with no other
  consumer: the `transpose_wh` -> `reblock_permute` pair on the trimul tail ([1,128,512,512], 134.2
  MB written and 134.2 MB read, once per trimul) is the clearest instance. **~2-4 %.**
* **Not counted as removable:** `reblock_permute_gated`'s 1,073.7 MB. 536.8 MB of that is a LEDGER
  error already found by `b2z2-byte-axis-reopened` (the reader touches two of four channel slices),
  not bytes on the silicon. Correcting it makes the true block total **7.51 GB**, and I will report
  both denominators rather than banking a ledger fix as a win.

## P2 — block ratio

**1.02x, band 1.00x - 1.05x, on the paired interleaved block A/B with its own A/A floor.**

The byte model says N % of bytes should buy roughly N % of the 45.2 % of the block that is
byte-proportional (`b2z2-byte-axis-reopened`'s corrected figure), so 6.5 % of bytes is
6.5 % x 45.2 % = 2.9 % of the block = 1.030x at realization 1.0. The measured realization of a
byte change on this block is **0.53 - 0.58** (`trimul_out` 0.576, `trimul_in`/`triatt_gate` 0.53 in
the loss direction), which puts the honest point estimate at **1.017x** and is where the 1.02x
comes from. The band's top end assumes realization 1.0 and the full 12 %.

## P3 — the fused-kernel census must not move

Every arm reports `qkv_heads`, `trimul_tail`, `_pair_proj_minimal_matmul` eligibility counts. If a
lever deletes bytes and an eligibility count falls, the arm is disqualified as a byte measurement,
because that is exactly how `b2z2-byte-axis-reopened`'s sign got flipped: 20.13 % of the bytes went
away and the block got 2.2 % slower, because the tuned kernels stopped serving.

## The falsifier

**If deleting N % of the bytes bit-exactly, with every fused-kernel eligibility count unchanged,
does not move the block by roughly N % x 45.2 %, then the byte model that replaced the tile model
is also wrong.** That is the finding, not the failure. Concretely: R1 removes 3.33 % of the bytes;
the byte model predicts 1.015x on the block at realization 1.0 and 1.009x at realization 0.576.
A measured block ratio inside 1.000x +/- the A/A floor, with the eligibility census flat, refutes
the byte model at this site. A ratio at or above 1.015x on 3.33 % of bytes refutes the 0.576
realization and says the byte model UNDER-prices.

## What would change my mind before I start

A buffer-level count showing the three triangle-attention projections do not in fact read the same
buffer (for instance if `gate_proj` consumes a copy), or showing a redundancy an order larger than
R1 that the shape-keyed census cannot see.
