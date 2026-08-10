# z-h5-infold — predictions, committed before the device was opened

Written and committed on `wk/protenix-trunk--z-h5-infold` before `h5_infold.py` or `h5_kchunk.py`
opened a device. The predecessor committed P1-P7 the same way and three of the seven were wrong,
which is the point: a post-hoc explanation cannot be wrong.

Card: qb2 chip 0, ttnn 0.68.0, 11x10 grid. Every number below is a ratio and owes a qb1/0.67.4
re-take before it drives a decision (charter §4.8).

---

## Part A — the 13x

The disagreement, restated from the two docs so the falsifiers are unambiguous:

- isolated OFF-minus-ON **region** delta at `[1,512,512,64]` = **0.172 ms/region**
  (`z-progcfg-h5`, `h5_cells.py`: C1D 0.7461, C1L 0.5737),
- in-fold `body:TriangleMultiplication` OFF-minus-ON at 512 aa = **+367.4 ms**, divided by 160
  template-track regions = **2.30 ms/region** (`size512-ab`),
- ratio **13.4x**.

One arithmetic fact frames every prediction below and I state it before measuring anything:
**the isolated OFF arm's whole region costs 0.7461 ms.** A saving of 2.30 ms/region is **3.08x the
entire cost of the thing that is supposed to be doing the saving.** So at least one of these must
hold: the wall contains ops outside the region, the in-fold OFF region is >3x its isolated self, or
the denominator is not 160.

### HA1 — the wall contains the cascade, the region does not (PRIMARY)

The narrow wall (`p_out` + `g_out` + the region `multiply_`) reproduces the isolated
**0.172 ± 0.05 ms/region** in a live fold, and the excess lives between the narrow wall and the wide
wall — in the residual `add_` and in whatever the trimul body does around the region.

*Predicted numbers:* narrow OFF−ON = **0.15 to 0.25 ms/region**; body OFF−ON = **1.8 to 2.8**;
wide OFF−ON = body + **0.03 to 0.06** (the residual `add_` reading 33.55 MB from L1 at ~725 GB/s
instead of DRAM at ~389 GB/s saves 0.040 ms, and there is one `add_` per region).

*Falsified if* the narrow wall reads outside 0.10-0.35 ms/region, or if the body-minus-narrow
remainder is under 0.5 ms/region.

**Consequence if confirmed:** `size512-ab`'s +367.4 ms is real as a fold effect but is **not the
projection's destination term**, and the org must stop quoting 2.30 ms/region for this site. The
row-blocking candidate in Part B then has to be priced against whatever the remainder turns out to
be, not against 2.30.

### HA2 — the isolated probe is the thing that is wrong

The narrow wall itself reads ~2.3 ms/region in-fold. The isolated probe would then be understating
the same op by 13x, and the org has three other numbers taken on isolated probes.

*I predict this is FALSE*, and the reason is the arithmetic above: for HA2 the in-fold OFF-arm
region would have to cost **≥ 2.30 ms** against 0.7461 isolated, i.e. the same two matmuls and one
`multiply_` running **>3.1x slower** inside a fold than alone. There is no cross-op DRAM contention
to appeal to — this device runs one program at a time — and the census already says the two arms
run byte-identical program configs at `512:64`.

*Confirmed if* the narrow wall is above 1.5 ms/region. `z-permute-bands` found the sign-level
version of this at an L1 destination (isolated 0.99x against an in-fold 1.04x), so it is live and
it is the first thing the narrow wall settles.

### HA3 — the excess is inside `TriangleMultiplication` but outside the region (SECONDARY, and I
rate it second-most-likely after HA1)

Named before measuring, with its mechanism: **the L1 output changes what `ttnn.reallocate` and the
channel loop's allocator see on the NEXT trimul.** `_transform_chunk` ends every channel chunk with
`ttnn.reallocate(chunk, memory_config)` (`tenstorrent.py:1580`), whose cost is a function of live
L1 fragmentation, not of bytes. With the flag ON the trimul returns an L1-resident 33.55 MB
`z_update` that stays live until the caller's `deallocate`, so the *next* trimul's channel loop
runs against a differently-shaped L1 free list. That is a per-op cost with no byte model behind it
and it is invisible to any isolated probe, which starts from an empty L1 every time.

*Predicted numbers:* if HA3 carries the excess, the op-level decomposition (`--depth ops`) puts
**more than 60 % of the body's OFF−ON delta** on `reallocate` + `channel-loop matmul` +
`permute/transpose`, and **under 25 %** on the three region ops.

*Falsified if* the region ops carry more than half the body delta, or if `reallocate`'s own delta is
under 0.1 ms/region.

### HA4 — a denominator that is not 160

Stated so the census can kill it rather than leaving it implicit. `size512-ab` counted **480**
`pair_proj` L1-output calls at `c=64` in a live 512 aa fold. Six per block: four from
`_trimul_out_proj` (2 trimuls x 2 projections) and two from `TriangleAttention`'s `x_out`
(`tenstorrent.py:1953`). Only the first four are in the `TriangleMultiplication` body, so the trimul
denominator is **320 projections = 160 regions**, and 480 is the whole-fold flag-affected count.

*I predict the census confirms exactly this: 480 total, 320 of them inside
`TriangleMultiplication`, 160 inside `TriangleAttention`, all at padded shape `[1,512,512,64]` with
weight `[64,64]`, and all on the L1 branch.* At `c_z=256` I predict **3144** calls, all on the DRAM
branch, and `_L1_OUT_REFUSED` **empty** — because `_pair_proj_config(out_l1=True)` returns `None`
from the static budget before any allocation is attempted, so there is no exception to memoise.

*Falsified if* any of those five counts differs, or if `_L1_OUT_REFUSED` is non-empty.

### What I predict the fold-level arithmetic will not support

480 flag-affected projections x the isolated per-call destination delta at `512:64`
(0.2502 − 0.1680 = **0.0822 ms**) = **39.5 ms**, plus ~6 ms of residual `add_` and ~13 ms of
`multiply_` operand reads = **~59 ms/fold** from the byte model, against `size512-ab`'s measured
**+508.0 ms** fold wall and **+476.96 ms** block wall. **8.6x.** So even before the trimul body is
split, the byte model does not explain the 512 aa fold delta, and HA1 is a claim about *where* the
excess is, not a claim that it is small.

---

## Part B — the k-chunked out-projection region at `[1,512,512,256]`

The hand-off to be tested: chunk the region into k row blocks so the live L1 set drops from
268.4 MB to 268.4/k, recovering a destination term the calibrated byte model puts at
**~0.72 ms/region**, i.e. **~760 ms/fold** over 1048 regions.

### K1 — k=2 does not allocate; **k=4 is the first k that does**

At k=2 the two live outputs are 67.1 MB each, **134.2 MB together** — the same total as one
unchunked output, and P7 already showed a single 134.2 MB L1 output OOMs under production's
`bw=8, obh=5, obw=8` because the circular buffers take the rest of the bank. Chunking halves the
per-block buffer but not the number of live blocks, so k=2 buys nothing. k=4 puts two live blocks at
67.1 MB total, 42 % of the 160.8 MB of L1.

*Falsified if* k=2 allocates, or if k=4 does not.

### K2 — the ~0.72 ms/region is overstated by about **2.2x**, because chunking forces an assembly
write the byte model never charged

The region's product has to come back as one `[1,512,512,256]` tensor for the caller's residual
`add_`, and 134.2 MB cannot be L1-resident at 512 aa. So the assembled result lands in **DRAM** and
the concat is a 134.2 MB DRAM write the unchunked path never pays.

Traffic per region, from this card's own measured rates (388.86 GB/s DRAM read, 272.69 GB/s DRAM
write, ~778 GB/s L1 clone at this shape — all to be re-taken this pass, not inherited):

| arm | DRAM write | DRAM read | L1 write | L1 read | predicted ms |
|---|---:|---:|---:|---:|---:|
| k=1, DRAM output (production) | 268.4 MB | 268.4 MB | 0 | 0 | 0.984 + 0.690 = **1.674** |
| k>=4, L1 output + DRAM concat | 134.2 MB | 0 | 268.4 MB | 268.4 MB + 134.2 MB | 0.492 + 0.345 + 0.517 = **1.354** |

**Predicted destination term ~0.32 ms/region, not 0.72** — 1.19x on the region's traffic — i.e.
**~335 ms/fold ideal over 1048 regions, before any chunking overhead.** This is the number the
org should expect to be arguing about, not 760.

*Falsified if* the best measured k beats 0.45 ms/region, or if it comes in under 0.15.

### K3 — the crossover is between k=8 and k=16, and it is dispatch, not occupancy

Core utilisation stays high right through the sweep: `per_core_M` is the 5-snapped
`ceil(m_tiles / 110)`, so at k=8 (`m_tiles`=1024) it is 10 and **103 of 110 cores** are engaged, and
at k=16 (`m_tiles`=512) it is 5 and again **103 of 110**. So if chunking loses, occupancy is not
why. What grows is the program launch count: 3k launches per region instead of 3, plus one k-way
concat. **Predicted per-launch dispatch floor 25-60 us** (`size512-ab` measured a per-call floor of
that order on its blocked permute), so k=8 spends 21 extra launches x ~40 us = **~0.84 ms/region**,
which is 2.6x the predicted 0.32 ms saving.

**So I predict the net measured gain is NEGATIVE at k=8 and k=16, and the best cell is k=4 at
roughly break-even: −0.1 to +0.15 ms/region.** If the org wants the destination term at 512 aa it
will have to come from somewhere other than chunking this region.

*Falsified if* any k returns more than +0.2 ms/region net against the k=1 tuned-DRAM control.

### K4 — a DRAM-destination chunked region is slower than the unchunked one at every k

The direct read-across from `z-rowblock`'s variant E, which measured exactly this for the permute
and got 0.65-0.78x at every R. It is the control that proves the L1 destination is the lever and
the chunking is only the thing that makes the destination reachable.

*Falsified if* any DRAM-destination k>1 cell beats the k=1 DRAM control.

### K5 — every k is `torch.equal` against the k=1 tuned-DRAM reference

Row blocking splits M. `_pair_proj_config` derives `in0_block_w` from `k_tiles` and the cap only,
and `k_tiles` is unchanged by an M split, so the bf16 accumulation order over K is identical in
every cell. `per_core_M` and the subblock schedule change, but those are the drain schedule, not
the contraction order.

*Falsified if* any cell is not `torch.equal`, which would make row blocking a parity decision rather
than a layout one and would change what can be proposed.

---

## What this leg will NOT do

Phase 2. Probes and throwaway harnesses only. **`tt_bio/` is not touched.** No merge is proposed.
`SEQ_LEN_MORE_CHUNKING` is not moved — it gates eight unrelated chunked paths and the CTO has ruled
that nobody moves it in this wave.
