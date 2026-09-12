# b2z2-trunk-byte-floor — the block's redundancy surface is 5.17 %, and a deleted read is worth 1.76x its bytes

TASK TYPE: VERIFY/BENCHMARK then one build off the census | PLAYBOOKS loaded: VERIFY/BENCHMARK +
ACCELERATE + ALWAYS-ON | memories read: `b2z2-radical-2x-wave2`,
`ttnn-graph-byte-count-must-dedupe-buffer-not-tensor-id`, `no-speedup-by-skipping-the-models-own-work`,
`roofline-roof-must-be-measured-not-asserted`, `one-size-tuning-is-a-standing-defect-class`,
`unified-solution-not-per-model-patches`, `merged-lever-defaults-off-is-not-a-landed-win`,
`perf-gate-single-shot-legs-recurring-false-alarm`, `baseline-median-must-recompute-from-draw-log`,
`whglx-unpinned-all-chip-open-breaks-every-cotenant`, `parity-gate-scores-installed-package-not-checkout`

VERDICT: GO. The census is complete, the table is published, the top site's tractable half is built,
  bit-exact and measured. **The byte model UNDER-prices a deleted read by 1.76x**, which is the
  pre-registered falsifier firing in the direction nobody set it up for.
BRANCH: wk/b2z2-trunk-byte-floor (pushed). `ef77563a` the pre-registered prediction, `8e05277c` the
  buffer tracer, `5b081c49` the census and its table, `07f3d58d` the fused projection and its
  bit-exact check, `ca1c8360` the A/B harness, `6f810244` the refactor, `78371796` the results.
CARD: whglx card 10, pinned `TT_VISIBLE_DEVICES=10`, leased as `worker:b2z2-trunk-byte-floor`,
  trace region capped at 512 MiB. No other card opened, no reset run, pc's banned card untouched.
  Worktree on whglx is `~/wt-bytefloor` (tt-admin), a clone — the shared checkout was not touched.
ARCH: WH for every number this row measured. The published cell is Blackhole; fractions transfer,
  seconds do not, and the one fold figure below is labelled PROJECTED and is under the fold's own
  measurability floor.

PREDICTED: `state/b2z2-trunk-byte-floor.PREDICTION.md`, committed at `ef77563a` before the first
  measurement. P1 removable bytes **6.5 %, band 3-12 %**, with R1 (the triangle attention's normed
  pair tensor read three times) named as **268.4 MB** and R2 (the trimul's read twice) as 1-3 %.
  P2 block ratio 1.02x, band 1.00-1.05x. P3 the fused-kernel eligibility census must not move.

MEASURED: **P1 CONFIRMED at 5.17 %, inside the band and below the point. R1 exact to the megabyte.
  P2 CONFIRMED at 1.0146x. P3 held. The realization coefficient is REFUTED and it is the finding.**

## The census — 416.3 MB, and 96.7 % of it is one defect shape

One `PairformerLayer` at 512 tokens, every operand keyed on its device ALLOCATION, not on the tensor
id and not on the shape (`perf/b2z2_byte_floor/trace_block.py` + `census.py`, table in `CENSUS.md`).
It agrees with the Blackhole capture the campaign's 8.0493 GB comes from op for op: 41 layer norms,
113 matmuls, 16 `generic_op`s in both.

| | this trace (WH c10) | BH census |
|---|---|---|
| DRAM read / write | 3,672.0 / 3,744.9 MB | 4,710.8 / 3,338.5 MB |
| DRAM total | 7.417 GB | 8.049 GB |

**REMOVABLE BYTES: 416.3 MB = 5.17 % of 8.0493 GB.**

| rank | site | removable | % of 8.0493 GB |
|---|---|---|---|
| 1 | triangle attention's normed pair tensor read THREE times — bias (128->32), qkv (128->384), gate (128->128) each read the whole 67.1 MB. x2 | **268.4 MB** | 3.33 % |
| 2 | triangle multiplication's normed input read TWICE — the fused four-way in-projection and the output gate `g_out`. x2 | **134.2 MB** | 1.67 % |
| 3 | transition `swiglu` weights re-read once per row chunk | 12.4 MB | 0.15 % |
| 4 | four small shared mask/bias tensors | 1.5 MB | 0.02 % |

Everything else in the block is read exactly once. Two call sites own every removable byte:
`_pair_proj_linear` (tenstorrent.py:3848) and `mm_generic.generic_minimal_matmul` (:359), reading
what the `ttnn.layer_norm` above them just wrote.

**The 3,338.5 MB of WRITES are not a redundancy.** 58.8 % of the block's traffic is 63 single-use
DRAM intermediates, written by one program and read by exactly one other: 2,181.0 MB each way.
Deleting those is program fusion, which this campaign already priced — the Pairformer megakernel
lost 2.9 %. Nobody should re-open the write side expecting to find a byte read twice; there is not
one larger than 2.1 MB.

## The build — rank 1's tractable half, bit-exact

`triatt_qkv.qkvg_heads`: q, k, v and the gate from ONE pass over the normed pair tensor, instead of
two matmuls that each read all 67.1 MB of it. **No kernel change.** The writer already splits the N
axis into `N_chunks` equal buffers (qkv is 12 N tiles as 3x4, the gate is 4 as 1x4); four chunks of
four is the same writer with one more destination. Bit-exact by the argument `_MM_BLOCK` already
makes for its own neighbouring entries: boltz2's qkv key (4,12) and gate key (4,4) both carry
`K_block = 4`, the whole contraction, so the fused key (4,16) folds K identically and every output
element is accumulated in the order it is accumulated today.

BIT-EXACT: `torch.equal` True on all four outputs, max abs **0.0**, and the negative control (one
perturbed gate weight column) differs, so the comparison can fail. `perf/b2z2_byte_floor/bitexact.py`.
The block's own output sum is identical arm to arm in every rep of every leg (-4713.108).

BYTES-REMOVED: **-134.2 MB of DRAM reads, +0.0 writes, +0.0 L1**, re-counted on the same instrument
with the lever on. **1.81 % of this trace's 7.417 GB, 1.67 % of the campaign's 8.0493 GB.** The
redundancy surface falls 416.3 -> 282.1 MB.

BLOCK-RATIO: **1.01459x** (median of 7, paired interleaved, one process, one card) against an
**A/A floor of 0.99983x** measured the same way in the same run, A/A spread 0.23 %. Dropping each
arm's first draw: 1.01491x. `perf/b2z2_byte_floor/out/ab_512_wh_c10.json` carries every draw.

ELIGIBILITY-CENSUS, per arm, identical in both: `qkv_heads` 28/28 served, head-major tail 28/28
served, `reblock_gated` 56. The lever does not drop a tuned kernel, it adds a call to one. This is
the check that flipped `b2z2-byte-axis-reopened`'s sign and it is flat here.

TILE-MOVEMENT-DELTA: -2.85 % of the block's input-tile wait. The arm removes 1.44 % of the
block wall (1.235 ms of 85.884, WH) and every millisecond of it has to come out of the movement
term: the arithmetic is bit-identical, the MAC count is unchanged, the output buffers are the same
four, and the only program property that moves is N. Carried onto the BH decomposition the
campaign scores in (wait 18.3366 ms of a 36.3438 ms block), 1.44 % of the block is 0.523 ms, which
is **2.85 % of the wait**. Measured on WH and transferred as a fraction, per CONTEXT
§2-CORRECTION: fractions transfer, seconds do not.

DEFICIT-SECONDS: 0.0 s removed from the shipped path, because the lever's default is OFF and a
fold A/B has not run. **0.147 s/fold located and measured on the block** (1.44 % of the trunk's
10.22 s of BH device span), which is the number a fold A/B would have to find. Separately this pass
EXPLAINS a term rather than removing it: the 1.76x realization above says the campaign's DRAM-byte
ledger mis-prices its own remaining levers by up to that factor in either direction depending on
whether the operand is multicast, and the remaining redundancy ledger (282.1 MB) is worth
re-pricing before the next row spends a pass on it.

FOLD-RATIO: **not claimed.** PROJECTED 1.0074x on the 20.079 s BH cell (1.01459x applied to the
trunk's 50.9 % share), which is under the cell's own 1.01x measurability floor. A fold A/B is the
next pass, not this one.

## The finding: a deleted read is worth 1.76x its DRAM bytes

The byte model this campaign replaced the tile model with says 45.2 % of the block is
byte-proportional (`b2z2-byte-axis-reopened`, corrected). 1.81 % of the bytes should therefore buy
0.82 % of the block — **0.702 ms of 85.884**. Measured: **1.235 ms. Realization 1.76**, against the
0.53-0.58 the same row measured for its own byte changes.

**The reason is the denominator, not the model.** The 8.0493 GB counts each operand tensor once "so
multicast counts once and this is a lower bound" (`b2z2-redteam-v2`, C1). The operand deleted here
is a matmul's `in0`, which is multicast to a whole row of the grid: one DRAM read, many delivered
bytes, and the block's cost is the delivered bytes the math thread waits on. So **removable bytes
priced in DRAM currency under-price a multicast operand and over-price a streamed one**, and the two
are not interchangeable. That cuts both ways and it explains the earlier row too: its 20.13 % byte
cut was a STORAGE-dtype change, which shrinks delivered bytes at every site uniformly and pays the
program-config penalty, while this is a DELIVERY change at one site that pays nothing.

Practical consequence for the remaining ledger: rank 2 (134.2 MB, the trimul's `g_out`) is the same
shape of operand at the same multicast width, so it should carry the same ~1.76 realization —
**another ~1.0146x on the block if it is built** — and the bias-projection third of rank 1 is a
narrow 32-wide read that is not multicast the same way and should be expected to pay less.

PARITY: bit-exact and proved twice — `torch.equal` at max abs 0.0 on all four projections with a
live negative control, and the block's output identical in all 28 timed calls. The default is OFF
(`TT_BIO_TRIATT_FUSED_QKVG`), so the shipped path is byte-for-byte untouched and no gate is owed
until the default flips. Nothing in the tree changes for a model that fails any guard: `qkvg_heads`
declines to exactly the union of what `qkv_heads` and `gate_proj` decline to, on shape properties
and never on a model name.
CHEAT-CHECK: clean. The lever computes the identical four tensors from the identical weights; it
changes which buffer a read comes from, not what is computed. No sampling step, recycle or MSA row
is skipped anywhere in this row.

## Next pass, in order

1. **Fold A/B with the flag on**, 512 aa, the full protocol, to turn the 1.0074x projection into a
   measurement or refute it. Only then is a default flip arguable.
2. **Rank 2, the trimul's `g_out`**: fold the output gate's weight into the fused four-way input
   projection so the normed input is read once. 134.2 MB, same operand class, predicted ~1.0146x on
   the block by the realization this pass measured. The obstacle is holding a [1,512,512,640]
   tensor across the trimul instead of [1,512,512,512]; the gate is consumed at the tail.
3. Re-price the ledger in DELIVERED bytes rather than DRAM bytes, since this pass showed the two
   rank levers differently. `b2z2-bh-tile-census` already has the multicast counts.
