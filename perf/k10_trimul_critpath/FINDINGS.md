# k10-p1-trimul-critpath — the critical path inside the fused GenericOp

VERDICT: GO. The brief's premise is wrong in a way that opens the target rather than closing it.
The census's single `GenericOp` row is not one fused kernel, it is SEVEN distinct programs, and the
largest of them is limited by a DRAM bank-serialisation defect worth a measured 1.6652x on the op,
and 1.5034x of that comes from permuting the columns of a weight.

Owner: `k10-p1-trimul-critpath`. All device work on **pc, Blackhole p150a, card 0, 13x10 = 130
cores** (the grant named card 3; pc has exactly one card, `/dev/tenstorrent/0`, and its lease was
free). Host thread cap `runtime.host_thread_cap_env(1, host_threads=8)` on every run. The static
analysis is the committed **Blackhole** capture `perf/b2z2_byte_floor/art/ops_perf_blocksum_qb2c0.csv.gz`
(qb2 card 0, 11x10 = 110 cores). Nothing here is Wormhole. No model code changed.

## The premise: "it is already a fused GenericOp, so fusing more is unavailable"

Grouping the 176 `GenericOp` rows of the Blackhole capture by their input/output signature splits
them into seven programs with a 3.2x spread in mean call time. 22 instances of each per capture, 44
for the trimul gate move, so the capture holds 11 pairformer-layer executions; total GenericOp
device time 147.762 ms / 11 = 13.433 ms a layer, against `k10-instrument`'s independently measured
13.5101 ms/rep for one 512 aa `PairformerLayer`. Same block.

| program | mean us | % of GenericOp | s/fold at 3.76 s | TRISC0 input stall | TRISC2 output stall |
|---|---|---|---|---|---|
| `reblock_permute_gated` (trimul gate + channel move) | 994.5 | 29.6 % | 1.113 | **69.8 %** | 1.5 % |
| `triatt_sdpa` (q,k,v,bias -> attn) | 1323.8 | 19.7 % | 0.741 | **22.5 %** | 13.3 % |
| trimul fused in-projection, [.,128] @ [128,512] | 1188.8 | 17.7 % | 0.665 | **32.1 %** | 0.3 % |
| TriAtt qkv projection, [.,128] @ [128,384] | 865.8 | 12.9 % | 0.485 | 78.5 % | 1.5 % |
| `reblock_permute_back` (channel move home) | 511.9 | 7.6 % | 0.286 | 80.0 % | 1.9 % |
| TriAtt output projection, [512,4,512,32] @ [128,128] | 420.9 | 6.3 % | 0.237 | 78.1 % | 1.6 % |
| TriAtt gate projection, [.,128] @ [128,128] | 416.3 | 6.2 % | 0.233 | 77.9 % | 1.6 % |

Time-weighted, those seven give 56.7 % input stall against `perf/k10_thread_attrib/FINDINGS.md`'s
56.6 % for `GenericOp` on the same capture, so the split reproduces the aggregate it decomposes.
Counter-bound check on all 176 rows: `wait_front > TRISC0` 0, `reserve_back > TRISC2` 0.

**The 58 % aggregate describes none of them.** The two largest programs by time, the SDPA and the
trimul in-projection, are the two LEAST input-stalled at 22.5 % and 32.1 %; the four small matmuls
and the two channel moves run 69.8-80.0 %. Averaging those together produced a number that pointed
at neither group.

Two corrections that follow. **`trimul_tail` (F1) is not in this fold at all** — `F1_BLOCK_KEYS =
{(8, 8)}` and Boltz-2's c_z = 128 gives the key (4, 4), so it declines 100 % of these calls; the
label "fused trimul tail" on this row is wrong. And **three of the seven are separate matmuls off
the same 512x512x128 pair tensor** (128->512, 128->384, 128->128), so "fusing more is unavailable"
is false on its face: it is the specific claim that was never checked.

## INSTRUMENT-CONTROL

INSTRUMENT-CONTROL: the rig is ablation, not counters. It deletes one stage from the kernel source
and reads d(wall); a stage fully hidden behind another must read zero. Three checks that it does.

1. **A/A floor.** `base` is timed twice per round under two labels through the same descriptor.
   Over the three sessions: 1.00078x, 1.00122x, 1.00075x, i.e. **under 0.13 %**, against effects
   from 0.77 % to 106 %.
2. **Known-zero arm.** `notranspose` swaps `transpose_wh_tile` for `copy_tile`, deleting the FPU
   transpose while changing no CB traffic and no thread structure. Predicted: on a kernel this far
   from compute-bound it must read zero. Measured **+4.7 us (0.39 %), inside the A/A floor** and on
   the wrong side of zero. The rig returns "nothing there" when there is nothing there.
3. **Known-large arm.** `noread` deletes the reader's two DRAM page reads a tile. Predicted large.
   Measured **-624.1 us, 2.0647x**. The rig separates the two by 130x.

Every arm that is not a deliberate correctness break was checked on device against `base`:
`fusedsig`, `deepcb`, `deepcb32`, `deepcb64`, `batch2d16`, `batch4d32`, `batch8d64`, `readbatch8`
are all **bit-exact, max|d| = 0**. `base` itself sits 0.0179858 from an fp32 host reference, which
is bf16 rounding of the two-op sequence and not a defect. Arms marked WRONG DATA are timing-only
and are never a correctness claim.

## CRITICAL-PATH

CRITICAL-PATH: for `reblock_permute_gated`, xw [1,512,512,512] -> out [1,128,512,512], 1024 groups
of 32 tiles over 130 cores, **1206.9 us base on pc**. Read out of the three shipped kernels, with
each link's time measured by deleting it.

    NCRISC reader           per group, 32 x { 2 x noc_async_read_page (2 KB, DRAM)
                                              -> noc_async_read_barrier }
                            = 64 DRAM reads, 32 serialised barriers            624.1 us  ON PATH
       | c_0 (p) and c_1 (g), depth 2*GRAN = 4 tiles
    TRISC0/1/2 compute      3 stages a tile, each its own acquire/pack/release
                            through its own CB:
                              S1 copy g -> DST, SFPU sigmoid, pack -> c_2       90.3 us  ON PATH
                              S2 copy p + copy sig -> DST, SFPU mul, pack -> c_3
                                 (the multiply itself)                          97.3 us  ON PATH
                              S3 transpose_wh from c_3, pack -> c_16             4.7 us  HIDDEN
                            whole 3-stage structure collapsed to 1             100.0 us
       | c_16, depth 64 tiles; the writer takes a WHOLE GROUP, cb_wait_front(c_16, 32)
    BRISC writer            per group, 32 x { 64 x noc_async_read_one_packet
                                              (L1->L1, 32 B) -> read barrier
                                              -> 1 x noc_async_write (2 KB, DRAM) }
                              the 2048 L1->L1 gather transactions               66.8 us  ON PATH
                              the 32 DRAM tile writes                          205.1 us  ON PATH

The links sum to 1024.0 us of 1206.9, so **84.8 % of this op's wall is attributable to a single
named link and 15.2 % is shared floor**. Dispatch is not in it: `OP TO OP LATENCY` is outside
`DEVICE KERNEL DURATION`.

**The reader is the critical path and nothing else is close.** 624.1 us, 51.7 % of the op, 3.0x the
next link. That refutes my own written prediction, which named the writer's L1->L1 gather (the
kernel's own header calls instruction count on that RISC "a floor over all kernel structures"); the
gather is **66.8 us, 5.5 %**. `perf/k10_trimul_critpath/PREDICTED.md`, committed before any of these
numbers existed.

It also refutes the obvious fix. Batching the reader's DRAM reads behind fewer barriers is
**monotonically worse**, and deepening the circular buffer does not rescue it:

| reader | us | vs base |
|---|---|---|
| 1 tile pair per barrier (shipped), CB 4 | **1206.9** | 1.0000 |
| 2 pairs per barrier, CB 16 | 1273.4 | 0.9478 |
| 4 pairs per barrier, CB 32 | 1323.2 | 0.9121 |
| 8 pairs per barrier, CB 16 | 1348.0 | 0.8953 |
| 8 pairs per barrier, CB 64 | 1354.5 | 0.8910 |

## Why: the reader hits ONE DRAM bank for a whole group, and it is arithmetic

The reader's page index is `page = (row * Nt + jt) * Ctw + off + ct`. At 512 aa `Nt = 16` and
`Ctw = 16`, so the inner loop's stride is `Nt * Ctw = 256` and `256 % 8 == 0` on a part with **8
DRAM banks**. `jt`'s stride is `Ctw = 16`, also 0 mod 8. So the only term that can move the bank is
`off + ct`, and inside a group `ct` is fixed: **every one of a core's 64 reads for a group lands on
the same DRAM bank.** Worse, the two slices do not even differ — call `a` has `p_off = 8`,
`g_off = 0`, and `8 % 8 == 0 % 8`, so the value and gate reads share the bank as well.

Tested directly, by XOR-ing `il & mask` into the low bits of the page (which keeps the address
inside the same 16-page block, so it is always valid) and reading nothing else:

| DRAM banks a core touches | op us | vs base | reader time (arm - `noread`) |
|---|---|---|---|
| 1 (shipped) | 1210.3 | 1.0000 | 624.1 us |
| 2 | 1103.8 | **1.0964x** | 517.6 us |
| 4 | 977.9 | **1.2376x** | 391.7 us |
| 8 | 766.1 | **1.5799x** | 179.9 us |
| 8, with p/g CBs 32 deep | 746.3 | **1.6216x** | 160.1 us |
| free reads (`noread`) | 586.2 | 2.0647 | 0 |

Reader time roughly halves per doubling of banks. That is bank serialisation, not latency and not
bandwidth: the op moves 201.3 MB in 1206.9 us = **166.8 GB/s**, 38 % of the 435.2 GB/s p150a roof,
and its reads alone run at 216.8 GB/s while its writes, whose page index carries `jt` and therefore
does span all 8 banks, run at **327.2 GB/s**. Half the roof on the reads against three quarters on
the writes, from the same kernel, in the same call.

## RECOVERABLE

RECOVERABLE: on `reblock_permute_gated` alone, **1.113 s/fold of the 3.76 s** is in this program,
and the ablation puts a measured **1.6652x** on it, i.e. **0.445 s/fold** removed by reading the
same bytes in a different bank order. Three named changes. Each was measured twice: once through an
XOR proxy that just spreads banks, and once through the EXACT page pattern the change would produce,
with only the data made wrong. Where the two disagree the exact pattern is the number quoted.

- **B1, weight-column permutation of the fused in-projection.** Emit the wide projection as
  `[a, ga, b, gb]` instead of today's `[ga, gb, a, b]`, so each call's value slice and gate slice
  sit 4 channel-tiles, and therefore 4 DRAM banks, apart. This is a permutation of the projection
  weight's output columns at load: **no runtime cost at all, and bit-exact by construction** because
  it permutes which column a value is written to and read from, not any arithmetic.
  **Measured 1.5034x on the op, 0.373 s/fold** (`b1`).
- **B2, group key `(it, jt)` with the Ct = 4 channel tiles read inside** instead of today's
  `(it, jt, ct)`. That is the key the ungated writer originally used; it was changed to keep the
  work split even on a ROW BLOCK, and at the whole-tensor 512 aa move it costs nothing (256 groups
  over 130 cores, 2 waves). It makes `ct` vary inside the inner loop, and `ct` is the only term in
  the page index that can move the bank. **On top of B1, 1.5986x, 0.417 s/fold** (`b1b2`).
- **B3, p/g circular buffers 4 -> 32 tiles.** One line in `_build_gated`, **bit-exact on device,
  max|d| = 0 against `base`**, independent of B1 and B2. Alone **1.0367x** (median of three
  sessions: 1.0367, 1.0368, 1.0333). On top of B1+B2 **1.6652x, 0.445 s/fold** (`b1b2d32`), so it
  does not overlap away.

**B1 is 83 % of the whole prize and it is the cheapest of the three.** The mechanism is sharper than
"spread the banks": the reader issues the value read and the gate read back to back and then
barriers on both, and today those two reads land on the SAME bank, so every barrier costs two
serialised bank accesses instead of one. Splitting just that pair recovers 404.4 us of the reader's
617.6 us. The XOR proxy that alternates one stream over two banks reads only 1.0964x, so the
difference is not bank count, it is which stream sits on which bank.

Two levers priced and NOT worth taking, recorded so nobody re-runs them: batching the reader's
barriers is negative at every batch size tested (down to 0.8910x), and the `sig_cb` round trip is
worth **1.0078x**, 9.3 us. The round trip is also **bit-exact to delete** — `sigmoid_tile` under
`fp32_dest_acc_en = false` already applies `float_to_fp16b`, so the DST value is bf16 before it is
packed and the CB adds no rounding. The kernel comment's stated reason for that round trip does not
hold in the configuration it ships in.

Owed by the brief and answered here: **tile-movement to tile-math inside the compute kernel is
7 : 3**, four unpacks and three packs against sigmoid, multiply and transpose, or 3.5 : 1 counting
only arithmetic. **SFPU and FPU work are blocked, not interleaved** — the instruction stream is
three serial `tile_regs_acquire / compute / commit / wait / pack / release` blocks a tile group, so
the FPU `copy_tile`/`transpose_wh` and the SFPU `sigmoid_tile`/`mul_binary_tile` never issue in the
same DST window. That is the input `k10-p1-hol-vs-producer` needs and it does not have to re-derive
it. It also costs almost nothing here: collapsing all three stages into one (`passthru`) is
**100.0 us, 8.3 %**, so on this op the blocking is real and cheap.

## PREDICTS

PREDICTS: per-lever predicted value for Phase 2, on the 512 aa Blackhole fold of record, 17.989 s.
Op-level ratios are MEASURED on pc; the fold-second column divides this program's 1.113 s share of
the 3.76 s by the op ratio. Sub-additivity is not assumed — B1, B2 and B3 are priced as a stack by
`bank8d32` and must be approved as a stack.

| lever | op ratio | s/fold removed | fold ratio | bit-exact | measured by |
|---|---|---|---|---|---|
| B3 p/g CB 4 -> 32 | 1.0367x | 0.039 | 1.0022x | **yes, on device** | `deepcb32` |
| B1 projection column permutation | **1.5034x** | 0.373 | **1.0212x** | yes by construction | `b1` |
| B1 + B2 (ct inside the group) | 1.5986x | 0.417 | 1.0237x | yes by construction | `b1b2` |
| B1 + B2 + B3 | **1.6652x** | 0.445 | **1.0254x** | yes by construction | `b1b2d32` |

The bank ladder behind them, XOR proxies, mechanism only, not levers: 2 banks 1.0964x, 4 banks
1.2376x, 8 banks 1.5799x, 8 banks with a 32-deep CB 1.6216x, free reads 2.0647x. The exact B1+B2
pattern beats the 8-bank proxy (1.5986x against 1.5804x) because it puts the two reads of a pair on
two banks rather than spreading one stream over eight.

Two extensions Phase 2 should screen before building anything, because the defect is arithmetic and
not specific to this kernel: **`reblock_permute_back` reads with the same 256-page stride** and is
80.0 % input-stalled at 0.286 s/fold, and the three TriAtt projections read the same pair tensor
with a channel-tile stride that is also a power of two. The rule to check is one line: *a reader
whose innermost page stride is a multiple of the DRAM bank count serialises the whole loop onto one
bank.* On Wormhole the bank count is 12, not 8, so 256 % 12 = 4 and this kernel does NOT have the
defect there — which is why no Wormhole screen would ever have found it, and why the campaign's
`k10-transfer-function` k for a layout lever cannot be applied to it.

## Kill criteria, answered

The kernel source was readable at the level this needed, in one pass: 437 lines across the three
shipped kernels, and the page arithmetic that turned out to be the answer is nine lines of the
reader. The critical path WAS separable from the hidden waits, but not with the available counters —
`k10-p1-hol-vs-producer` is right that the counters cannot do it, and the reason is visible here:
the compute cluster reads 69.8 % input stall on this program while the whole compute cluster is
worth 100.0 us, 8.3 % of the wall. **A 69.8 % stall on a link that owns 8.3 % of the path is what a
counter cannot tell you and an ablation can.** No model code was changed.

## Artifacts

`perf/k10_trimul_critpath/` on `wk/k10-p1-trimul-critpath`: `PREDICTED.md` (committed before the
numbers), `gated_ablate.py` (the rig, 23 arms, kernel variants generated from the shipped sources by
asserted string edits into `/tmp`), `ablate_stage1_bh_pc.json`, `ablate_reader_bh_pc.json`,
`ablate_bank_bh_pc.json`, `ablate_b1b2_bh_pc.json`, `genericop_split.py` and `genericop_split.json` (the seven-program split).
