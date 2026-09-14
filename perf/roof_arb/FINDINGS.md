# Three byte totals, one fold: which one, and what the other two are

The campaign held three DRAM byte totals for the same Boltz-2 512 aa fold and three floors at the
same measured 424.7 GB/s streaming roof:

    2.9449 TB   real_traffic.py at the tip          ->  6.934 s   the published floor
    3.4050 TB   roof_table.py, b2x-baseline-attrib  ->  8.017 s
    4.0106 TB   the same, re-run under RANGE        ->  9.443 s

**They are not three readings of one fold.** Two axes separate them, both measured here on one code
path, and neither is a disagreement about what the fold moves:

| | STACK span rule | RANGE span rule |
|---|---|---|
| **tip `f072ae02f` captures** | 2.3189 TB | **2.9449 TB** (published) |
| **b2x-baseline-attrib captures** | **3.4047 TB** (of record) | **4.0106 TB** |

* **SPAN, 1.178x, an instrument bug.** ttnn.graph drops `function_end` for some device operations,
  so a big capture is unbalanced; `itemize.top_level_spans`'s STACK rule then never returns to depth
  zero, collapses 428 top-level ops into 4, and a buffer read by ten ops is charged one read. RANGE
  owns every node by the last `ttnn.*` `function_start` at or before it and recovers all 428. The
  balanced `DiffusionModule` capture reads 6600.387 MB under both rules, which is the control.
* **TIP, 0.734x, a real change in the fold.** The two capture sets are two commits. The older one
  folds 7168 atoms where the tip folds 4480, and it predates every byte-deleting lever the campaign
  landed.

So 3.4050 TB is 4.0106 TB seen through a broken span rule, and 4.0106 TB is the tip's fold before
the levers. Nothing here needs averaging, and 2.9449 TB is the only one of the three that describes
the code that runs today. It is also still wrong, by less than any of the gaps above.

## The audit: all 22 pre-allocated buffers are writes

`roof-redteam-2` left this undecidable: `ttnn.generic_op` takes its destination in the same flat
tensor list as its operands, so `itemize`'s consumer set cannot tell a pre-allocated buffer's writer
from its readers, and one trimul block was 1745.6 MB or 2550.9 MB depending on the assumption.

`audit_prealloc.py` walks the AST of every file in `tt_bio` and reports, per site, the helper the
buffer is handed to, the argument slot, and whether it is returned. **22 sites, 22 destination
slots, 22 returned.** Exhaustive, and it catches the two sites a grep-and-read audit drops: the
`outs.append(...)` at `triatt_qkv.py:417`, which has no assignment target, and `esmc.py:735`, where
the buffer reaches its writer through a `ttnn.experimental.view` alias.

The slot is not always position 4. `G.generic_minimal_matmul(dev, in0, in1, outs, ...)` takes it
there, `ttnn.generic_op([xa, wa, xb, wb, out], pd)` puts it last in the io list, `SG.sdpa` takes it
at position 5, `reblock_permute_gated` takes it as `out=`, and the four RFdiffusion3 `_persist`
sites hand it to `ttnn.copy_host_to_device_tensor`. The invariant is the destination, not the index.

**Two sites are not single-writer,** and they are the audit's only non-uniformity:
`tt_bio/tenstorrent.py:5931` writes `a` and `b` once per row block, and `tt_bio/esmc.py:735` writes
`dst` once per block through the view. On such a buffer the correction below would charge writers
2..n a full read each. Neither fires at 512 aa: ESM-C is not in this fold, and all eight gated calls
across the two trunk captures take the whole `[1, 512, 512, 512]` projection with the row axis at
full length, not a row block (`multiwriter_guard` in `arbitration.json`, `gated_calls_row_blocked`
0 everywhere). Above 512 aa the row-blocked path can open and the correction must be re-guarded.

## Three corrections, itemised

`corrected_traffic.py` reproduces `real_traffic.py` byte-for-byte on all 26 captures with the
corrections off, so every delta below is a change to one charging line and not a substituted
instrument.

| | pairformer MB | MSA MB | denoiser MB | fold TB | floor s | delta s |
|---|---|---|---|---|---|---|
| published | 6758.294 | 22100.181 | 4035.777 | 2.9449 | 6.934 | |
| `+L1` | 6902.063 | 22247.678 | 4051.112 | 2.9883 | 7.036 | +0.102 |
| `+L1+PRE` | 6969.172 | 22449.005 | 4051.112 | 3.0093 | 7.085 | +0.049 |
| `+L1+PRE+GATE` | **6432.301** | **21912.134** | **4051.112** | **2.8589** | **6.732** | -0.354 |

The three interact, so they are a ladder and never three addends: `GATE` is worth nothing until
`PRE` makes the gated calls readers at all, and `PRE` is worth +67.1 MB on the pairformer block with
`L1` on and 0 with it off.

**L1** — `moves_dram` is built from the DRAM rows alone, so an op whose output landed in L1 is not
charged as a reader of the DRAM it consumed. Widened to "allocated anything, DRAM or L1".
`roof-redteam-2`'s finding, reproduced to the megabyte.

**PRE** — a pre-allocated buffer is charged twice: a write to `ttnn.allocate_tensor_on_device` and
then a phantom read, because the consuming `generic_op` allocates nothing of its own and so fails
`moves_dram`. Now the allocation moves nothing (the capture shows `create_device_tensor` plus
`buffer_allocate` and no host copy) and its first consuming op is charged the write; later consumers
become readers, which they were not before.

**GATE** — `reblock_permute_gated` reads two channel-tile slices of the wide fused projection, not
all of it. `reader_reblock_permute_gated.cpp` issues exactly two `noc_async_read_page` per
(row-tile, col-tile, channel-tile) group, at `p_off + ct` and `g_off + ct` with `ct < Ct`; at the
trimul Ct = 4 against Ctw = 16, so a call reads half its input, and the two calls sharing one
projection read disjoint halves. N = 512 is a multiple of the tile height, so the reader's
padding-row path never fires and the fraction is exact, not a model.

The trimul bracket `roof-redteam-2` could not close: **1611.366 MB**, below both ends of its
1745.6-2550.9 MB range, because the right answer charges the write once to the op that performs it
and the reads at the fraction the kernel issues.

## Calibration: two known-answer cases, one of them on the pre-allocated path

A dense bf16 matmul's minimum byte count is exactly `(M*K + K*N + M*N) * 2`.

**The 8192 cube, a plain `ttnn.matmul`.** The counter returns 536.871 MB against an exact
402.653 MB, 4/3, reproducing `roof-budget`. The whole excess is 134.218 MB and it is one term: the
output buffer has no consumer inside a one-op capture, so the counter's own no-consumer rule charges
it a read. Subtract that term and the ratio is 1.000000. **Stronger than roof-budget stated it**: on
all sixteen ladder shapes `bytes_counter == bytes_arith_min + batch*M*N*2` holds exactly, so the
overcount on a dense matmul is always one phantom read of the output and never anything else. Raw
ratios run 1.111 to 1.889 purely with the shape. The corrected counter returns the same numbers
here, which is right: a plain matmul has no pre-allocated buffer and no L1 output, so there is
nothing for the corrections to touch.

**The fused trimul in-projection, which does exercise the pre-allocated path.** It is the same
M=262144, K=128, N=640 the ladder runs as a plain `ttnn.matmul`, so its exact minimum is
402.817024 MB. Charged over its three ops (two allocations and the fused `generic_op` that writes
both destinations):

    published   603.980 MB   1.4994x
    +L1+PRE     402.817 MB   1.000000x

The correction lands on the arithmetic answer to the byte, on a case the published counter misses by
half its own output. A counter that got this wrong would be unfit; this one does not.

## One floor

Granularity: **per top-level `ttnn` op**, which is the finest the captures support, then summed over
each unit's calls. Max taken per op, not on the aggregate.

    corrected bytes, traffic only, no max               6.732 s
    corrected bytes, max per op, dense-cube 104.93 TFLOP/s   7.086 s
    corrected bytes, max per unit, shape-honest roof        11.127 s

`roof-shape-honest-roofs` has landed (VERDICT GO) and its number is used: the fold's own matmul
shapes reach a FLOP-weighted harmonic mean **18.8 %** of the dense cube over 97.1 % of the census
FLOPs, i.e. 19.73 TFLOP/s, giving an arithmetic floor of 11.134 s. This work reproduces 11.127 s
from its own per-op FLOP sum, 0.06 % apart.

**The floor of record is 11.134 s and arithmetic sets it.** The traffic term never binds: at 6.732 s
it is 1.65x below the arithmetic one, and deleting every byte the fold moves would not move the
floor. The 17.340 s cell is 6.206 s above the binding floor, not the 10.406 s the published
bandwidth floor implied.

Two per-op numbers reconciled while getting here. `roof-redteam-2`'s per-op bracket 7.036-8.194 s
reproduces: 8.099 s from its own committed script and captures at published bytes, and 8.195 s with
the L1 correction on against the 8.194 s its table row states, the 0.001 s being weight bytes with no
owning op that this counter keeps in the per-op sum. With all three corrections the same
per-op max against the same dense-cube roof is **7.086 s**, 1.16x below 8.194 s, because charging
the fused `generic_op` kernels their real bytes is precisely what stops them reading as compute
bound. The per-op floor with the shape-honest rate applied uniformly is 15.193 s; it is quoted only
as an indication, because that rate is an aggregate over 15 classes whose own spread is 13.0-76.4 %
and applying it op by op is not something this task measured.

## Reproducing

    git checkout origin/wk/roof-budget    -- perf/roof_budget
    git checkout origin/wk/roof-redteam-2 -- perf/roof_redteam2      # for the cross-check only
    python3 perf/roof_arb/audit_prealloc.py        # the 22-site audit
    python3 perf/roof_arb/arbitrate.py             # everything else -> arbitration.json
    python3 perf/roof_arb/corrected_traffic.py perf/roof_budget/captures/cap_*.json.gz

No device. Committed captures, committed scripts and the source only. The 424.7 GB/s streaming roof
and the 104.93 TFLOP/s dense-cube roof are `roof-budget`'s, measured in one session on qb2 card 2,
and neither was re-measured here.
