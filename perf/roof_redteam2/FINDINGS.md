# roof-redteam-2: the 6.934 s floor and the table built on it

VERDICT: **the floor is too low and the ranking is not safe below rank 1.** Three of five attacks
broke the claim, one confirmed it, one broke the hypothesis and found a different defect in the
same place.

Under test: `perf/roof_budget/ROOF_BUDGET.md` + `roof_budget_512_qb2c2.json` on
`origin/wk/roof-budget`, qb2 card 2 (p300c, 11x10, AICLK 800 MHz), tip `f072ae02f`.
Pre-registration: `PREREGISTER.md`, committed before anything was computed. Two of my five
predictions were wrong and are marked as such.

Instrument calibration first. `roof_budget_table.py` re-run here on its own committed captures
reproduces the published figures exactly: 219.49 TFLOP, 2.9449 TB, floor 6.934 s, and every row of
the table. Everything below is a change to that script's inputs or to where it takes a max, never a
different instrument silently substituted.

| # | attack | outcome |
|---|---|---|
| 1 | contention scaling reorders the ranking | **BROKEN below rank 1** — one fitted per-op host gap swaps rank 2 and rank 3 |
| 2 | the three units do not tile the fold | CONFIRMED — disjoint and 99.65 % exhaustive; the 0.6 % the brief flagged is an identity |
| 3 | the floor sums traffic instead of max(traffic, compute) | **BROKEN** — it does; 6.934 s should be 7.036-8.194 s |
| 4 | the 35-row MSA is a benchmark artifact | **BROKEN** — the same repo holds the same target's real 8833-row alignment |
| 5 | the two byte instruments disagree and neither says so | CONFIRMED, but not for the reason given; three numbers, 2.9449 / 3.405 / 4.0106 TB |

---

## Attack 1 — the ranking is not safe below rank 1

Predicted: robust. Wrong at rank 2.

Start with what the scalar is. `cell_scale = 17.340 / 24.731` where 24.731 is
`baseline_summary.plain_median_s`, and that is the **mean of two reps**, 27.415 s and 22.047 s, at
loadavg 28.11 / 28.60 / 27.72 (`attrib_512_tip_qb2c2.json`, `n: 2`). The two reps are 1.24x apart,
so the same session supports any scalar in 0.6325-0.7865. Separately, the `ms/call` medians the
scalar multiplies come from a **different run**: the instrumented fold, 74.159 s wall, whose own
`note` field reads "per-call device syncs serialise host and device; this fold's wall is a
diagnostic, not a baseline".

Now the non-uniformity, measured rather than modelled. Dispatch density over the three units, from
the same captures the table uses:

| unit | ttnn ops/call | ms/call | ops per ms |
|---|---:|---:|---:|
| `PairformerLayer\|1x512x384,1x512x512x128` | 410 | 51.924 | 7.90 |
| `MSALayer\|1x512x512x128,1x1024x512x64` | 1064 | 169.362 | 6.28 |
| `DiffusionModule\|` | 1702 | 41.128 | **41.38** |

The denoiser dispatches 5.2x more ttnn ops per millisecond than the pairformer block and 6.6x more
than the MSA block. And one level down there is direct evidence, not a proxy:
`DiffusionTransformer|1x512x768,1x512x768` measures 27.194 ms and its 24 child
`DiffusionTransformerLayer` calls sum to 20.105 ms, while the parent capture contains **exactly the
same 1224 ttnn ops** as the 24 children. So 7.089 ms per call, 26.1 %, is time inside the denoiser
with no device op in it at all. The same subtraction gives 2.644 ms (5.1 %) for the pairformer block
and 1.941 ms (1.1 %) for the MSA block.

Give host contention one cost per dispatched op, the same everywhere, and fit it so the three units
still sum to their share of the cell. That gap is **15.82 us/op** and it lands:

| unit | published waste at the cell | per-op-gap model | k |
|---|---:|---:|---:|
| pairformer block | 5.410 | **7.795** | 0.875 |
| denoiser step | 3.867 | **0.942** | 0.346 |
| MSA block | 1.067 | **1.608** | 0.901 |

Rank 2 and rank 3 change places. The threshold is a gap of **13.76 us/op**; anything above that
swaps them, and the value that reproduces the cell is 15.82.

Rank 1 survives everything. For the denoiser to pass the pairformer it needs `k = 0.889` against the
pairformer's 0.701, a ratio of **1.267** — the denoiser would have to suffer *less* from host load
than the pairformer while dispatching 5.2x as many ops per millisecond. The MSA block would need
`k = 1.734`, which is not a number.

What this does to the campaign: the pairformer block is the right rank-1 row on any model. The
denoiser's 3.867 s is a single-scalar artefact of the same size as the number itself, and the queue
should not treat the 3.867 vs 1.067 gap as real. The fix is cheap and needs no card: take the
per-unit times from a benchlocked run, or measure the host gap per unit instead of fitting one
scalar to the whole fold.

## Attack 2 — the three units do tile the fold, and the 0.6 % proves nothing either way

Predicted: disjoint yes, exhaustive no. Wrong on the second half.

Disjointness is settled by the instrumented tree in `attrib2_512_tip_qb2c2.json`, which the table
never reads. `PairformerLayer|1x512x512x128` (16 calls) is `TrunkModule/MSA/MSALayer/
PairformerLayer` — a child of the MSA block, exactly as the brief suspected. `Diffusion`,
`DiffusionTransformer` and `DiffusionTransformerLayer` all hang under `DiffusionModule`. So the
nesting the brief points at is real. `roof_budget_table.py`'s `TOP` list names none of them, so
nothing is double counted.

Three independent additivity checks, one per unit, and the arithmetic closes on all three:

| parent | GFLOP parent | GFLOP children | ms parent | ms children | ops parent | ops children |
|---|---:|---:|---:|---:|---:|---:|
| pairformer block | 508.189 | 508.2 | 51.924 | 49.280 | 410 | 395 |
| MSA block | 1368.318 | 1368.3 | 169.362 | 167.421 | 1064 | 1029 |
| `DiffusionTransformer\|1x512x768` | 289.998 | 290.0 | 27.194 | 20.105 | 1224 | 1224 |

Exhaustiveness holds too, and the session's own numbers say why rather than the coincidence the
brief suspected. The three units cover 24.644 s of a 24.731 s plain fold (99.65 %). The same
session's instrumented arm splits its 25.465 s into `device_s 25.0561`, `host_s 0.401`,
`transfer_s 0.0002` and `residual_s 0.0071` — so there is only 0.4 s of host time in the whole
timed region for anything to be outside those three units, and featurisation and the CIF write are
inside that 0.4 s.

The one thing the brief is right to distrust is the 0.6 % itself, but not for the stated reason. It
is an identity. `top_level_s_above_roof_at_cell = top_s * scale - floor` and `cell_s_above_roof =
cell - floor` subtract **the same floor**, so 10.345 vs 10.406 measures only the 0.35 % coverage gap
and carries no information about whether the units overlap. Quoting it as corroboration is circular.

One real defect, small: bytes are not additive the way FLOPs are. The pairformer children sum to
6353.4 MB against the parent's 6758.3 (404.9 MB, 6.0 %, in the parent's own 15 ops, which is fine),
but the `DiffusionTransformer` children sum to 2912.2 MB against the parent's 2894.1 — the children
are 0.6 % **over** the parent. A capture boundary charges a buffer that the enclosing capture sees
as internal. Per-child byte shares in this table are good to about a percent, no better.

## Attack 3 — the floor sums traffic for every unit, including the ones that are not traffic bound

Predicted: broken, 5-15 %. Right, at the top of that band.

`roof_budget_table.py` decides the binding roof per row and then does not use it:

    "traffic_floor_s": round(fold_B / stream_roof, 3),
    "binding_floor_s": round(fold_B / stream_roof, 3),

There is no `max` in the file. At the granularity of the three top-level units this costs nothing —
their arithmetic intensities are 73.6, 61.5 and 78.3 FLOP/byte against a 247.1 balance. It is not
free one level down, because a roofline floor on an aggregate is at or below the sum of its parts'
floors: `max(sum) <= sum(max)`.

Re-evaluated on the same two roofs (424.7 GB/s stream, 104.93 TFLOP/s dense cube), the same three
captures, changing only where the max is taken:

| floor | s | vs published |
|---|---:|---:|
| published: aggregate traffic, `real_traffic.py` verbatim | 6.934 | 1.000x |
| aggregate traffic, L1-output readers charged (attack 5) | 7.036 | 1.015x |
| **per-op max, L1-output readers charged** | **8.194** | **1.181x** |
| per-op max, compute priced at the shape-honest 26.38 TFLOP/s | 14.058 | 2.028x |

The honest headline is a bracket, **7.036-8.194 s**, not a point. The 1.158 s increment splits
1.006 / 0.128 / 0.025 s over the pairformer block, the MSA block and the denoiser, and the
pairformer's 1.006 s rests on **8 `ttnn.generic_op` fused kernels the byte counter charges zero
bytes** (attack 5) which therefore read as infinitely compute bound when they are not. The
denoiser's 0.025 s has no zero-byte ops in it and is genuine. Either way the published 6.934 s is
below the bottom of the bracket and the 10.406 s of headroom is at most 9.146 s.

The fourth row uses `wk/roof-pair-transition` (6d8d50b00), which measured the pair Transition's own
shapes at 31.40 TFLOP/s against a 124.88 TFLOP/s dense cube in its session — 25.14 %. That session
is a **different card** (pc p150a, 130-core firmware), so only the ratio transfers: 25.14 % of this
card's 104.93 cube is 26.38 TFLOP/s. Applying one unit's shape rate to every op in the fold is a
fiction in the opposite direction to the dense cube, and it is in the table to bracket how little of
the 10.406 s headroom is safe, not as a number to quote.

## Attack 4 — the 35-row MSA is a benchmark artifact, and the repo already holds the counter-example

Predicted: broken. Right, and more cleanly than expected.

The published cell folds `perf/size512/fixtures/cdk2x2_512.yaml`, `n_msa: 35`. Every one of the 30
a3m files in that directory carries exactly 35 sequences, from 128 aa to 1568 aa, so the depth is a
constant of the harness and not a property of any target.

The same repository holds the real alignment for the same protein.
`perf/capacity/cdk2_1hcl_colabfold_deep.a3m.gz` is the committed ColabFold search for CDK2
(PDB 1HCL) — the exact 298 aa chain that `cdk2x2_512` tandem-repeats — and it is **8833 rows, 8832
unique**. `scripts/capacity_fixture.py` uses it and says why in its own docstring: a single-sequence
pass "proves nothing a user hits".

What a deployed request gets: `--max_msa_seqs` defaults to 8192 in the CLI and 16384 in the worker,
`--subsample_msa` is off by default, and when no MSA source is given
`tt_bio/main.py:_resolve_msa_default` falls through to step 5 and enables the online ColabFold
server. So the deployed depth for this target is the search result capped at 8192, not 35.

The depth axis is the three units whose shapes carry it — `OuterProductMean`,
`PairWeightedAveraging` and `Transition|1x1024x512x64` — 15107.2 MB of the MSA block's 22100.2
(68.4 %) and 1.305 s of the 17.340 s cell at a padded depth of 1024. That cost scales with the
padded depth, so:

| MSA depth | padded, 1024 rule | padded, ladder | depth axis at the cell | ladder prize |
|---|---:|---:|---:|---:|
| 35 (the fixture) | 1024 | 64 | 1.305 s | **1.223 s** |
| 1024 | 1024 | 1024 | 1.305 s | 0.000 s |
| 8192 (deployed cap) | 8192 | 8192 | **10.438 s** | **0.000 s** |
| 8833 (the real search) | 9216 | 9216 | 11.743 s | 0.000 s |

`wk/roof-msa-ladder` (c0802981a) ships `MSA_PAD_LADDER = (64, 128, 256, 512, 1024)` and above the
top rung it falls back to multiples of 1024, so its prize is **exactly zero at any alignment 1024
rows or deeper**. The campaign's rank-1 row is a shallow-alignment lever. It is a real lever for
shallow alignments — orphan sequences, designed binders, `--single_sequence` — and it is worth
nothing on the deep-alignment folds JapanFold serves by default.

The second consequence is the bigger one. At the deployed cap of 8192 the depth axis costs 10.44 s
instead of 1.305 s, and the 512 aa fold projects to **26.5 s, not 17.340 s**. The published cell is
not the cost of folding this target; it is the cost of folding it with a 35-row alignment.

Two honest caveats. The projection assumes those three units scale linearly in rows, which is what
their shapes say and what the table's own MSA note assumes, but nobody has run it — that is a
measurement, and it needs a card. And `cdk2x2_512` is a tandem repeat whose structure is meaningless
anyway, which `scripts/capacity_fixture.py` states outright about the identical construction; the
depth finding does not depend on that, but a "representative 512 aa fold" claim would.

## Attack 5 — the hypothesis is wrong, the conclusion is right, and there are three numbers not two

Predicted: `real_traffic.py` has its own defect of the same sign. Right, but smaller than the one
that matters.

`real_traffic.py` does **not** share `census.py:split_io`'s defect. Round 1 found `split_io` drops
the read of a read-modify-write; `real_traffic.counts` charges an in-place op `rd += size` **and**
`w += size` on its destination. That rule is correct and the tip's 2.9449 TB is not affected by
round 1's 8.49 %.

It has a different one:

    def moves_dram(i):
        name = ops[i]["name"]
        if name in NO_TRAFFIC: return False
        return alloc_by_op[i] > 0 or name.endswith("_")

`alloc_by_op` is built from the DRAM rows only, so it is zero for an op whose output was allocated
in L1, and such an op is then not a reader of any DRAM it consumes. In
`cap_PairformerLayer__1x512x384,1x512x512x128`, `ttnn.linear` op 19 reads **67.14 MB of DRAM**,
allocates 67.11 MB in L1, and is charged nothing. 110 of that capture's 410 ops are in the class.
A buffer whose only consumers are all in the class falls through to `if not readers:` and is charged
once against its producer, which bounds the damage. Widening that one line to "allocated anything,
DRAM or L1":

| | published | corrected | |
|---|---:|---:|---:|
| pairformer block | 6758.3 MB | 6902.1 MB | +2.13 % |
| MSA block | 22100.2 MB | 22247.7 MB | +0.67 % |
| denoiser step | 4035.8 MB | 4051.1 MB | +0.38 % |
| **fold** | **2.9449 TB** | **2.9883 TB** | **+1.47 %** |
| floor at 424.7 GB/s | 6.934 s | 7.036 s | |

The unresolved one is structural and larger. Every `ttnn.generic_op` in this fold allocates nothing:
it writes into a buffer produced by `ttnn.allocate_tensor_on_device`. `itemize`'s consumer set
therefore cannot tell a pre-allocated buffer's **writer** from its **readers**. In
`cap_TriangleMultiplication__1x512x512x128,1x512x512`, buffer 21 is 268.435 MB with consumers
{`generic_op` 3, `generic_op` 5, `generic_op` 7}: op 3 writes it, ops 5 and 7 read it, and nothing
in the capture says so. Charge every consumer a read and the block is 2550.9 MB; charge none and it
is the published 1745.6 MB — **1.46x apart on one block**, 2.9449 TB vs 3.6930 TB on the fold. The
truth is inside that range and no instrument in this campaign can currently decide where. Anything
that turns a trimul byte count into a lever ranking is resting on the low end of a 1.46x bracket.

And the disagreement the brief asks about is wider than two numbers. The campaign now carries three
byte totals for the same 512 aa fold, on three files, none of which mentions the others:

| TB | instrument | floor at 424.7 GB/s |
|---:|---|---:|
| 2.9449 | `real_traffic.py`, the tip | 6.934 s |
| 3.4050 | `baseline_attrib.py:Census.charge`, round 1 | 8.018 s |
| 4.0106 | `ROOF_DEFICIT.md` re-run under the RANGE rule | 9.443 s |

### Two published prizes still on the superseded denominator

Round 1 corrected the block census used by the fusion rows from 7416.9 MB to 8046.8 MB, an 8.49 %
undercount from `split_io`. Two branches merged **after** that correction landed still quote the old
denominator:

- `state/roof-fuse-gate-epilogue.md:63` — 268.4 MB, "3.62 % of the block's 7416.9 MB". It is
  **3.34 %** of 8046.8 MB.
- `state/roof-fuse-qkv-sdpa.md:87` — 402.7 MB per block, "5.43 % of the block's 7416.9 MB". It is
  **5.00 %** of 8046.8 MB.

Neither restatement changes a verdict. Both are in the campaign's ranking inputs.

---

## What the campaign should do with this

1. The pairformer block is rank 1 on every model tried. Below it, stop ranking off this table until
   the per-unit times come from a benchlocked run.
2. The floor of record is **7.036-8.194 s**, not 6.934 s, and the headroom is at most 9.146 s.
3. `roof-msa-ladder`'s 1.223 s is worth exactly that at a 35-row alignment and exactly zero at 1024
   rows or deeper. Keep the lever, drop the fold-second from the ranking.
4. Re-measure the 512 aa cell at the deployed MSA depth before any number from it is published as
   what a user gets. The repo already has the alignment.
5. Pick one byte counter. Three files, 1.36x apart, none of them aware of the others.

## Reproducing

    git archive origin/wk/roof-budget perf/roof_budget | tar -x --strip-components=2 \
        -C perf --one-top-level=_rb_tmp
    python3 perf/roof_redteam2/scoreboard.py      # attacks 1, 2, 3, 4 and the byte totals
    python3 perf/roof_redteam2/oplevel_floor.py   # attack 3 on its own
    python3 perf/roof_redteam2/l1_output_gap.py   # attack 5's byte correction

`perf/_rb_tmp` is gitignored: it is `origin/wk/roof-budget`'s tree, not this branch's.
