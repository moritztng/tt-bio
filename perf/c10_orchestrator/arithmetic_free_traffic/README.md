# Half the fold's bytes move through ops that do none of its arithmetic

>  **MEASURED 2026-09-17 — the direction was right and the cost was underpriced by 1.64–1.99×.**
>
> `c10-fold-census` measured the three classes named below on qb2 at a during-sampled 1350 MHz:
> `multiply_`, `layer_norm` and `add_` move **1.432 TB in 4.1310 s / 5,576.9 Mcycles at
> 346.7 GB/s**, which is **78.3 % of the measured DRAM roof**, with the large pair keys **at** it —
> `add_` on `1x512x512x128` reads **444.7 GB/s, 100.40 % of roof**.
>
> The bracket below prices them at **2.074–2.523 s**, so it was **1.64–1.99× too low**. In the
> census's words, "the 22 % spread between those two roofs was not the error that mattered" — I
> treated a 22 % disagreement between two in-house roofs as the uncertainty when the real error was
> nearly 2×, and both roofs underpriced the same way.
>
> The finding itself stands and is now measured rather than modelled: this traffic is real, it is
> the fold's binding constraint, and it is larger than this note claimed.

`c10-trace-lever` killed the ladder's largest item and [`../floor_vs_measured/`](../floor_vs_measured/)
showed the clock-immune term `F` = 3.9830 s is not host overhead. That leaves bytes as the only
known lever against 27 % of the fold — and the corpus's byte work was aimed at the compute ops
(bfp8, matmul fusion). This asks where the bytes actually are.

**The core number needs no roof.** Call counts and byte counts are graph facts, and the campaign's
two independent capture artifacts agree on the fold's call total exactly and on its byte total to 3
parts per million. Only turning bytes into seconds needs a roof, and there the two disagree by
22 %, so cost is a bracket and never a point.

## The fact

| | calls | share of calls | bytes | share of bytes |
|---|---|---|---|---|
| zero-arithmetic, moves bytes | 201,744 | 43.3 % | 1.401 TB | **49.0 %** |
| zero-arithmetic, moves nothing | 149,760 | 32.2 % | 0 | 0 % |

Those ops perform **0.049 %** of the fold's 219.2 TFLOP. Half the traffic, none of the maths.

The second row is bookkeeping — `deallocate` alone is 122,112 calls — and both instruments price it
at exactly zero. **That zero is now measured rather than assumed**: `c10-trace-lever` deleted 76 % of
the fold's `ttnn.deallocate` calls, 101,559 → 24,413, and the fold did not get shorter (−0.0214 s,
inside a 0.055 s A/A floor). ttnn dispatch is asynchronous and the host stays ahead of the device, so
these calls were never on the critical path.

## Where it is concentrated, and it is not a long tail

| op | calls | bytes | share of fold bytes |
|---|---|---|---|
| `ttnn.multiply_` | 42,720 | 0.327 TB | 11.4 % |
| `ttnn.layer_norm` | 35,304 | 0.282 TB | 9.9 % |
| `ttnn.add_` | 14,616 | 0.272 TB | 9.5 % |
| `ttnn.permute` | 12,432 | 0.109 TB | 3.8 % |

**Three op classes move 0.881 TB — 30.8 % of the fold's entire byte traffic — and compute
essentially nothing.** They are in-place elementwise and normalisation over the pair and single
representations: pure DRAM round trips, exactly the shape a producer's epilogue can absorb.

That traffic is 2.07–2.52 s depending on which roof you believe. **The realistic prize is a third of
it, 0.69–0.84 s**, because this project has already measured that this class of fusion returns about
a third of what it deletes rather than all of it. A control fails if that third is ever quietly
promoted to the full figure.

For scale: the ladder's entire priced total is 0.21 s against a 4.881 s gap.

## The census under-counts its own bytes, and the correction goes our way

The census carries an internal identity: `B == calls × (in_tiles + out_tiles) × 2048`. Across the
**50 recorded shapes that have both tiles and bytes it holds at a median of exactly 1.000.** Seven
shapes record nonzero tiles against exactly zero bytes.

Five of those are semantics rather than a hole — an allocation moves nothing, and a metadata-only
`reshape` or `unsqueeze` moves nothing on device. Two are not:

| shape | calls | tiles/call | bytes the identity implies |
|---|---|---|---|
| `ttnn.multiply_` on `1x16x512x512` | 8,960 | 8,192 | 150.3 GB |
| `ttnn.layer_norm` on `1x140x32x128` | 2,400 | 1,120 | 5.5 GB |

An in-place multiply over the 16-head pair tensor must read two operands and write one, and the
same op class records bytes correctly at other shapes. Together that is **155.8 GB, 5.45 % of the
fold's recorded traffic**, and **both are in-place elementwise or normalisation ops**, so both land
in the class this directory is about. Correcting them moves the headline from 49.0 % to **51.6 %**
of bytes and the three-class concentration from 30.8 % to **34.4 %** — the published figures are
the conservative ones.

This is the **third** byte-counting defect the campaign has found, after `c10-roofline-reset`'s
known-answer overcount of exactly one output tensor (1.3333x) and the recorded dedupe-on-tensor-id
error. Any row that counts bytes should run this identity against its own capture first; it is two
lines and it caught this one.

## Why the cost is a bracket

The two in-house roofs put this class at 3.299 s and 4.012 s. The measured `F` = 3.9830 ± 0.1181 s
sits inside that, which is consistent with `F` being arithmetic-free data movement — and is not
proof, because two roofs 22 % apart will bracket a lot of things.

## What is new, and what is not

New: the axis is against `F`, so it does not compete with arithmetic and does not shrink when the
clock rises — which is exactly why it was mis-ranked while every ratio in the corpus was a wall-time
ratio at an unrecorded clock. New: the concentration is three classes, not the 60-shape tail
[`../shape_rank/`](../shape_rank/) found. New: the framing survives both roofs being wrong.

Not new: the byte axis itself. [`../floor_mix/`](../floor_mix/) already capped it at 1.487x of the
floor with an unknown part consumed by shipped levers. This is a sharper target on a known axis.

**Nothing here is a lever or a measurement of one.** Layout traffic cannot all be deleted — tensors
genuinely need different layouts, and an in-place add exists because something has to add. Sizing an
axis is not finding a lever. And "zero arithmetic" is a *model* statement: a `layer_norm` does real
SFPU work the census does not price.

    python3 arithmetic_free_traffic.py                      # writes the JSON
    python3 -m pytest test_arithmetic_free_traffic.py -q    # 18 controls
