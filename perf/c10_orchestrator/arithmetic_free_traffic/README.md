# Half the fold's bytes move through ops that do none of its arithmetic

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
at exactly zero. `c10-trace-lever`'s null says host per-call cost is not this fold's constraint, so
that is probably right, but it is 32 % of the fold's calls resting on an assumption rather than a
measurement.

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
    python3 -m pytest test_arithmetic_free_traffic.py -q    # 13 controls
