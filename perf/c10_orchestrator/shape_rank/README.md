# There is no giant hiding in the fold: 60 shapes, none over 3.6 %

The class split says where the floor is. This says which *shape*, which is what a lever has to
target, with per-call cost to separate "many small calls" from "a few large ones".

The 512 aa floor of 12.706 s spreads across 60 recorded shapes and **the top 20 hold only 31.5 %**.
No single (class, shape) holds more than 3.6 %. That is the headline: a lever here has to hit a
class across many shapes rather than one kernel. There is no one kernel to fix.

> **2026-09-17: the "remove per-call cost" half of that advice is withdrawn.** It meant *host*
> per-call cost, and `c10-trace-lever` measured that at zero — ttnn dispatch is asynchronous and the
> host was never on the critical path. The per-shape long tail below is still real; what a lever has
> to hit is a *class of device work*, and on the measured evidence that class is the N² pair
> representation. See [`../dispatch_hypothesis/`](../dispatch_hypothesis/) and
> [`../frontier/`](../frontier/).

The largest single item is worth naming anyway:

| item | calls | floor | per call |
|---|---|---|---|
| `ttnn.add_` on `1x512x512x128` (the pair representation) | 1,416 | 0.453 s | 319.6 µs |
| `ttnn.linear` `1x512x768 @ 768x1536` | 11,200 | 0.391 s | 34.9 µs |
| `ttnn.linear` `1x16x512x512 @ 512x128` | 8,448 | 0.370 s | 43.8 µs |
| `ttnn.linear` `1x16x512x128 @ 128x512` | 9,472 | 0.324 s | 34.2 µs |
| `ttnn.linear` `1x512x768 @ 768x768` | 10,200 | 0.294 s | 28.9 µs |

That first row is an **in-place add over the pair representation with no arithmetic term at all** —
a pure DRAM round trip of a 512×512×128 tensor, 1,416 times. It is exactly the shape a producer's
epilogue can absorb, and this project has measured that kind of fusion before: it returns about a
third of the cost it deletes, not all of it. So call it roughly 0.15 s, not 0.45 s, and note that
it sits inside the byte axis whose ceiling is already established at 1.487x.

Pure-traffic elementwise across every shape is 1.182 s, 9.3 % of the floor, over 117,800 calls.

## What this is for

`c10-fold-census` should start from this table and replace it with measured numbers. The
comparison is the interesting part: these are modelled `max(traffic, arithmetic)` floors divided by
a compute roof that **carries no recorded clock and was taken at loadavg 6.2**, so the arithmetic
column of every row is only valid at an unknown clock. Where the measurement disagrees with the
model, the model is the suspect.

    python3 shape_rank.py                      # prints shape_rank.json
    python3 -m pytest test_shape_rank.py -q    # 7 controls
