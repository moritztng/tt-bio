# What binds the fold's floor, and why you cannot read it at burst clock

The roof campaign's floor of record for 512 aa is 12.706 s over 465,664 top-level ttnn calls,
`max(traffic, arithmetic)` per op. Split by which term binds:

| | seconds | share |
|---|---|---|
| arithmetic-bound | 9.408 s | **74 %** |
| traffic-bound | 3.299 s | **26 %** |

and it is extremely concentrated. Three classes — `ttnn.generic_op` (the fused trimul/triatt/SDPA
kernels, 3,920 calls, 4.084 s), `ttnn.linear` (108,608 calls, 3.805 s) and `ttnn.matmul` (1,632
calls, 1.518 s) — hold **74 % of the floor from 24.5 % of the calls**. Everything else is a long
tail: `ttnn.multiply_` 0.770 s, `ttnn.layer_norm` 0.665 s, `ttnn.add_` 0.640 s, then nothing above
0.3 s.

This is independent support for `c10-lever-corpus`'s byte-axis ceiling of 1.470x. Byte deletion
acts on the traffic term, and the traffic term is a quarter of the floor.

## The part that matters for this campaign

**The compute roof every one of those arithmetic seconds is divided by carries no clock.**
`perf/roof_shape/shape_roofs_qb2c3_shipped.json` records 108.54 TFLOP/s on a 110-core grid on qb2
card 3 — host, arch, grid, card and loadavg, but no AICLK — and it was taken at **loadavg 6.2**.

That is not a footnote. An arithmetic-bound floor second scales with AICLK and a DRAM-bound one
does not, so the 74/26 split above is only true at whatever clock that roof happened to run at. At
a higher clock the arithmetic side shrinks, the DRAM side does not, and the mix moves toward
traffic — by an amount this artifact cannot tell you, because it does not record where it started.

There is a second complication the census cannot resolve either: L1 and NOC bandwidth track the
core clock while DRAM does not, so a traffic term is only AICLK-immune to the extent that it is
DRAM rather than L1/NOC, and this census does not separate them.

So the honest reading is not "the fold is 74 % arithmetic-bound at burst". It is **the existing
floor cannot be read at burst clock at all**, in either direction. That is what `c10-fold-census`
has to fix, and it is why its brief requires both roofs measured on the same chip in the same
process at the same recorded clock rather than rescaled from anything here.

    python3 floor_mix.py                        # prints floor_mix.json
    python3 -m pytest test_floor_mix.py -q      # 6 controls
