# The byte axis is worth 1.487x, and it lives in the elementwise ops

The roof campaign's floor of record for 512 aa is **12.706 s over 465,664 top-level ttnn calls**,
`max(traffic, arithmetic)` per call. Summed across the fold: arithmetic 8.543 s, traffic 6.732 s,
floor 12.706 s. Those do not add up to the floor and they are not supposed to — the max is taken
per call, so a class is not "bound by" one term and comparing its two columns says nothing.

What an aggregate can honestly answer is a ceiling:

| if you could | the floor becomes | you win | ratio |
|---|---|---|---|
| delete every byte the fold moves | 8.543 s | 4.164 s, 32.8 % | **1.487x** |
| make arithmetic free | 6.732 s | 5.975 s, 47.0 % | 1.888x |

**1.487x independently reproduces `c10-lever-corpus`'s byte-axis ceiling of 1.470x**, which it
derived from the campaign prose rather than from this census. Two unrelated routes to the same
number is about as good as this corpus gets.

## Where the byte axis actually is

Ranked by what deleting all of a class's bytes could win — which is *not* the same as the biggest
classes:

| class | byte headroom | floor | arithmetic | traffic | calls |
|---|---|---|---|---|---|
| `ttnn.multiply_` | 0.770 s | 0.770 | 0.000 | 0.770 | 42,720 |
| `ttnn.generic_op` | 0.708 s | 4.084 | 3.376 | 2.149 | 3,920 |
| `ttnn.layer_norm` | 0.665 s | 0.665 | 0.000 | 0.665 | 35,304 |
| `ttnn.add_` | 0.640 s | 0.640 | 0.000 | 0.640 | 14,616 |
| `ttnn.permute` | 0.256 s | 0.256 | 0.000 | 0.256 | 12,432 |
| `ttnn.linear` | 0.140 s | 3.805 | 3.665 | 0.955 | 108,608 |

`ttnn.linear` is the single largest call class in the fold, 108,608 calls and 3.805 s of floor, and
the entire byte axis inside it is worth **0.140 s** because it is arithmetic-bound. The byte axis
lives in the elementwise tail — `multiply_`, `layer_norm`, `add_`, `permute` — which is exactly
where this project's 1.02-1.05x byte-deletion levers have been landing, each one shaving a slice of
the same 4.164 s.

## The caveat that matters more than the numbers

**The compute roof every arithmetic second is divided by carries no clock.**
`perf/roof_shape/shape_roofs_qb2c3_shipped.json` records 108.54 TFLOP/s on a 110-core grid on qb2
card 3 — host, arch, grid, card and loadavg, but no AICLK — and it was taken at **loadavg 6.2**.

Arithmetic seconds scale with AICLK and DRAM seconds do not, so both ceilings above are only true
at whatever clock that roof happened to run at. At burst the arithmetic side shrinks, the DRAM side
does not, and the byte axis is therefore worth **more** than 1.487x at 1350 MHz — by an amount this
artifact cannot quantify, because it does not record where it started. There is a second
complication it cannot resolve either: L1 and NOC bandwidth track the core clock while DRAM does
not, so a traffic term is AICLK-immune only to the extent that it is DRAM rather than L1/NOC.

That is the Phase 0 question, and this is as far as CPU arithmetic can take it. `c10-fold-census`
has to measure both roofs on the same chip, in the same process, at the same recorded clock.

    python3 floor_mix.py                        # prints floor_mix.json
    python3 -m pytest test_floor_mix.py -q      # 7 controls
