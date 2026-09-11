## What a 512 aa fold actually moves, measured on tt-quietbox2 card 2 (p300c)

One real fold, every `PairformerLayer` and diffusion `DiffusionTransformerLayer` call
timed with a device sync on both sides, one settled call of each shape captured with
`ttnn.graph` and its DRAM traffic summed (`perf/bioir_roofline/fold_bytes_512.py`).
Call counts are what the fold actually issued, and they reproduce the architecture
exactly: 264 = 64 trunk blocks x 4 passes + 8 confidence, 16 = 4 MSA blocks x 4 passes,
24 and 6 layers per sampling step.

| phase | calls/fold | ms/call | DRAM GB/call | GB/s achieved | % of 430 GB/s roof | TFLOP/s | % of 86.0 TFLOP/s roof |
|---|---|---|---|---|---|---|---|
| atom transformer layer | 1200 | 1.770 | 0.517 | 292 | 68 % | 2.6 | 3.0 % |
| diffusion token layer | 4800 | 0.908 | 0.214 | 236 | 55 % | 13.5 | 15.7 % |
| pairformer block (trunk + confidence) | 264 | 41.633 | 12.170 | 292 | 68 % | 12.1 | 14.0 % |
| MSA block | 16 | 39.061 | 11.809 | 302 | 70 % | 15.2 | 17.7 % |
| **instrumented total** | | **18.10 s** | **5.05 TB** | **279** | **65 %** | **11.4** | **13.3 %** |

The instrumented phases are **18.10 s of the 23.504 s published cell** (78 % of its
23.12 s of device time) and carry **100.0 % of the fold's FLOPs**. The capture fold's own
uninstrumented remainder is **5.42 s** (embedder, templates, recycling glue, confidence
heads, diffusion conditioning), so the instrumented phases plus that remainder
predict a **23.52 s** production fold against the **23.504 s** cell, **0.1 % apart**: the
per-call syncs did not dominate and the per-shape scaling is sound.

Charging the uninstrumented remainder of the fold the same achieved rate: **6.45 TB of
DRAM traffic per fold**, against a **20.38 TB** unfused count and a **110.0 GB** fully
resident bound. So this implementation already moves **3.2x less** than an unfused one
and still **59x more** than the compulsory minimum.
