# The device term's ceiling, priced against a measured frontier — 2026-09-17, c12-orchestrator

`device_ceiling.py` against `origin/wk/c10-fold-census:.../sweep2/replay.json`, 70 keys, clock
1350 MHz (pinned and during-sampled in the session that produced the census).

    ceiling  3.6766 s  /  4,963.4 Mcycles  =  193.7 % of the 1.898 s the device term owes for 10.0 s

Per class, fold seconds -> achievable seconds -> prize:

| class | keys | fold s | achievable s | prize s | owner |
|---|---:|---:|---:|---:|---|
| `linear`       | 27 | 4.6604 | 2.5155 | **2.1449** | `c12-kblock-unlock` |
| `matmul`       |  6 | 1.4794 | 0.5594 | **0.9200** | `c12-matmul-key-attribution` |
| `layer_norm_w` |  9 | 1.3570 | 0.8428 | **0.5142** | unowned |
| `layer_norm`   |  2 | 0.1756 | 0.1320 | 0.0436 | — |
| `multiply_`    | 11 | 1.7511 | 1.7297 | 0.0214 | at the roof, byte deletion only |
| `multiply`     |  4 | 0.1256 | 0.1097 | 0.0159 | — |
| `add`          |  4 | 0.1403 | 0.1246 | 0.0157 | — |
| `add_`         |  7 | 0.8473 | 0.8467 | **0.0006** | at the roof |

Top keys:

| key | fold s | achievable s | prize s | Mc | rate -> frontier | binds |
|---|---:|---:|---:|---:|---|---|
| `matmul\|1x128x512x512\|K=512` | 1.0724 | 0.2931 | 0.7794 | 1052.1 | 17.94 -> 65.66 TFLOP/s | flops |
| `linear\|1x16x512x512\|K=128` | 1.1671 | 0.4729 | 0.6943 | 937.2 | 163.0 -> 402.3 GB/s | bytes |
| `linear\|1x16x512x128\|K=512` | 0.7920 | 0.2364 | 0.5556 | 750.1 | 120.1 -> 402.3 GB/s | bytes |
| `linear\|1x512x512x128\|K=128` | 0.5276 | 0.1702 | 0.3574 | 482.5 | 142.5 -> 441.6 GB/s | bytes |
| `layer_norm_w\|1x512x512x128` | 0.6060 | 0.4206 | 0.1854 | 250.3 | 306.5 -> 441.6 GB/s | bytes |
| `linear\|512x512x128\|K=1024` | 0.1578 | 0.0217 | 0.1361 | 183.7 | 61.3 -> 444.9 GB/s | bytes |
| `layer_norm_w\|1x16x512x128` | 0.2345 | 0.0985 | 0.1360 | 183.6 | 160.3 -> 381.8 GB/s | bytes |
| `linear\|1024x512x32\|K=64` | 0.1752 | 0.0656 | 0.1095 | 147.9 | 165.5 -> 441.6 GB/s | bytes |
| `matmul\|1024x32x512\|K=512` | 0.2805 | 0.1738 | 0.1067 | 144.1 | 275.6 -> 444.9 GB/s | bytes |

Concentration: the top 3 keys are 2.0293 s (106.9 % of what is owed), the top 8 are 2.9537 s
(80.3 % of the ceiling), and **48 of 70 keys are worth under 0.010 s each**. Twelve keys clear the
campaign's 0.055 s fold A/A floor. So a 10.0 s route is a three-to-five-key problem or it does not
exist.

## What this number is, and what it is not

The envelope is `frontier_bytes(B) = max{GB/s of any key with bytes-per-call <= B}` and
`frontier_flops(F) = max{TFLOP/s of any key with FLOPs-per-call <= F}`, both monotone, both built
only from points this chip produced in that session. A key's achievable time is the larger of its
byte time and its FLOP time on those envelopes.

**It is shape-blind, and that makes it loose.** It says a key moving B bytes per call could reach the
best rate some key with an equal or smaller transfer reached; it does not know about aspect ratio, K
thinness, tile masking or operand placement, any of which can make that rate unreachable. The
biggest entry is the clearest example: `matmul|1x128x512x512|K=512` is priced against 65.66 TFLOP/s
because a key with fewer FLOPs per call hit that, and a 512-row matmul plausibly cannot. So 3.6766 s
is an **upper bound on an upper bound** — the right use is to rule routes out, not to promise them.

Controls, all asserted in the script:
- best-arm fold seconds sum to **10.5365 s** against the census's 10.5368 s of record;
- worst-arm sum is **17.2487 s**, which exceeds the whole 10.898 s device term, so the fold
  demonstrably realises approximately the better arm and per-key arm ambiguity cannot hide a large
  prize in the aggregate;
- byte total reproduces at **2,451.2 GB**;
- the four roof rows price to **0.0-0.44 % of their own time** (they define their own frontier
  points, so their prize must be session drift and nothing more).
