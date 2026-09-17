# What 10.0 s would actually require

`c10-fixed-cost` measured both terms of `T = F + W/f` at a pinned, during-sampled 1350 MHz, so the
target stopped being a wall-clock wish and became a budget:

    F = 3.9830 ± 0.1181 s        W = 14665.0 ± 121.1 Mcycles = 10.8630 s at 1350 MHz
    fold = 14.8460 s             10.0 s means removing 4.8460 s

Any `(ΔF, ΔW)` with `ΔF + ΔW/1350 = 4.846 s` reaches it. Two corners bound the problem:

- **Cycles alone:** cut **6,542 Mcycles, 44.6 %** of the work term.
- **`F` alone: impossible at any completeness.** `F` would have to be **−0.86 s**. Delete every
  microsecond of host and DRAM-bound cost and the fold still reads 10.86 s.

## Everything the campaign has ever numbered comes to 0.90 s

| lever | fold s | evidence |
|---|---|---|
| ttnn trace of the diffusion loop | **0.00** | measured on Blackhole at 1350 and 1000 MHz |
| fuse the arithmetic-free elementwise traffic | 0.69 | derived: traffic × a measured ⅓ return |
| `TT_BIO_HEAD_PAD_TAIL` | 0.21 | Wormhole, no recorded clock |
| `TT_BIO_DIT_FUSED_QKV` | 0.00 | excluded — jointly 0.713 Å with the row above vs a 0.60 Å bar |
| `TT_BIO_SDPA_GRID_Q_CHUNK` sign | 0.00 | unmeasured, −0.46 to +0.21 s |
| per-class grid sizing | 0.00 | withdrawn as a cross-architecture transfer |
| size-independent share of the work term | 0.00 | **refuted** — N^1.827 ± 0.030 on 512 → 768 aa |

**0.90 s, 18.6 % of the requirement** — and that is an upper bound, because perturbations stack
strongly sub-additively on this fixture.

## The one axis with enough headroom

The fold does **219.06 TFLOP** of matmul in a modelled **11.019 s**, an achieved **19.88 TFLOP/s**.
Four independent Blackhole dense-cube measurements cluster at **104.93–114.20 TFLOP/s**. So the
fold's matmul class runs at **17.4–18.9 % of the cube**, and closing that gap entirely would take
matmul time to ~2.0 s — saving 8.9–9.1 s, which overshoots the target.

Reaching 10.0 s on this axis alone needs the class lifted from 19.88 to **35.49 TFLOP/s**, about
**53 % of the gap to the cube**.

That is the only axis the campaign has found that is big enough. It is also its least trustworthy
number: the per-shape rates behind `matmul_floor_s` were taken **on pc's custom 130-core firmware**,
not on qb2's 110-core p300c, and not all of the gap is recoverable anyway — the cube is 4096³ and
the fold's matmuls are not. **Re-measuring it on qb2 at a recorded clock is `c10-fold-census`'s
single most valuable deliverable.**

## A fourth clock artifact, found from ratios alone

Those four cube measurements agree within **8.8 %**. A fifth number, **67.59 TFLOP/s**, is quoted by
[`../shape_rank/`](../shape_rank/) and [`../grid_evidence/`](../grid_evidence/) as "the same-session
dense cube". Against the cluster maximum it implies a chip at **799 MHz** — the clock floor.

It is the campaign's founding failure mode for the fourth time, after the 17.34 s cell, the 2.309 s
prize, and the 800 MHz latch that three investigations blamed on hardware. No new measurement was
needed to find it; the ratio was enough.

**What survives from that session:** ratios taken *within* it, because both sides were throttled
together — triangle multiplication at 11.47 TFLOP/s is still 5.89x off its own session's cube. The
absolute rates do not, and at burst clock they would be about 1.69x higher.

## Status

No lever here is approved, measured as a stack, or scored for accuracy. `matmul_floor_s` is
modelled, and the achieved rate inherits that artifact's contamination. The frontier itself is exact
arithmetic on two measured terms; everything scored against it is not, and each row carries its
evidence class rather than being flattened into a total.

    python3 frontier.py                      # writes frontier.json
    python3 -m pytest test_frontier.py -q    # 13 controls
