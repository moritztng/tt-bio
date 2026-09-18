# A Blackhole core-count sweep, and what it says about the flat per-call cost

Nobody had measured one. The lever ledger's "per-class grid sizing" entry, 810 Mcycles, came from
a Wormhole 8x9 sweep (`perf/roof_tri_arith/tri_close_whglx_wh.json`, host `j10glx02`, loadavg 9.85,
no recorded clock) on which the triangle product wanted 32 cores at 2.10x and the SDPA wanted all
72 at 1.36x the other way. Opposite signs in one session, so the sign is per class, and it does not
transfer across architectures either. The parent withdrew the number. This measures the curve.

Measured on qb2 node 0, Blackhole p300c, 11x10 = 110 cores, AICLK forced to 1350 MHz and sampled
during every timed region. 360 scored rows, **zero dropped for clock**, 45,694 clock samples.

## The answer: every class is already at its optimum on the full grid

| arm | 110 | 72 | 64 | 48 | 32 | 24 | 16 | optimum |
|---|---|---|---|---|---|---|---|---|
| `cube` 4096³ (control) | **1105.71** | 1720.44 | 1774.17 | 2367.67 | 3417.41 | 4868.55 | 6761.92 | 110 |
| `mm_768_1536` | **22.41** | 24.04 | 24.04 | 30.82 | 37.18 | 46.77 | 66.21 | 110 |
| `mm_768_768` | 14.59 | **14.58** | 14.59 | 17.98 | 21.45 | 25.09 | 35.01 | 72 (1.0001x) |
| `mm_1536_768` | **23.78** | 23.80 | 23.79 | 30.51 | 37.56 | 45.75 | 65.38 | 110 |
| `mm_768_3072` | **36.78** | 41.83 | 44.83 | 57.92 | 70.92 | 88.95 | 129.55 | 110 |
| `mm_pair_qk` | **65.07** | 70.76 | 68.57 | 74.22 | 84.24 | 87.23 | 104.02 | 110 |
| `mm_pair_av` | 87.49 | 88.34 | 86.85 | 84.81 | **84.25** | 87.57 | 93.66 | 32 (1.0384x) |
| `trimul` unit | **6189.71** | 6974.51 | 7463.32 | 7753.50 | 8960.78 | 10640.12 | 13526.26 | 110 |
| `triatt` unit | **3768.97** | 4786.56 | 5315.34 | 5876.86 | 7974.63 | 10275.31 | 14674.53 | 110 |

Microseconds per call, minimum over five interleaved reps. The A/A pair (11x10 measured a second
time, late in the ladder) reads 0.004 % to 0.606 % across the nine arms.

**The Wormhole transfer has the wrong sign on Blackhole.** Applying its 32-core optimum to both tri
classes would cost **3.684 s / 4973 Mcycles** of the 14.881 s fold; even 72 cores costs
**0.952 s / 1285 Mcycles**. There is no per-class grid win here to take.

## The mechanism: per-program cost, not occupancy

Fit `t(c) = A/c + B` over each ladder. `B` is the part of the per-call cost that no number of cores
removes.

| arm | A (µs·cores) | B fixed µs | r² | B as % of the 110-core call | what all 110 cores are buying |
|---|---|---|---|---|---|
| `cube` (control) | 107244 | 159.84 | 0.9965 | 14.5 % | 945.9 µs |
| `mm_768_1536` | 839.7 | 12.55 | 0.9916 | 56.0 % | 7.6 µs |
| `mm_768_768` | 397.0 | 9.42 | 0.9849 | 64.6 % | 3.6 µs |
| `mm_1536_768` | 812.3 | 13.18 | 0.9861 | 55.4 % | 7.4 µs |
| `mm_768_3072` | 1739.4 | 18.84 | 0.9954 | 51.2 % | 15.8 µs |
| `mm_pair_qk` | 714.8 | 59.26 | 0.9850 | 91.1 % | 6.5 µs |
| `mm_pair_av` | 101.8 | 84.73 | 0.3915 | 96.9 % | 0.9 µs |
| `trimul` | 133982 | 5057.31 | 0.9947 | 81.7 % | 1218 µs |
| `triatt` | 202240 | 1894.97 | 0.9976 | 50.3 % | 1838 µs |

`percall_residual/` asked which of two mechanisms produces a flat ~22 µs per-call cost. The fit
answers it: on the 768-family linears **9.4 to 18.8 µs of the per-call time is core-independent**,
and an infinite grid would remove only the 3.6 to 15.8 µs that is left. The residual is
**per-program cost, not grid occupancy**. The cheaper explanation is eliminated and the campaign's
per-call question points at program setup.

The 16-head pair matmuls are the extreme case: 91 % and 97 % core-independent, and `mm_pair_av`'s
curve is flat enough that the fit barely holds (r² 0.39). They run at 16.50 and 12.27 TFLOP/s, well
under the 43 TFLOP/s their arithmetic intensity was said to cap them at, and adding cores does
nothing for them.

**The control works.** The dense cube scales 6.115x from 16 to 110 cores against an ideal 6.88x,
89 % occupancy efficiency, r² 0.9965. An instrument that can see that and still reads the fold's
shapes flat is reading the shapes, not itself.

## Host issue is not the limiter, but it is close

Every region also measures the Python-side cost of issuing one call into an empty queue. It is
**6.49 to 7.47 µs for every matmul arm and independent of grid** (258 µs and 204 µs for the two
composite tri units, which issue many ops each). That is below the measured wall time everywhere,
so the device is the limiter in this harness. It is within 1.4x of `mm_768_768`'s 9.42 µs fixed
term, which is worth knowing before anyone reads the fixed term as purely a device property.

## This session's own rates at 1350 MHz

Dense 4096³ cube **124.3 TFLOP/s**. Triangle attention unit 30.20, triangle multiplication 13.88,
the 768-family linears 41.4 to 65.7, the 16-head pair matmuls 12.3 and 16.5. Board power peaked at
239 W on the cube and 64 to 151 W on the rest. The retired 67.59 TFLOP/s cube figure was taken at
loadavg 4.6 to 6.2 with no recorded clock and is not used here.

## Reproduce

    export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH PYTHONPATH=$PWD
    unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
    /home/ttuser/.coworker/scripts/benchlock.sh c10-grid-sweep -- \
      env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10-grid-sweep \
      python3 perf/c10_grid_sweep/grid_sweep.py --reps 5 --warm 2 --region-ms 120
    python3 perf/c10_grid_sweep/reduce.py          # CPU only, writes out/curves.json

`probe.py` is the preflight (device grid, ARC clock response, `core_grid` acceptance per shape).
`PREREG.md` holds the predictions and kill criteria, committed before the first scored row.

## Limits

The two tri arms sweep the grid through `TT_BIO_FORCE_GRID`, which also moves the grid-derived
tuning thresholds the shipped code applies, so their curve is "this class at that grid as the
engine would actually run it" rather than a pure core count. Per-call time includes one
`ttnn.deallocate` per call, the same on every grid. The matmul arms run back to back in isolation,
which is the best case for a shape; the fold interleaves them with everything else. Six matmul
shapes cover 68,496 of the class's 108,608 calls at 512 aa; the other shapes were not swept. The
fold-level conversions use the 512 aa call census and assume every call of a shape could take the
grid in question, which makes them upper bounds.

No production code changed.
