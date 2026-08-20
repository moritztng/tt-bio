# `TT_BIO_SDPA_WIDE_K` accuracy and performance envelope

Triangle attention picks its SDPA `k_chunk` by searching downward from a 256 cap. The fused
triangle-attention kernel refuses any call whose `k_chunk` does not divide the padded sequence, so
at a padded length whose 32-aligned divisors all sit *above* the cap, the kernel declines every call
and the fold falls back to the stock op reading a mask padded out again. `TT_BIO_SDPA_WIDE_K=1`
offers those wider dividing `k_chunk`s, widest first, with today's pick last.

Off by default. Turning it on changes the online-softmax reduction order, so it is **not
bit-exact**, which is why it is gated the same way `--fast` is (see
[boltz2-fast-parity.md](boltz2-fast-parity.md)) rather than flipped silently.

## Which sizes it touches

Only padded token lengths whose shipped `k_chunk` fails to divide them. Swept exhaustively to 1536,
that is twenty lengths:

    288  352  416  544  608  704  736  832  864  928  992
    1056 1088 1184 1216 1248 1312 1376 1472 1504

At every other length the candidate list has one entry and the code path is byte for byte the
default. Reproduce the sweep with `scripts/sdpa_wide_k_census.py`, which reads the ladder out of
`tt_bio.tenstorrent` instead of restating the arithmetic.

Models that reach the fused triangle-attention SDPA at all: Boltz-2 and BoltzGen (padded to
multiples of 64, so they can present 704, 832, 1088, 1216, 1472), Protenix-v2 and OpenDDE (no token
pad, so every multiple of 32). OpenFold3 routes its four pairformer sites through fp32 softmax,
Boltz-2's affinity trunk has its own fp32 triangle attention, and ESMFold2 and RFD3 have no triangle
attention, so none of those four is affected either way.

## Performance

Op level, qb1 card 3 (Blackhole p150a, 13x10), h=8 d=32, batch=seq, arms interleaved, median of
three blocks of three (`perf/sdpa_widek/out_qfix/`):

| padded | default pick | wide-k pick | speedup |
|--:|---|---|--:|
| 288 | q288 k64 stock | q288 k288 fused | 2.83x |
| 352 | q352 k64 stock | q352 k352 fused | 2.23x |
| 416 | q416 k256 stock | q416 k416 fused | 3.34x |
| 704 | q352 k256 stock | q352 k704 fused | 2.45x |
| 832 | q416 k256 stock | q416 k416 stock | 1.27x |
| 864 | q288 k256 stock | q288 k864 fused | 4.39x |
| 544, 608, 736, 928, 992 | unchanged | unchanged | bit-exact fall-through |

Eight legs in that run are `torch.equal` between arms, so they run identical code and their spread
is the instrument's own floor: 0.955x to 1.013x. Every speedup above clears it.

At the fold the whole-run wall is the wrong denominator. It carries model load, MSA staging, feature
prep and 200 diffusion steps, none of which this lever touches, and on a busy host those stages move
further than the effect does. Measure the trunk stage instead. Protenix-v2, `examples/686.yaml`
(686 tokens, padded 704), 10 recycles, 200 sampling steps:

| | trunk stage |
|---|--:|
| default | 120.0 s |
| `TT_BIO_SDPA_WIDE_K=1` | 106.3 s |
| | **1.1285x** |

The fold serves exactly one triangle-attention shape, `686x686`, at `(352, 256, stock)` by default
and `(352, 704, fused)` with the lever on, 1208 calls per fold with zero fall-backs. The op screen
predicted 1208 x 11.05 ms = 13.35 s; the trunk moved 13.7 s, so predicted and measured agree to 2.6%.

## Accuracy

Judged against the two controls `--fast` uses. Same cell, per-chain Kabsch RMSD (single chain, so
global and per-chain coincide and no inter-chain placement enters).

Read the first row before the rest. **Protenix-v2 is bit-deterministic at a fixed seed on this
path**: two default runs at seed 0 agree to 0.0000 Å, PCC 1.000000, identical pLDDT. That is not the
case for Boltz-2 diffusion, where `--fast` was judged against a 1.6-4.7 Å determinism floor, so the
`--fast` argument does not transfer unchanged. Here the floor is zero, which means the lever is the
only source of deviation at a fixed seed and its deviation is strictly above the floor rather than
buried in it. The band that makes it acceptable is the seed-to-seed one, not the determinism one.

| comparison | RMSD (Å) | coord PCC | lDDT | ΔpLDDT |
|---|--:|--:|--:|--:|
| **determinism floor**, default s0 vs s0 rerun | **0.0000** | 1.000000 | 1.0000 | +0.000000 |
| seed spread, default s0 vs s1 | 7.28 | 0.9689 | 0.934 | +0.0005 |
| seed spread, default s0 vs s2 | 7.19 | 0.9714 | 0.900 | +0.0041 |
| seed spread, default s1 vs s2 | 3.69 | 0.9925 | 0.934 | +0.0036 |
| **lever, seed 0** | **0.113** | 0.999992 | 0.9992 | +0.000099 |
| **lever, seed 1** | **0.060** | 0.999998 | 0.9997 | +0.000101 |
| **lever, seed 2** | **0.146** | 0.999988 | 0.9977 | +0.000076 |

The worst lever leg is 25x inside the smallest of the three seed-spread controls and 50x inside the
largest, and it moves pLDDT by 0.0001 against a seed-to-seed 0.0041. What it is NOT is free: a fold
that is reproducible today stops being reproducible against its own earlier output when you set this
flag. 0.15 Å on a 686-residue chain is far below any structural interpretation, and 0.0001 pLDDT is
below the reported precision, so the change is not meaningful. It is still a change, and that is the
reason the flag is opt-in rather than the default. Reproduce with `perf/sdpa_widek/widek_fold_ab.py` (runs the
legs, asserts out of each worker process which pair it actually served) then
`perf/sdpa_widek/widek_fold_score.py`.

## Why it is still opt-in

The envelope above covers one model at one affected length. The lever is neutral by construction
everywhere it does not fire, and where it does fire it has now been measured safe on this cell, but
Boltz-2, BoltzGen and OpenDDE have not been folded under it, and 832's 1.27x is the one op-level win
close to the instrument floor. Default-ON needs those cells, not a stronger argument from these.
