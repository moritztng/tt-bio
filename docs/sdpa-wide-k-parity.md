# `TT_BIO_SDPA_WIDE_K` accuracy and performance envelope

Triangle attention picks its SDPA `k_chunk` by searching downward from a 256 cap. The fused
triangle-attention kernel refuses any call whose `k_chunk` does not divide the padded sequence, so
at a padded length whose 32-aligned divisors all sit *above* the cap, the kernel declines every call
and the fold falls back to the stock op reading a mask padded out again. `TT_BIO_SDPA_WIDE_K=1`
offers those wider dividing `k_chunk`s, widest first, with today's pick last.

Off by default. Turning it on changes the online-softmax reduction order, so it is **not
bit-exact**. It is gated on evidence rather than flipped silently, like `--fast`
(see [boltz2-fast-parity.md](boltz2-fast-parity.md)), but it does not inherit `--fast`s argument:
the models `--fast` was accepted on are nondeterministic at a fixed seed and this path is not.
See Accuracy below.

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

Op level, qb1 card 3 (Blackhole p150a, 13x10), batch=seq, arms interleaved, median of three blocks
of three (`perf/sdpa_widek/out_qfix/`). Measured at all three head counts the shipped models use,
since each is a different model:

| padded | Boltz-2 / BoltzGen (h=4) | Protenix-v2 (h=8) | OpenDDE (h=12) |
|--:|--:|--:|--:|
| 288 | not reachable | 2.83x | 3.01x |
| 352 | not reachable | 2.23x | not measured |
| 416 | not reachable | 3.34x | not measured |
| 704 | **3.41x** | 2.45x | 3.51x |
| 832 | **1.33x** | 1.27x | 1.32x |
| 864 | not reachable | 4.39x | 4.01x |
| 1056 | not reachable | 1.28x | not measured |
| 1088, 1216, 1472 | bit-exact | bit-exact | bit-exact |
| 1248 | not reachable | 1.01x, numerics change | not measured |
| 544, 608, 736, 928, 992, 1184, 1312, 1376, 1504 | not reachable | bit-exact | not measured |

The speedup tracks the padded length, not the model: 832 reads 1.33 / 1.27 / 1.32 across the three
head counts, 704 reads 3.41 / 2.45 / 3.51, and the numerical perturbation matches to five digits
(rmsd/std 0.017081 / 0.017083 / 0.017083 at 832). Heads are the batch dimension of this SDPA, which
is what the mechanism predicts.

**Boltz-2 and BoltzGen have the smallest blast radius of the affected models.** Their 64-multiple
token pad can only present five of the twenty firing lengths, and only two of those five gain, 704
and 832. At 1088, 1216 and 1472 the ladder falls through and the arms are `torch.equal`.

**Padded 1248 changes numerics for nothing.** It accepts q416/k416 on the stock op and reads 1.0090x,
inside the instrument's own floor. Documented rather than allow-listed out: a hard-coded length list
would be calibrated on this 13x10 grid alone, and that is how the reblock_permute lever became a
0.62x loss on the other part.

Per-length h=8 detail:

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

## Should it be the default?

Not yet, on three specific grounds rather than on caution.

**The determinism floor is zero.** Protenix-v2 reproduces bit-exactly at a fixed seed today. Flipping
this default would silently end that at seven padded lengths, so a user comparing a new run against a
stored one would see a change with no flag to explain it. A default that alters previously
reproducible output needs its own release-gate arm, not an inherited one.

**The fold-level envelope covers one of four affected model entries.** Protenix-v2 at padded 704
passed 3/3 seeds. Boltz-2, BoltzGen and OpenDDE have op-level wins and no fold-level accuracy arm.
Their op numbers are strong (Boltz-2/BoltzGen 3.41x at 704) and the mechanism is shown to be
head-count independent, but "the mechanism generalises" is not the same evidence as a fold that ran.

**One length pays without being paid.** Padded 1248 changes numerics for 1.01x. A default should not
do that anywhere, and the fix is a measurement on the other grid rather than a length list.

What would change the answer: a fold-level envelope on Boltz-2 or BoltzGen at padded 704 (their
biggest win and the smallest blast radius of any affected model, five reachable lengths of which
three are already bit-exact), the same on OpenDDE, and a Wormhole run to check the wins are not
13x10-specific. With those, default-ON is a two-length change for Boltz-2/BoltzGen and defensible.

Until then the flag is the honest surface: opt in, get 1.1x on a Protenix-v2 trunk and up to 4.4x on
the op, and accept a 0.15 A structural change you can measure and a reproducibility break you cannot
un-see.
