# narrow-q is live on Blackhole at 896 aa, and the timing is still owed

`TT_BIO_TRIATT_NARROW_Q_FALLBACK` was measured on a **Galaxy Wormhole 8x9**. The open question was
whether the mechanism transfers to a p300c, because the thing the lever depends on is an L1
refusal, and L1 budget moves with the core count. It does transfer. Counts, not times, so none of
this needed a quiet box.

## The mechanism, stated so it can be falsified

Only q_chunks that DIVIDE the padded length are offered to the fused path — three call sites
filter the ladder that way (`tenstorrent.py:1986`, `:2113`, `:2458`). The production chunk is the
one entry that need not divide, so at a non-dividing length it is filtered out and the fused path
has only the WIDE entries to try. If L1 refuses all of them, `TRIATT_PERSISTENT_MASK` serves
nothing and the fold falls back to the stock op. The lever adds the dividing chunks BELOW the
production one, which is what keeps a fused rung available.

## The ladders, executed on the Blackhole grid rather than read

    896 aa (padded 896)
      OFF  full (896, 448, 256)                    offered to fused: (896, 448)          2 candidates
      ON   full (896, 448, 224, 128, 64, 32, 256)  offered to fused: (896, 448, 224, 128, 64, 32)   6

    1024 aa (padded 1024)   -- the built-in negative control, prod divides
      OFF  offered: (1024, 512, 256)      ON  offered: (1024, 512, 256)    identical

## The census, measured on this part

`perf/land_standing/out/.census-rf3-896-narrowq-on/dumps/pid638187.json`, rf3 at 896 aa,
**grid 11x10**, narrow-q **ON**:

    TRIATT_PERSISTENT_MASK   served 1088   declined 2176   rejects {pm_over_l1: 2174, l1_budget: 2}

2176 is exactly **2 declines per served call**. So per call the first two candidates are refused by
L1 and the third serves. The first two are 896 and 448 — the only two the OFF ladder has.

## What follows

With the lever OFF at 896 aa on Blackhole the fused path has exactly the two candidates the census
shows are refused, and nothing after them, so it serves **0 of 1088**. That is the same 0/1088 the
Galaxy measured. The lever is not Wormhole-specific.

**Owed, and honest about it: the OFF census is INFERRED here, not measured.** The inference is
tight — an executed pure function for the candidate list, a measured refusal count for the
refusals — but one OFF fold at 896 aa settles it directly, and it is load-insensitive, so it does
not need a quiet box. What DOES need a quiet box is the timing, which is a separate and still
entirely open question: firing is not a win. See
`perf/land_standing/narrowq_contention_attrib.py` for why the existing 896 aa timing cannot
resolve it (9.707 % A/A floor from a cold-subprocess-per-leg harness).

## Correction to a claim carried in this row's brief

The brief says the lever is inert at "exactly the multiples of 256 — 256, 320, 384, 512, 768,
1024, 1280". The list is right and the rule is wrong: 320 and 384 are not multiples of 256. They
are inert because the production chunk is **64** at those lengths and 64 divides them. The correct
rule is **inert exactly where the production q_chunk divides the padded length**. Executed over 28
tile-aligned lengths from 256 to 1536, the lever changes the ladder at 20 and is inert at 8:
256, 320, 384, 512, 768, 1024, 1280, 1536.
