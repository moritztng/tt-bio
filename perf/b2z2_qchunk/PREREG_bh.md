# Pre-registration: does `TT_BIO_SDPA_GRID_Q_CHUNK` have one sign on Blackhole?

Written and committed before the first device run of this row. qb2 card 2 (p300c, 11x10),
`TT_BIO_ATOM_L1` OFF, main's atom path, Boltz-2 512 aa cell, 200 sampling steps, 3 recycles,
seed 0.

The question is not how fast it is. It is whether the sign is a property of the lever or of the
call site. Two published numbers disagree: 1.01831x on the diffusion step, measured on a branch
already carrying `TT_BIO_ATOM_L1`, and 0.868x on the sampler stage from one rep on a box that
went to loadavg 176 mid-session.

## Predictions

**P1 - the atom site does not move at all.** `_grid_q_chunk` returns the shipped ceiling
unchanged when `padded <= SDPA_CHUNK_TILE`, and Boltz-2's atom attention is windowed with 32
query rows. So the rule cannot narrow that site, and the atom-site MOVED call count is **0**.
If that holds, the 0.868x sampler reading cannot be an atom-SDPA occupancy effect, and the
mechanism story in MERGE-LADDER 0-C ("the atom SDPA's batch x heads term is large, so the chunk
it picks there is a different chunk") is **wrong about this lever** even if the sampler number
is real.

**P2 - the token DiT site moves, to q_chunk 128.** 512 aa padded 512, work = 1 x 16 heads,
110 cores: 64 units of 110 against the shipped 256's 32 units. Every token-DiT call in the fold
takes it; MOVED count equals the site's call count and DECLINED is 0.

**P3 - op-level sign on Blackhole is positive at the token DiT site,** between 1.05x and 1.25x,
bracketing the 1.1398x the p150a ladder read at 110 cores. Negative at neither site.

**P4 - the fold ratio is small and may be a null.** Predicted band 1.000x-1.020x with an A/A
floor around +/-1.5 %. A 12 us per-call saving only reaches the fold if the token DiT SDPA is a
percent-level share of a ~24 s wall; I am not assuming it is. A ratio inside the floor is
reported as "cannot distinguish from zero at n=7", not as a win.

**P5 - bit-exact on Blackhole**, `torch.equal` at the op and an identical CIF sha256 on the
fold, because `q_chunk` partitions independent query rows and the reduction order lives in
`k_chunk`, untouched.

**P6 - verdict.** Given P1, I expect SHIPS-or-NULL rather than PER-SITE: with the atom site
provably untouched there is no second site for the sign to flip on in the Boltz-2 sampler. If
the fold nevertheless reads a regression outside its floor, the cause is not this rule's chunk
pick at the atom SDPA and I have to name what it is.

## What would falsify each

P1: any census row with `site=atom` and `chunk != shipped`. P2: a token_dit row that declined,
or a call count of zero (the prior row could not read this counter at all - it censused the CLI
parent, not the spawned device worker; this row drives the fold in-process). P3: an op ratio
below 1.0 at either site outside the op harness's own repeat spread. P4: nothing - the band is
wide on purpose and the floor decides. P5: a single differing byte.
