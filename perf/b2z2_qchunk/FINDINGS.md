# `TT_BIO_SDPA_GRID_Q_CHUNK` on Blackhole, isolated

qb2 card 2, p300c Blackhole, 11x10 = 110 cores, main's atom path (`TT_BIO_ATOM_L1` does not
exist in this tree), Boltz-2 cdk2x2 512 aa, 200 sampling steps, 3 recycles, seed 0, benchlocked.
Raw draws: `census_ops_bh_serial_artifact.json` (census + the discarded serial op phase),
`ops_bh.json` (interleaved op A/B), `fold_bh.json` (both, plus 28 timed folds).

## The sign is single and positive, because only one site moves

| site | q x k | work | d | calls/fold | shipped -> rule | units / 110 cores | op ratio | `torch.equal` |
|---|---|---|---|---|---|---|---|---|
| `atom` | 32 x 128 | 560 | 32 | 1200 | 32 -> **32 (declined)** | 560 (5.1 passes) | 0.9902x / 1.0060x | True |
| `token_dit` | 512 x 512 | 16 | 64 | 4800 | 256 -> **128** | 64 (one pass) | **1.1269x / 1.1293x** | True |

Two independent sessions, 50 paired reps each. The atom site **cannot move**: its 32 query rows
are one tile, so `_grid_q_chunk` returns the shipped ceiling and both arms get a byte-identical
program config. 1200 of 1200 calls declined, in both sessions.

That refutes the mechanism MERGE-LADDER 0-C attributes the 0.868x sampler regression to. There is
no second chunk for the sign to flip on: the rule touches the token DiT SDPA and nothing else in
this fold. For the lever to cost a sampler stage 15.2 %, the token DiT SDPA would have to get
~2.4x slower; it is 1.13x faster, bit-exact, twice.

## The instrument's own artifact, caught by the declining site

The first op phase timed 25 base reps and then 25 ship reps. The **atom** row -- identical
configs, zero possible effect -- read **1.3622x**, and its first-arm per-call time was 715.8 us
against 77.8 us once interleaved. That is first-arm compile and warm-up, and it is 3x the size of
the effect this row exists to measure. Interleaving base/ship/ship/base per rep and taking the
median of adjacent paired ratios removes it: the same control then reads 0.9902x and 1.0060x.

A declining call site is the cheapest A/A control available in an occupancy study, because it is
identical **by construction** rather than by assumption.

## The fold: real, and 0.36 %

28 timed folds, `base ship ship base` per rep x 7, cold discarded, loadavg 1.12-2.90.

    paired median ratio   1.00355x   95 % CI [1.00180, 1.00506]
    A/A floor (matched)             [0.99934, 1.00147]   13 same-arm adjacent pairs
    medians               base 19.779 s   ship 19.699 s
    bit-exact             ONE sha256 across all 28 folds, plDDT 0.800144 on every one

It clears the floor, and the op census predicted it before the folds ran:
4800 calls x 13.58 us = **0.065 s** of a 19.78 s fold = **1.0033x** predicted, **1.00355x**
measured. The per-call delta times the call count is the whole effect; nothing else moves.

## What that does to the published 1.01831x

24 token-DiT SDPA calls per sampling step x 200 steps = the 4800 counted. At 13.58 us saved per
call the step saves **0.326 ms**. The published claim is 41.5915 -> 40.8436 ms, **0.748 ms**, on a
branch carrying `TT_BIO_ATOM_L1` and taken from a truncated precursor fold. On Blackhole, on
main's path, this lever is worth **1.0079x on the diffusion step**, not 1.01831x. The Wormhole op
ratio was 1.3058x against Blackhole's 1.1269x, which is the whole difference: same rule, same
pick, less of a win on the part with more cores.
