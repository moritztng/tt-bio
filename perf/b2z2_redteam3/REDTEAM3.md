# The closing statement, re-derived

`b2z2-redteam-v3`. Host only, no device, no fold. Every input is a committed artifact copied into
`src/` from the branch that produced it, so `redteam3.py` runs without fetching nine other branches.

    python3 perf/b2z2_redteam3/redteam3.py --json perf/b2z2_redteam3/out/redteam3.json

Full write-up and the scored pre-registration: `~/.coworker/state/b2z2-redteam-v3.md` and
`~/.coworker/state/b2z2/FINDINGS.md`.

## What reproduces

1.12862x paired and 1.12290x global to five decimals. The byte target: 27.7 % cut needed, 5.17 %
available, block floor 1.7641x rising to 1.8602x. The atom-axis fits at R² 0.998603 and 0.998765.
The gather defect at max abs 4.21875 over 13 windows, first at 127.

## What does not

**The both-shards composition.** `ceiling_v3.py` prices the sampler shard on a Wormhole step census
(atom track 12.340 of 40.366 ms) and applies it to Blackhole sampler seconds. A Blackhole census of
the same 1066 programs exists: 6.0538 ms of 22.0152 ms of kernel, 22.93 % of the 26.402 ms wall,
because 16.62 % of that wall is exposed dispatch a mesh shard cannot divide. Its `split_stack()`
brackets the trunk's share of the stack saving between 0.0000 s and 0.4758 s; the headline session's
own per-fold `block_s` column measures **0.4542 s**, so the bracket's high end is refuted.

| route | published | corrected | + the WH→BH calibration the file prints and never applies |
|---|---|---|---|
| trunk cap, measured link | 1.633 - 1.670x | 1.6341x | 1.6004x |
| trunk cap, free link | 1.83x | 1.7747x | 1.7316x |
| both shards at caps | **1.789 - 1.822x** | **1.7431x** | **1.7021x** = 11.82 s |

**The A/A floor**, corrected once already from 1.01162x to 1.02860x, is still understated. The null
draws ten pairs without replacement from a pool of 25 and gives them ten independent numerators; the
headline has five, each used twice, correlated at rho = +0.645. Structure-matched: **1.03547x**. No
verdict changes — the headline clears it 3.63x rather than 4.50x.

**The sampler term of the fold decomposition** is 5.158 s, not 5.365 s: the published figure is a
stage-wall number mixed with a device-span trunk. Cross-checked at 26.4020 ms x 200 against CONTEXT's
26.400 ms.

**The competitor denominator** has not moved since 2026-08-12 while the Tenstorrent cell was
re-measured five times in the same month.

## What holds

The trunk is closed, by one leg — the byte census is a bound and the other four directions are
failures to find a lever. Units checked on both sides, decimal, no slip. The atom axis shards,
bit-exact at both widths with both controls firing. And `TT_BIO_ATOM_KEY_WINDOW`'s ratios stand for a
better reason than the closing statement gives: both gather implementations emit the same
`[1, 140, 128, 128]` and the defective one is bit-exact over 140 real windows, so it computes a full
gather and is not faster for doing less of the model's work.
