# b2z2-bh-compose-v2 — PREDICTED, written before any Blackhole number exists

Pre-registered per CONTEXT.md §5 rule 7. Commit this before the first fold runs.

## The question

`wk/b2z2-layout-op-elision` is the one bit-exact lever wave 2 produced that has a whole-fold
number, and it has never run on Blackhole. On Wormhole (whglx card 4) it reads **1.04694x on the
sampler** and **1.00926x on the whole 512 aa fold**, the latter inside its own A/A floor of
0.99107 and explicitly not its branch's claim.

## Do I expect the WH fold ratio to survive onto Blackhole?

**Partially, and I expect the fold number to land inside or within a hair of the cell's A/A
floor.** The prior is against survival and I am not fighting it.

The mechanism is arithmetic-free byte movement: the atom key gather goes from ~10,656 tile writes
per call to ~2,816, -73.6 %, over 1200 calls per fold. Deleted movement is paid back at the
architecture's datum rate, and `b2z2-datum-rate-floor` measured **20.82-30.44 ns/tile on
Blackhole against the 71.3 ns/tile the Wormhole pricing assumed**. A deleted tile pass buys
2.3-3.4x less wall on this part than on the part it was measured on. That is exactly how K2 came
in: predicted 1.015x-1.025x from a WH fraction, measured **1.00448x**, falsified low by 4x.

Against that, two things argue for better transfer than K2 got. The lever is in the **sampler**,
which `b2z2-diffusion-loop-attack` measured at **93.8 % device-bound**, so a device-side saving
is not hidden behind a host residual the way a trunk saving can be. And the replacement runs the
same ~5 programs, so the per-program 3.0 us constant is neutral: this is a pure byte deletion on
the axis that `2-CORRECTION-B` says actually correlates (bytes Spearman +0.909).

## PREDICTED numbers

| quantity | predicted | reasoning |
|---|---|---|
| ELI, BH sampler ratio | **1.015x - 1.030x** | WH 1.04694x discounted 1.5-3x for the datum rate |
| ELI, BH fold ratio | **1.004x - 1.012x** | sampler is ~5.34 s of a ~20.19 s fold, so a sampler ratio scales down by ~3.8x |
| ELI, seconds off the fold | 0.08 - 0.24 s | |
| UNION = ELI+K2+DST, fold ratio | **1.008x - 1.016x** | ELI plus K2's measured 1.00448x; DST is 0.99748x, i.e. nothing |
| union additivity discount | 0 % +/- 2 % | the three are disjoint (atom gather / tri-att projection / trimul gated move); wave 1 measured 0.17 % additive on disjoint phases and the predecessor measured -1.08 % |
| bit-exactness, composed | **holds** — one CIF sha256 across base, ELI and UNION | all three are reassociations with no rounding-point change |

**The headline call: ELI alone will NOT clear the cell's 1.00229x A/A floor by a margin a fold can
resolve at n=10 with 1.24 % base spread. The honest fold-level verdict I expect to write is
between "1.01x, marginal" and "not measurable at fold scale".** The sampler stage is the
instrument that should resolve it, because it is a smaller, quieter denominator and it is where
the lever lives.

## Falsifiers, each one able to fire

1. **ELI's fold ratio lands inside the A/A floor** -> the lever does not survive to fold scale on
   Blackhole and the composed answer for this wave is 1.00x. Predicted as the likely outcome; it
   is a falsifier of the *brief's* premise that this is the strongest candidate, not of mine.
2. **ELI's sampler ratio lands inside the sampler's own A/A floor** -> refutes me specifically.
   I am claiming the sampler resolves what the fold cannot. If the sampler is flat too, the lever
   does not transfer to Blackhole at all and the 93.8 %-device-bound argument for transfer is
   wrong.
3. **ELI exceeds 1.02x on the fold** -> the WH->BH datum-rate discount does not apply to byte
   deletions the way it applied to K2's tile-pass deletion. That would be worth understanding and
   would reopen the byte axis harder than `2-CORRECTION-B` already has.
4. **UNION's CIF sha256 differs from base while ELI's and K2's each match** -> two bit-exact
   changes reassociated into one that is not, which is the exact composition risk the brief names.
5. **UNION lands below the best single arm** -> destructive interference between levers, or a
   silently disabled lever in the composed tree.
6. **Any arm's lever counters do not move** (`ATOM_SHIFT_GATHER_STATS`, `FUSED_STATS`) -> the
   lever is not live and no number may be attributed to it.

## Not in this pass, and why

The **MSA depth ladder is excluded**. It reads 0.9765 A all-atom at 512 aa against a 0.60 A kill
line (`b2z2-bh-compose-landed`), independently confirmed by its own branch's 0.8844 A on
Wormhole. No union containing it is quoted here.
