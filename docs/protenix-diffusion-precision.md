# Protenix diffusion precision

`tt-bio predict --model protenix-v2 --diffusion_precision bf16` runs the diffusion module in bf16
instead of fp32. On a Wormhole chip it saves about 9 % of a fold. On most complexes the structures
match fp32 far more closely than two fp32 seeds match each other, but on complexes where the model is
unsure of the binding mode, bf16 can land a different one, and that one can rank first. The default
stays fp32, which matches the reference implementation.

## Speed

Wormhole Galaxy, one chip per fold, AICLK 1000 MHz sampled during every fold. 730-token complex
(580 + 150 aa), MSA depth 9947, 10 recycles, 200 diffusion steps, 5 samples. Warm folds.

| Setting | Fold (s) | vs default |
|---|---|---|
| default (fp32 diffusion) | 623.1 / 624.2 / 624.8 | |
| `--diffusion_precision bf16` | 568.7 / 570.1 | -8.7 % |

Where the default fold's time goes (call-path timers with device syncs, so they add up to 676 s
rather than 624 s): the trunk is 73 % (MSA module 41 %, pairformer 29 %), diffusion 21 %, confidence
5 %. bf16 only touches the diffusion share, which is why it cannot buy more than this.

## Accuracy

Each bf16 sample is compared with the fp32 sample of the same seed, so the two differ only by the
precision. The yardstick is fp32 against fp32 at a different seed, the variation a user accepts by
picking a seed. lDDT is computed on one backbone atom per residue and needs no superposition;
interface lDDT counts only the pairs between the two chains.

Four seeds, five samples each. Higher lDDT is closer; binder RMSD is chain B superposed on itself.

| Comparison | lDDT | interface lDDT | binder RMSD (A) |
|---|---|---|---|
| bf16 vs fp32, seed 101 | 0.975 | 0.953 | 0.18 |
| bf16 vs fp32, seed 102 | 0.953 | 0.934 | 0.49 |
| bf16 vs fp32, seed 103 | 0.972 | 0.986 | 0.17 |
| bf16 vs fp32, seed 104 | 0.955 | 0.892 | 0.17 |
| fp32 vs fp32 across seeds (6 pairs, mean) | 0.892 | 0.877 | 1.08 |
| fp32 vs fp32 across seeds, closest pair | 0.916 | 0.935 | 0.92 |

Every bf16 row is closer to fp32 than the closest pair of fp32 seeds on lDDT and binder RMSD.

The first chain has several domains that move against each other from seed to seed, so superposed
RMSD of the whole complex reads 9 to 18 A between two fp32 seeds and is not a useful yardstick here.

Confidence is unchanged: mean pTM 0.7058 against 0.7055, ipTM 0.9232 against 0.9230 over the same
four seeds. The same is not true of trunk precision settings, which is why none of them is offered:
`TT_BIO_TRUNK_MATH_FIDELITY=hifi2` moves pTM by +0.031 (about thirty times the seed spread) and buys
no time on Wormhole, and `--fast` is 2.6 % slower than the default on Wormhole at this size.

## Accuracy on eleven complexes

The table above is one complex. A second grade folded eleven two-chain complexes released after the
model's training cutoff (515 to 791 tokens), three fp32 seeds and two bf16 seeds, five samples each,
Wormhole at 1000 MHz. Deviation here is RMSD per chain over residues the fp32 model is confident about
(pLDDT 70 or higher), worst chain, bf16 sample against the fp32 sample of the same seed.

On nine of the eleven, bf16 is a rounding of fp32: median deviation 0.11 to 0.24 A, against 0.34 to
1.29 A between two fp32 seeds. The top-ranked structure scores the same against the deposited
structure (DockQ and TM-score within 0.03).

On the other two, fp32 does not find the deposited interface at any seed (DockQ 0.01), and bf16 picks
a different binding mode:

| Complex | Seed | fp32 top-ranked (DockQ / TM) | bf16 top-ranked (DockQ / TM) |
|---|---|---|---|
| 9W8A | 101 | 0.01 / 0.83 | 0.88 / 0.99 |
| 9W8A | 102 | 0.01 / 0.83 | 0.00 / 0.82 |
| 9W89 | 101 | 0.01 / 0.83 | 0.06 / 0.10 |
| 9W89 | 102 | 0.01 / 0.84 | 0.06 / 0.86 |
| 9W89 | 103 | 0.01 / 0.83 | 0.05 / 0.86 |

At 9W8A seed 101 bf16 finds the deposited interface. At 9W89 seed 101 four of five bf16 samples fold
the first chain about 41 A away from fp32, wrong against the deposited structure (TM 0.10), and they
carry a higher ranking score (0.71 to 0.76) than the one sample that folded correctly (0.49). fp32
folded that chain correctly in all fifteen of its samples. A precision setting that can hand you a
confidently wrong top structure is why bf16 is not the default.

The raw measurements and the scripts are in `perf/pfm_ttfast/`.
