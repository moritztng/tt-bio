# Does the pad-up move OpenFold3's structure? Five folds of HSA say no.

`TT_BIO_TRIATT_HIFI_PAD_UP` was built for BindCraft 2 and graded against float64 at BindCraft 2's
shape, but it sits inside `_tri_att_sdpa_hifi_inner`, which is the triangle attention every model
in the repo shares, and OpenFold3's trunk passes `tri_att_sdpa_hifi=...("openfold3.trunk", True)`
by default. Counters measured in the process that folds say the pad-up takes **all 384** of
OpenFold3's trunk calls at 544 and 608 residues, and that all 384 decline with it off. So it moves
a second shipped model onto a different route, and the question is what that does to the structure
the user gets.

## The target

`examples/hsa.yaml`, human serum albumin, 585 residues, already a parity fixture in this repo.
585 tile-pads to a pair axis of **608 = 32 * 19**, one of the lengths with no legal fused config,
so the pad-up fires and pads to 640. Its MSA was already searched (1000 sequences, hash
`a35bb68136a5125a`, from the protenix-v2 parity runs) and is passed with `--msa_cache_only`, so a
cache miss is an error rather than a silent single-sequence fold. Every leg reads
`msa_depth 1001` and **pLDDT 0.909 to 0.913**: the model has a definite opinion about this fold,
which is what the earlier 544-mer probe did not (`msa: empty`, pLDDT 0.28, both arms folded to
noise and the 7.2 A between them measured the noise).

All five folds on qb1 UMD card 1, one at a time, **AICLK median and max 1350 MHz sampled from
sysfs during each fold** (`clk_*.txt`; the 800 in each file is the idle sample before the fold
starts).

## The arm swap against the seed floor

Superposed CA-RMSD, Kabsch, over all 585 CA:

| pair | what it measures | CA-RMSD |
|---|---|---|
| pad-up on vs off, seed 1 | **the arm** | **0.0298 A** |
| pad-up on vs off, seed 2 | **the arm, again** | **0.0312 A** |
| on seed 1 vs on seed 3 | the seed | 0.8326 A |
| on seed 2 vs on seed 3 | the seed | 0.9191 A |
| on seed 1 vs on seed 2 | the seed | 1.1591 A |

**The arm swap is 0.03 A and the kill bar is 0.60 A, so it is 20x under the bar and 28x smaller
than the smallest move a seed change makes on this same input.** It reproduces at a second seed to
within 0.0014 A. Confidence moves the same way: pLDDT 0.909254 against 0.909316 between the arms,
where three seeds of the one arm span 0.909254 to 0.913107, a spread 62x larger.

| leg | pLDDT | pTM | fold s | pad-up counters, in the child that folded |
|---|---|---|---|---|
| on, seed 1 | 0.909254 | 0.879613 | 142.2 | served 384, declined 0, padded 608 -> 640 |
| on, seed 2 | 0.909502 | 0.880933 | 66.5 | served 384, declined 0, padded 608 -> 640 |
| on, seed 3 | 0.913107 | 0.888198 | 68.4 | served 384, declined 0, padded 608 -> 640 |
| off, seed 1 | 0.909316 | 0.880243 | 86.6 | served 0, **declined 384** |
| off, seed 2 | 0.909649 | 0.881800 | 88.7 | served 0, **declined 384** |

The parent process of every leg reads all zeros and is not the measurement: `tt-bio predict` spawns
its worker and stops it with `terminate()`, so the child never runs `atexit`. Each `inert_*.json`
here is one pid, and the two per leg are the parent and the worker.

## Speed, and the one number not to read off this table

Warm, the pad-up is **1.30x** on the whole fold: 66.5 and 68.4 s on against 86.6 and 88.7 s off.
The trunk phase alone, from the progress log, is 21 s on against 29 s and 27 s off.

**The 142.2 s in the seed-1 on row is not a regression.** It was the first process to touch the
card in this series and paid the cold ttnn kernel compile; every later leg reused it. Any
arm-to-arm speed claim here rests on the warm legs and on the trunk phase, not on that number.
