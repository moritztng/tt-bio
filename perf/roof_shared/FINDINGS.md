# TT against upstream Boltz-2, with the sampler's draws shared

`k10-p1-accuracy-anchor` measured the port against upstream `boltz==2.2.1` at fp32 and got
0.90077 Å at 298 aa and 1.42726 Å at 512 aa (worst per-pseudo-domain, all-atom). Both stacks drew
their own diffusion noise, so those numbers contain the sampler's spread: the reference compared to
itself with nothing but its noise realisation changed reads 0.92565 Å and 1.82527 Å. This run folds
the TT side with `TT_BIO_SHARED_DRAW_SEED=0`, which puts both stacks on the same noise, and scores
it against the committed `gpurefshared` CIFs. What is left is arithmetic.

## The draws do align, and that is checked rather than assumed

The sampler's first draw is recorded in `out/folds.json` for every fold. At 298 aa the shared arm
draws `(1, 2400, 3)` with digest `84c748a64d2a8963`, and `torch.manual_seed(0); torch.randn((1,
2400, 3))` on the CPU generator gives exactly `84c748a64d2a8963`. At 512 aa it is `(1, 4128, 3)`
and `5936883ad8f94f53`, again equal to the canonical CPU stream. Both stacks run torch
2.14.0+cu130, and `perf/k10_anchor/gpu_ref_fold.py --mode shared` reseeds the same generator at
`AtomDiffusion.sample` entry and draws the same shape, so the two samplers consume the same noise
from the same position.

The padded atom axis is why the shape is 2400 and not the CIF's 2397 atoms: `atoms_per_window_
queries = 32` in both featurizers (`tt_bio/data/featurizer.py` is a verbatim port of
`boltz/data/feature/featurizerv2.py`), so the atom count is rounded up to a multiple of 32 on both
sides. Had one side padded and the other not, the streams would have diverged at the first draw.

## The result

Seed 0, one Wormhole chip (whglx card 5), commit `209fe9f23` (= `origin/main` plus this directory),
3 recycles, 200 sampling steps, 1 diffusion sample, potentials as shipped, the committed 35-row
a3m.

| 298 aa, one pseudo-domain | all-atom | CA | CA-lDDT arm vs ref |
|---|---|---|---|
| **TT shared vs `gpurefshared`, same draws** | **0.43277 Å** | **0.31302 Å** | **0.99288** |
| TT plain vs `gpuref`, independent draws | 0.88049 Å | 0.52089 Å | 0.98561 |
| reference vs itself, only the draws changed | 0.92565 Å | 0.51112 Å | 0.97880 |
| reference seed floor, 6 pairs | 0.74099 - 0.84656 Å | | |

| 512 aa, worst per-pseudo-domain | all-atom | CA | hinge | CA-lDDT arm vs ref |
|---|---|---|---|---|
| **TT shared vs `gpurefshared`, same draws** | **0.68922 Å** | **0.55158 Å** | **1.12°** | **0.97197** |
| TT plain vs `gpuref`, independent draws | 1.87250 Å | 1.46249 Å | 115.01° | 0.89197 |
| reference vs itself, only the draws changed | 1.82527 Å | 1.50140 Å | 165.81° | 0.92848 |
| reference seed floor, 6 pairs | 0.94540 - 2.29123 Å | | | |

Per pseudo-domain at 512 aa the paired reading is 0.68922 Å on copy 1 and 0.53884 Å on copy 2,
hinge-free 0.63079 Å, whole-structure all-atom 0.87073 Å.

**48 % of the anchor's unpaired number survives the pairing at both sizes** (0.43277 of 0.90077 and
0.68922 of 1.42726). The other half was the sampler. Against the unpaired arm folded in this same
session on this same card it is 49 % at 298 aa and 37 % at 512 aa.

The two terms add in quadrature, which is the check that the split is real: 298 aa
sqrt(0.88049² - 0.43277²) = 0.76679 Å of implied noise against a directly measured floor of
0.86083 Å (TT, s0 vs s1) and 0.92565 Å (reference, same seed, different realisation); 512 aa
sqrt(1.87250² - 0.68922²) = 1.74104 Å against 1.85827 Å and 1.82527 Å.

The hinge is the clearest single statement. `cdk2x2_512` is a chimeric fixture whose two
pseudo-domains have no interface, and unpaired it opens by 95-166°, which is what drives
whole-structure RMSD to 11-22 Å. With the draws shared it opens by **1.12°** and whole-structure
all-atom falls to 0.87073 Å. The hinge is a degree of freedom of the sampler's noise, not of the
port's arithmetic.

## The standing CA-lDDT deficit, now measured paired

Against the experimental structure 1HCL, seed 0, both stacks on the same draws:

| | reference (`gpurefshared-s0`) | TT (`ttshared-s0`) | paired delta | anchor's unpaired estimate |
|---|---|---|---|---|
| 298 aa, CA-lDDT | 0.98351 | 0.97765 | **-0.00586** | -0.0102 |
| 512 aa copy 1, CA-lDDT | 0.94304 | 0.92817 | **-0.01487** | -0.0156 |
| 512 aa copy 2, CA-lDDT | 0.94169 | 0.92707 | **-0.01462** | -0.0240 |
| 298 aa, CA RMSD vs 1HCL | 0.60389 Å | 0.77810 Å | +0.17421 Å | |
| 512 aa copy 1, CA RMSD | 1.56904 Å | 1.83222 Å | +0.26318 Å | |
| 512 aa copy 2, CA RMSD | 0.92932 Å | 0.98779 Å | +0.05847 Å | |

The deficit survives pairing, same sign at all three positions, and it is the one reading here that
is not explained by the sampler. It is also smaller than the unpaired estimate at two of the three
positions: copy 2 halves, from -0.0240 to -0.01462. One shared draw exists on the reference side, so
this is n=1 and the sign is worth more than the third decimal. Cause is out of scope for this row
and predates the campaign.

## Controls

* **A/A floor: 0.00000 Å.** The four seed-0 folds were run twice in two separate processes on the
  same card and the CIFs are byte-identical (`8b8182e7917e60f4`, `9e8e0a0ff814702f`,
  `2ad635147a490598`, `8cabfca3308ed3cd` in both runs). Nothing quoted above is run-to-run noise.
* **The instrument reproduces the anchor.** `perf/b2z2_fusebias/score.py` is used unmodified, and
  scoring `gpuref-s0` against `gpurefshared-s0` through this pipeline returns 0.92565 Å and
  1.82527 Å, the anchor's own published numbers, and the reference's 6-pair seed floor comes back
  at the same 0.74099 - 0.84656 Å and 0.94540 - 2.29123 Å.
* **Shared draws are not a smoothing trick.** The TT shared arm compared to the TT plain arm at the
  same seed, which is the same code with a different noise realisation, reads 1.11126 Å at 298 aa
  and 2.70152 Å worst per-pseudo-domain at 512 aa, i.e. seed-floor scale. The pairing narrows the
  cross-stack comparison, it does not narrow everything it touches.

## Folds

Wormhole B0, 8x9 grid, model load 4.361 s, `out/folds.json`:

| size | arm | seed | wall | plDDT |
|---|---|---|---|---|
| 298 | shared | 0 | 22.095 s (first fold of the session) | 0.915212 |
| 298 | plain | 0 | 19.628 s | 0.915316 |
| 298 | shared | 1 | 19.638 s | 0.912197 |
| 298 | plain | 1 | 19.719 s | 0.917048 |
| 512 | shared | 0 | 37.643 s | 0.811365 |
| 512 | plain | 0 | 36.348 s | 0.800356 |
| 512 | shared | 1 | 36.721 s | 0.835675 |
| 512 | plain | 1 | 36.127 s | 0.815252 |

Card note: this is a Wormhole chip, while the anchor's TT arm ran on Blackhole (qb2). Both TT arms
here, paired and unpaired, are folded in one session on the one card, so the comparison between
them is card-consistent; only the comparison against the anchor's absolute Blackhole figures
crosses card types. pc's p150a was excluded on purpose (`pc-card0-512aa-fold-nondeterminism`) and
no qb2 card was free.

## Scoring the prediction in `PREDICTED.md`

1. Draws align — **right**, and verified by digest rather than inferred.
2. 298 aa paired 0.10-0.35 Å — **wrong**, 0.43277 Å, high by 0.08 Å.
3. 512 aa paired 0.15-0.55 Å — **wrong**, 0.68922 Å, high by 0.14 Å.
4. "at least 60 % of the unpaired figure dissolves" — **wrong**, 52 % dissolves.
5. CA-lDDT deficit moves by less than ±0.005 — **right at two positions of three**: 298 aa moves
   0.0043 and copy 1 moves 0.0007, copy 2 moves 0.0094 and shrinks by more than the band allowed.
6. TT plain against TT shared lands at seed-floor scale — **right** at 298 aa (1.11126 Å, band was
   0.7-1.3), **high** at 512 aa (2.70152 Å against a 1.0-2.0 Å band, and against the reference's own
   0.94540-2.29123 Å floor it is inside).

The systematic direction of the miss is worth keeping: the port's arithmetic sits about 1.6x
further from upstream than an order-of-magnitude-above-bf16 argument predicted.

## Reproducing

```
fold_shared.py --out out/folds.json --cifdir cif --sizes 298,512 --extra-seeds 1
assemble.py --tt-runs out/folds.json --tt-cif cif --out-cifdir scorecif --out-runs out/score_runs.json
../b2z2_fusebias/score.py scorecif --runs out/score_runs.json --split 298 \
    --out out/score_shared.json --arms gpurefshared,ttshared,gpuref
../b2z2_fusebias/score.py scorecif --runs out/score_runs.json --split 298 \
    --out out/score_plain.json  --arms gpuref,ttplain,ttshared
report.py --shared out/score_shared.json --plain out/score_plain.json --out out/report.json
```
