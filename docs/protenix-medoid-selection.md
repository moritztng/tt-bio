# Protenix-v2 medoid/consensus selection — investigated, does NOT generalize, NOT adopted

## Motivation

Protenix-v2's confidence head barely discriminates between diffusion samples and can
*anti-rank* on hard targets: on 7ROA it delivers a 3.87 Å sample as "best" while a
2.34 Å sample sat unused in the ensemble (`docs/protenix-accuracy-investigation.md`).
This is Protenix-v2's own confidence-head weakness — reproduced in the official
upstream reference (`docs/protenix-v2-reference-rootcause.md`), *not* a tt-bio port
bug, and not present in Boltz-2 / ESMFold2 (whose heads rank fine).

A standard trick when a confidence signal is unreliable but the ensemble clusters
near the right answer is **medoid / consensus selection**: ignore the score, hand
back the *most typical* structure — the sample with the lowest mean pairwise
CA-RMSD to the others. This note tests whether that recovers any of the gap, with
**zero change to the diffusion model** (selection-time only).

Harness: `scripts/protenix_medoid_selection.py` (reuses `tests/test_structure.py`'s
Kabsch/TM primitives verbatim; does not re-derive RMSD). Two distinct ground-truth
targets, both folded on-device (qb1 card 0, `--use_msa_server --sampling_steps 200
--diffusion_samples 5 --seed 0`, protenix-v2):

- **7ROA** (`examples/prot.yaml`) — 117-res α monomer, shallow MSA. The hard target
  where the confidence gap actually exists.
- **hemoglobin** (`examples/hemoglobin.yaml`) — α₂β₂ tetramer, deep MSA. An easy
  target, as a generalization control.

## Results (real on-device runs, CA-RMSD in Å)

**7ROA** — per sample, ranked by confidence (rank 0 = today's delivered pick):

| rank | gt_rmsd | gt_tm | mean_pair | note |
|-----:|--------:|------:|----------:|------|
| 0 | 3.705 | 0.714 | 1.934 | best-conf (delivered) |
| 1 | 3.631 | 0.709 | **1.874** | MEDOID |
| 2 | 4.521 | 0.699 | 2.650 | |
| 3 | **2.374** | 0.806 | 2.328 | **oracle** |
| 4 | 2.493 | 0.794 | 2.910 | |

**hemoglobin**:

| rank | gt_rmsd | gt_tm | mean_pair | note |
|-----:|--------:|------:|----------:|------|
| 0 | 1.349 | 0.976 | 0.920 | best-conf (delivered) |
| 1 | 1.311 | 0.977 | **0.759** | MEDOID |
| 2 | **0.882** | 0.990 | 1.014 | **oracle** |
| 3 | 1.668 | 0.964 | 0.933 | |
| 4 | 1.351 | 0.976 | 0.843 | |

| target | best-conf | medoid | medoid + pTM tiebreak | oracle |
|--------|----------:|-------:|----------------------:|-------:|
| 7ROA        | 3.705 | 3.631 (**+0.074**) | 3.705 (**= best-conf**) | 2.374 |
| hemoglobin  | 1.349 | 1.311 (**+0.038**) | 1.349 (**= best-conf**) | 0.882 |

## Verdict: does not help, does not generalize — NOT adopted

1. **Negligible standalone effect.** Bare medoid moves the delivered RMSD by
   **+0.074 Å (7ROA)** and **+0.038 Å (hemoglobin)** — both far inside TT
   diffusion's seed-to-seed noise (~2.4 Å pairwise spread on 7ROA,
   `docs/protenix-accuracy-investigation.md`). It is a coin-flip, not a signal.

2. **With the pTM tiebreak it is a literal no-op.** When two samples are within
   0.5 Å of the medoid distance, breaking the tie by confidence collapses the pick
   back to rank 0 on **both** targets — i.e. medoid + tiebreak ≡ the current
   confidence pick. No structure changes hands.

3. **The premise is refuted — the good sample is the ensemble OUTLIER.** On both
   targets the oracle-best sample has one of the *highest* mean-pairwise distances
   (7ROA rank 3: 2.328; hemoglobin rank 2: 1.014), not the lowest. The diffusion
   mode is a wrong basin (7ROA, ~3.6–4.5 Å cluster of 3) or a slightly-worse basin
   (hemoglobin); the best structures are the *minority*. A medoid is a density
   estimator, so it steers **toward the crowded basin** — the exact opposite of
   what recovering the oracle requires. This is fundamental to consensus selection
   whenever the model lands in a wrong basin more often than the right one, so it
   is not fixable by a smarter distance/linkage — any consensus/density method
   fails the same way here.

4. **No confidence gap to close on the easy target anyway.** On hemoglobin the
   confidence head already delivers 1.349 Å / TM 0.976 (near the 0.882 Å oracle) —
   the flat pTM (0.816–0.818) doesn't hurt when the whole ensemble is good.

**Decision:** the Protenix-v2 selection path (`tt_bio/worker.py`
`_predict_protenix_one`, confidence-ranked `_score`/`order`) is **left unchanged**.
Medoid selection is not wired into `predict`. The prototype
(`scripts/protenix_medoid_selection.py`) is kept as a reusable diagnostic for any
future selection idea, but this heuristic is a dead end for the documented gap.

Consistent with the STOP-if-it-doesn't-generalize discipline: no marginal or
fabricated win is forced. Recovering Protenix-v2's oracle here would require a
better *per-sample* quality signal (the confidence head itself), not a selection
rule over samples the head already can't tell apart.
