# Protenix-v2 `--diffusion_samples` default — investigated, NOT raised

## Motivation

`tt-bio predict` defaults `--diffusion_samples` to **1** for structure prediction
(`tt_bio/main.py`) while `--diffusion_samples_affinity` already defaults to **5**. Since
Protenix-v2's confidence head is known to barely discriminate — and can outright
anti-rank — between diffusion samples on hard targets (`docs/protenix-accuracy-investigation.md`,
`docs/protenix-v2-reference-rootcause.md`, both closed root-causes, not re-derived here),
raising the structure-prediction default to match the affinity default (N=5) was
proposed as a cheap accuracy lever: more draws, same confidence-based pick. This note
measures whether it actually helps and at what cost.

This is a different question from the medoid/consensus selection investigation
(does a smarter *pick* among existing samples help — answered no): here the pick
rule is unchanged (confidence-selected, exactly what ships today) and only the
sample *count* N varies.

## Method

Two ground-truth targets, on-device (qb1 card 0, protenix-v2, `--use_msa_server
--sampling_steps 200`), N ∈ {1, 3, 5}, seed 0:

- **7ROA** (`examples/prot.yaml`) — 117-res monomer, shallow MSA. The hard target
  where the confidence gap is known to exist.
- **hemoglobin** (`examples/hemoglobin.yaml`) — α₂β₂ tetramer, deep MSA. Easy-target
  generalization control.

Ground-truth CA-RMSD/TM of the **confidence-selected** structure (`{name}.cif`,
exactly what a user receives) via `tests/test_structure.evaluate` — reused verbatim,
not re-derived. Wall-clock read from each run's `results.json` `runtime_s`.

Seed-noise floor: 7ROA re-run at N=1 with seeds 0, 1, 2, to know how much of any
N-driven change is actually just seed variance.

## Results (real on-device runs, CA-RMSD in Å, confidence-selected pick)

**7ROA** — delivered (confidence-selected) RMSD is **bit-identical across N**:

| N | delivered RMSD | delivered confidence | oracle-best in ensemble | runtime_s |
|--:|---------------:|----------------------:|-------------------------:|----------:|
| 1 | 3.705 | 0.7255 | 3.705 (only draw) | 13.5 |
| 3 | 3.705 | 0.7255 | 2.374 | 25.1 |
| 5 | 3.705 | 0.7255 | 2.374 | 36.9 |

Per-sample confidence scores at N=5 (from `results.json`), for reference — genuinely
distinct, not a selection bug, and rank-ordered *opposite* to ground-truth quality:

| rank | confidence | gt_rmsd |
|-----:|-----------:|--------:|
| 0 | 0.7255 | 3.705 (delivered) |
| 1 | 0.7223 | 3.631 |
| 2 | 0.7211 | 4.521 |
| 3 | 0.7164 | **2.374 (oracle)** |
| 4 | 0.7161 | 2.493 |

7ROA seed-noise floor (N=1, 3 seeds): **3.705 / 2.290 / 2.371 Å** — a ~1.4 Å swing from
the seed alone, an order of magnitude bigger than anything N could plausibly buy.

**hemoglobin** — same pattern, delivered RMSD identical across N:

| N | delivered RMSD | delivered confidence | oracle-best in ensemble | runtime_s |
|--:|---------------:|----------------------:|-------------------------:|----------:|
| 1 | 1.349 | 0.7898 | 1.349 (only draw) | 71.6 |
| 3 | 1.349 | 0.7898 | 0.882 | 97.0 |
| 5 | 1.349 | 0.7898 | 0.882 | 122.8 |

**Cost** — going from N=1 to N=5 is not the naive 5x (fixed per-fold overhead
amortizes some of it), but it is substantial and roughly linear in N beyond that
fixed cost:

| target | N=1 | N=5 | ratio |
|--------|----:|----:|------:|
| 7ROA (small, shallow MSA) | 13.5s | 36.9s | 2.73x |
| hemoglobin (large, deep MSA) | 71.6s | 122.8s | 1.71x |

## Why the delivered pick never moves

The generation of sample index 0 is deterministic given the seed, independent of how
many total samples are requested, and its confidence score is a per-sample function
that does not depend on the ensemble size (confirmed: confidence score for the
delivered structure is bit-identical across N=1/3/5 on both targets, to 6 decimal
places). On both targets tested, sample 0's confidence also happens to be the
*highest* in the N=3 and N=5 ensembles. So confidence-based selection reliably
re-picks the exact same structure regardless of N — raising N only ever adds
candidates *below* the incumbent in the anti-ranked ordering. This is a direct,
concrete consequence of the already-documented confidence-head weakness
(`docs/protenix-accuracy-investigation.md`); it is not a new root cause and was not
re-investigated here.

## Verdict: does NOT justify raising the default — NOT adopted

1. **Zero measured accuracy improvement.** On both targets, at N=3 and N=5 the
   confidence-selected (delivered) structure is bit-identical to N=1's single draw.
   The oracle-best sample genuinely improves with more draws (7ROA 3.705→2.374 Å,
   hemoglobin 1.349→0.882 Å at N=5) — but that improvement is **unreachable** by the
   selection rule the CLI actually ships, so it is not a real user-facing gain.
2. **The seed floor dwarfs it.** 7ROA's seed-to-seed spread at fixed N=1 (~1.4 Å) is
   far larger than any change N produced (0.000 Å on both targets tested) — the
   compute is better spent elsewhere.
3. **Real added cost for zero return.** N=5 costs 1.7-2.7x the wall-clock of N=1
   with no accuracy benefit measured.

`tt_bio/main.py`'s `--diffusion_samples` default is left at **1** for structure
prediction. `--diffusion_samples_affinity`'s default of 5 is a pre-existing,
unrelated choice for the affinity path and is out of scope here (not touched).
Fixing this for real requires a better *per-sample* confidence signal, not more
samples fed through the same signal — consistent with the medoid/consensus
selection investigation's conclusion (selection-side fixes don't recover this gap
either). No code change is proposed; this is a documented dead end, same standard
as `protenix-medoid-selection` / `boltzgen-batch-threshold-dead-end`.
