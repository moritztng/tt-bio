# Protenix-v2 accuracy investigation + ground-truth release gate

Prompted by a report that tt-bio Protenix-v2 folds `examples/prot.yaml`
(117-res chain A, PDB **7ROA**, *EntV136* from *E. faecalis*) to ~10 Å vs the
experimental structure, while the v0.2.0/v0.2.1 release gate only checked
device-vs-*reference* self-consistency (seed0-vs-ref Kabsch 8.7 Å, "within
variance") — a check that passes even when the fold is wrong.

## What the old gate missed

The shipped self-consistency harness (`scripts/protenix_fold_e2e.py`) used
`n_step=10, n_sample=2` and compared seed-to-seed / vs a reference *trajectory*,
never vs the experimental structure. Two independent flaws:

1. **Undersampling.** `n_step=10` is far below production. Measured on 7ROA
   (best-confidence sample, MSA, 5 samples): `n_step=10` → **6.48 Å** (TM 0.46);
   `n_step=200` → **3.87 Å** (TM 0.71). The old harness's tiny step count alone
   accounts for ~2.6 Å of apparent error.
2. **Self-consistency ≠ accuracy.** Seed-vs-reference RMSD says nothing about
   whether the fold matches reality.

## Ground-truth results on 7ROA (production settings)

`--use_msa_server --sampling_steps 200 --diffusion_samples 5 --seed 0`, CA-RMSD
of the **confidence-selected** structure (model 0) via `tests/test_structure.py`:

| model      | best-conf CA-RMSD | best-conf TM | oracle-of-5 | verdict |
|------------|-------------------|--------------|-------------|---------|
| Boltz-2    | **1.55 Å**        | 0.93         | 1.38 Å      | excellent |
| ESMFold2   | **2.28 Å**        | 0.83         | 2.28 Å      | good |
| Protenix-v2| **3.87 Å**        | 0.71         | **2.34 Å**  | correct topology, weakest of the three |

Takeaways:

- **The "~10 Å" was an eval artifact** (undersampling + self-consistency-only).
  With production sampling Protenix-v2 folds 7ROA with the correct topology
  (TM 0.71–0.81), not garbage. This is *not* a catastrophic release bug.
- **But Protenix underperforms the other two tt-bio models on an easy target.**
  Boltz-2 (1.55 Å) proves 7ROA is easy; ESMFold2 (2.28 Å) — same diffusion
  family, same stack — confirms tt-bio's pipeline folds it well.
- **Most of Protenix's delivered gap is a weak confidence head, not diffusion.**
  Its diffusion *reaches* 2.34 Å (oracle-of-5, comparable to ESMFold2's 2.28),
  but the pTM head barely discriminates (pTM 0.715–0.726 across samples) and
  *anti-ranks* — it delivers the 3.87 Å sample as "best" while 2.34 Å was
  available. With 5 near-flat scores the selection is essentially noise.

## Open question (needs the reference)

Whether even the 2.34 Å oracle matches what the **official upstream Protenix-v2**
(torch) produces on this input is UNRESOLVED. A correct Protenix-v2 with a good
MSA is typically ~1–2 Å on a monomer this easy, so a residual port-fidelity gap
cannot be excluded. Running the reference end-to-end needs its full data pipeline
(CCD `components.v20240608.cif` ~468 MB, biotite/rdkit/ml_collections, MSA
featurizer) — a multi-session lift, not done here. Recommended definitive test:
feed the reference model (real weights, `scripts/protenix_ref_build.py`) tt-bio's
own `feats` dict + reference EDM sampler at n_step=200 and compare — isolates
port fidelity from the data pipeline.

## Secondary finding: MSA-depth asymmetry (accuracy-affecting, fixable)

Boltz-2's MSA path searches the environmental DB unconditionally
(`run_mmseqs2(..., use_env=True)`, 94 seqs for 7ROA); the Protenix/ESMFold2 path
(`_generate_esmfold2_a3m` → `run_mmseqs2(..., use_env=use_envdb)`) skips it
unless `--use_envdb` is passed (34 seqs). Not the primary cause here (ESMFold2
uses the same shallow path and still folds to 2.28 Å), but it hands Protenix a
thinner MSA than Boltz by default. Consider defaulting env-DB on for these paths.

## The release gate (this branch)

`tests/test_structure.py` is now a ground-truth floor usable for any foldable
target, not just a manual RMSD print:

- Kabsch CA-RMSD **and** TM-score (TM is length-normalized; RMSD alone is
  inflated by a single flexible tail on an otherwise-correct fold).
- Scores the **confidence-selected** structure (model 0 = best-of-N by pTM/pLDDT,
  exactly what a user receives), and also reports the oracle-best and per-sample
  spread so a mis-ranking confidence head is visible.
- gemmi + numpy parsing (was biopython `MMCIFParser`, which crashed on ESMFold2's
  minimal cif that omits `_atom_site.occupancy`) — works for all three models.
- `--max-rmsd` / `--min-tm` turn it into a pass/fail gate (exit 1 on miss).

Usage (after folding with production settings):

    python tests/test_structure.py prot --max-rmsd 6 --min-tm 0.5

The generous floor (RMSD ≤ 6 Å, TM ≥ 0.5) catches gross failures — the
undersampled `n_step=10` run FAILS it (6.48 Å / TM 0.46) while all three
production folds PASS. **Self-consistency alone is no longer sufficient; a tagged
release must clear a ground-truth floor on at least one foldable target per
model.** Tighten thresholds per model/target as baselines are established
(Boltz-2 clears ≤ 2 Å; Protenix-v2 currently needs the ≤ 6 Å floor).
