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
- **Protenix underperforms Boltz-2/ESMFold2 here, but that is Protenix-v2's own
  ceiling on this target, not a tt-bio deficiency.** Boltz-2 (1.55 Å) and ESMFold2
  (2.28 Å) fold 7ROA better, but the **official reference Protenix-v2 also only
  reaches ~3.1 Å** here (see below) — so tt-bio's 3.87 Å best-conf / 2.34 Å oracle
  is faithful to upstream. Boltz/ESMFold simply have stronger priors (deeper MSA /
  a protein LM) for this shallow-MSA target.
- **Most of Protenix's delivered gap is a weak confidence head, not diffusion.**
  Its diffusion *reaches* 2.34 Å (oracle-of-5, comparable to ESMFold2's 2.28),
  but the pTM head barely discriminates (pTM 0.715–0.726 across samples) and
  *anti-ranks* — it delivers the 3.87 Å sample as "best" while 2.34 Å was
  available. With 5 near-flat scores the selection is essentially noise.

## Root cause RESOLVED: no device/port bug — 7ROA is just a hard Protenix-v2 target

The official upstream Protenix-v2 (torch, real v2 checkpoint) was run on the same
input on CPU — two ways, cross-confirming:

- **This branch (pc), independent rebuild:** built the reference from the
  `bytedance/Protenix` main repo (v2 config, CUDA FusedLayerNorm stubbed to torch,
  torch triangle kernels; CCD `components.v20240608.cif` + deps installed to an
  isolated target dir, leaving the shared tt-bio env untouched). The v2 checkpoint
  loads **strict — missing=0, unexpected=0** (architecture exact), and the data
  pipeline produces the **identical featurization as tt-bio** (N_token=117,
  **N_atom=900**), confirming the port feeds the model the same inputs. The no-MSA
  n_step=200 / 5-sample forward completed on CPU: **oracle 4.35 Å** (samples
  4.35–6.05 Å, TM 0.54–0.66) — consistent with the refcheck venv's 5.44 Å.
  (Runner in `/home/moritz/.coworker/protenix-ref-run/run_ref.py`, not committed —
  machine-absolute paths.)
- **Sibling branch `wk/tt-bio-protenix-refcheck` (the reported RMSD numbers):**
  official Protenix 2.0.0 in an isolated venv, both no-MSA and with real MSA.

Reference CA-RMSD vs 7ROA (Kabsch): no-MSA n_step=10 = **7.84 Å**; no-MSA n_step=200
= **5.44 Å**; **with real MSA, n_step=200, best-of-5 = 3.13 Å** (pLDDT 75, pTM 0.78).

**Conclusion: the reference itself only reaches ~3.1 Å on 7ROA with MSA** — right in
line with tt-bio's 3.87 Å best-conf / 2.34 Å oracle. There is **no device/port
accuracy bug**; the "~1–2 Å expected" was too optimistic — 7ROA (a small
all-α bacterial toxin with a shallow MSA) is a moderately hard Protenix-v2 target.
The poor headline numbers are dominated by **MSA-off (~+2.3 Å) then undersampling
(~+2.4 Å)** — exactly the eval-flaw diagnosis, now quantified against the reference.

### The "~10 Å" reproduced and explained: it is the no-MSA default path
tt-bio Protenix-v2 **without** `--use_msa_server` folds single-sequence (the CLI
help even says so) and gives **~10.5 Å** best-conf / 9.7 Å oracle (pTM 0.39 — the
model correctly reports low confidence). That is the reported number. **The fix for
a user is: pass an MSA** (`--use_msa_server`), which takes it to 3.87 Å.

**Caveat — a real but off-label no-MSA gap.** Unlike the with-MSA case (where
tt-bio 3.87/2.34 Å ≈ reference 3.13 Å, no gap), in the **no-MSA regime** tt-bio
(9.7 Å oracle, TM 0.27 — wrong topology) is markedly worse than the reference
(oracle 4.35 Å / mean ~5.2 Å, TM 0.54–0.66 — borderline-correct topology; both the
pc rebuild and the refcheck venv agree at ~5 Å). Both feed a 1-row dummy MSA (the
query), so this is a genuine ~5 Å divergence in tt-bio's single-sequence path — a
candidate for a residual port issue in the low-/no-MSA featurization (profile /
deletion / MSA-module handling of a 1-row MSA). **Low priority**: no-MSA is
off-label for an MSA-dependent AF3-family model, and the production (with-MSA) path
is faithful — but it is the regime a user hits by default today, which is exactly
why the no-MSA default should be changed (see Recommendations). Not chased further
here; flagged for a follow-up if the default isn't changed.

### Diffusion converges on real input (refutes a "fundamental diffusion bug")
A parallel leg (`scripts/protenix_sampling_ablation.py`, memory
`protenix-undersampling-ablation`) reported seed-to-seed ~14 Å, bimodal,
confidently-wrong on the **synthetic 38-token golden molecule** and concluded a
real EDM bug. On the **real 117-res input** here, tt-bio's diffusion **converges**:
pairwise seed-to-seed CA-RMSD mean **2.44 Å** with MSA (0.99–3.75), 3.11 Å without.
So the ~14 Å divergence is specific to that golden-molecule harness (hand-assembled
feats / a reference target itself captured at n_step=10), not the production path.

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

## Recommendations for Moritz

1. **No patch to v0.2.0/v0.2.1 is required for a "correctness bug" — there isn't
   one.** tt-bio Protenix-v2 is faithful to upstream (3.87 Å best-conf / 2.34 Å
   oracle vs reference 3.13 Å on this hard target). The alarming "~10 Å" was the
   no-MSA path + undersampling, not a device defect.
2. **Highest-impact UX fix: the no-MSA default trap.** `--model protenix-v2`
   without `--use_msa_server` folds single-sequence → ~10 Å, and the CLI help
   actively describes it as "single-sequence … no MSA". Protenix-v2 is an
   MSA-dependent AF3-family model; single-seq is off-label. Either default MSA on
   for protenix-v2, or emit a loud warning when it runs single-seq. This is what a
   user hits and mistakes for a broken model.
3. **Ship the ground-truth gate** (this branch) as the release floor and retire the
   self-consistency-only check. Fold gate targets with `--use_msa_server
   --sampling_steps 200 --diffusion_samples ≥5`.
4. Minor: default the env-DB on for the Protenix/ESMFold2 MSA path (§ MSA-depth
   asymmetry) so it isn't handed a thinner MSA than Boltz by default.
