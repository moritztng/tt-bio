# ESMFold2 GB1 coordinate-RMSD gap: root cause

## Question

The multi-seed ESMFold2 parity run (`scripts/esmfold2_e2e_parity.py`) reported that
trp-cage's (L=20) device-vs-reference coordinate gap sat at the sampler noise floor
(Kabsch X/floor ≈ 1.18) but GB1's (L=56) did not (X/floor ≈ 2.0 on Kabsch RMSD,
≈ 3.6 on 1−distance-matrix-PCC), and that more seeds did not resolve it. pLDDT,
distogram and pTM all agreed to > 0.999. Was GB1 a real, reproducible device-vs-
reference divergence in the ttnn port, or something else?

## Answer

It was a **measurement artifact in the parity harness**, not a port divergence.
`kabsch_rmsd()` (and the distance-matrix-PCC) reduced over **all 448 atom slots in
`sample_atom_coords`, including the 13 padding/non-structure atoms** the model emits
at arbitrary, run-varying positions ~10–11 Å from the origin. The rigid alignment
already used `atom_attention_mask`, but the RMSD *mean* did not, so the padding
atoms dominated it. Because the two backends place those unconstrained atoms
differently, they inflate the cross-backend term X far more than the same-backend
floor R, manufacturing a gap.

Reducing both metrics over the real (masked) atoms only removes the artifact
completely. GB1's gap collapses into the noise floor, and GB1 is then no worse than
trp-cage — in fact tighter.

The fix (score real atoms only) is in `scripts/esmfold2_e2e_parity.py`
(`kabsch_rmsd`, `pair_metrics`).

## Evidence

**Reconciliation (identical saved coords, single pair).** The old whole-atom method
reproduces the published number exactly, and masking removes it:

| pair | metric | all atoms | masked (real) |
|---|---|---|---|
| GB1 dev-vs-ref (X) | Kabsch RMSD | 2.954 Å (= published 2.95) | 0.238 Å |
| GB1 ref-vs-ref (R) | Kabsch RMSD | 1.240 Å | 0.322 Å |
| GB1 dev-vs-ref (X) | 1 − dm-PCC | 0.0684 (= published 3.60×) | 0.0008 (PCC 0.9992) |

**Corrected 3-seed R/D/X (real atoms), from the patched harness.** A third target,
ubiquitin (L=76), was added (the host was free this time):

| protein | L | metric | dev-vs-ref X | ref-floor R | dev-floor D | X/floor | within floor |
|---|---|---|---|---|---|---|---|
| trp-cage | 20 | Kabsch RMSD (Å) | 0.605 | 0.505 | 0.165 | 1.20 | yes |
| trp-cage | 20 | 1 − dm-PCC | 0.0073 | 0.0066 | 0.0006 | 1.11 | yes |
| GB1 | 56 | Kabsch RMSD (Å) | 0.332 | 0.295 | 0.182 | 1.13 | yes |
| GB1 | 56 | 1 − dm-PCC | 0.00078 | 0.00061 | 0.00033 | 1.27 | marginal* |
| ubiquitin | 76 | Kabsch RMSD (Å) | 0.755 | 0.921 | 0.232 | 0.82 | yes |
| ubiquitin | 76 | 1 − dm-PCC | 0.0034 | 0.0047 | 0.00035 | 0.73 | yes |

\* GB1's 1−dm-PCC ratio is 1.27 only because the reference floor itself is tiny
(0.0006); the absolute agreement is PCC 0.9992. It is noise-floor-level.

Reads directly off this table:

- **No length effect.** Parity improves with length, opposite to a length-dependent
  bug: X/floor is 1.20 (L=20) → 1.13 (L=56) → 0.82 (L=76). Ubiquitin, the longest,
  has the device *more* consistent with the reference than the reference is with
  itself.
- **The published GB1-vs-trp-cage ordering reverses.** On real atoms GB1 (1.13) is
  tighter than trp-cage (1.20). GB1 only looked anomalous because its real structure
  is so well reproduced (floor R as low as 0.0006 on dm-PCC) that a fixed amount of
  padding noise stood out more against it.
- **The device is far more self-consistent than the reference** (D ≪ R everywhere).
  If either sampler were "wrong", it is the noisier torch reference, not the port.

## Per-residue localization (Kabsch-aligned, real atoms)

`scripts/esmfold2_gb1_localize.py` folds each protein at 3 seeds on both backends,
dumps per-atom coords, and computes per-residue Kabsch-aligned deviation profiles
for the R/D/X legs. If the gap were a systematic module bug it would concentrate in
a specific structured region (e.g. GB1's β-sheet core); if it were benign sampler
diversity it would track the reference's own seed-to-seed variance and the model's
low-confidence regions.

The data says diffuse and benign:

- **corr(X-profile, R-profile) = +0.80 (GB1), +0.78 (trp-cage)** — the residues
  where the device differs from the reference are the same residues where the
  reference differs from *itself*. Same floppy regions in both backends.
- **corr(X-profile, pLDDT) = −0.51 (GB1), −0.79 (trp-cage)** — divergence
  concentrates in low-confidence residues, as expected for sampler diversity.
- By GB1 secondary structure the profile is flat (X ≈ 0.1–0.34 Å across β1, β2,
  helix, β3, β4); the rigid β-sheet core is not a hotspot. The single elevated
  residue (res 42, start of β3, X/floor ≈ 2.3, pLDDT 0.81) is isolated and
  low-confidence, not a contiguous structured region.
- Estimated systematic dev-vs-ref mean-structure offset on real atoms: 0.15 Å (GB1),
  0.33 Å (trp-cage), ~0 for ubiquitin (X < R). Sub-Ångström, negligible.

## Verdict

Not a port bug. The ttnn ESMFold2 folding port reproduces the torch reference
coordinates to within the reference's own diffusion-sampler noise floor across
L = 20 / 56 / 76, on both an alignment-based (Kabsch RMSD) and an alignment-free
(distance-matrix PCC) metric. The GB1 signal was padding atoms being scored; the
residual is benign, confidence-linked sampler diversity that the reference exhibits
more strongly than the device.

## For Moritz

The customer-facing `docs/pharma-benchmark.md` ESMFold2 section previously disclosed
GB1 as "a real, disclosed, reproducible gap above the floor." That disclosure was
driven by the artifact and has been rewritten: with the metric corrected, all three
targets sit at the noise floor and the honest claim is "device reproduces the
reference within the sampler noise floor" (pLDDT/distogram/pTM > 0.999 as before).
The harness fix and the regenerated `esmfold2.json` are on this branch, unmerged,
for your review.

Operational note from this run: the assigned card (physical 1) wedged with a stuck
ethernet core (`Timed out while waiting for active ethernet core (x=31,y=25)`) that
two `tt-smi -r` resets did not clear; this investigation ran on a free healthy card.
That card likely needs a power-cycle-level recovery or pulling from rotation.
