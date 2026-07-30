# DockQ cross-validation: shipped labels vs tinyprot

Every AbAg-XM label is DockQ 2.1.3 on the ARK-declared interface
(`scripts/abag_xm_dockq_interface.py`). This cross-validation re-scores a
stratified sample of (target, generator, sample) triples with an independent
implementation, `tinyprot` (github.com/bjing2016/tinyprot, v0.1.0, MIT,
commit `dd2178b650899460223af1a72eb5d85f9ee19619`), on the SAME declared chain
pair, and compares. The two implementations are definitionally identical for
protein-protein pairs (same DockQ formula, same 5.0 A fnat / 10.0 A interface
thresholds, same N,CA,C,O backbone sets, same larger-chain-receptor rule), so
agreement isolates implementation and wiring bugs, not definitional drift.

## Sample

120 triples, seeded (RandomState 20260730): 40 per generator (boltz2,
opendde-abag, protenix-v2) across 83 of the 161 scorable targets
(`docs/abag-xm-dockq-xval-triples.csv`). Stratification: 23 triples in the
0.15-0.30 DockQ band (threshold-flip zone at 0.23), 18 in the 0.70-0.90 band
(0.8 zone), the rest proportional to population decile. Forced coverage, all
present: multi-copy (21av), HL-only antibody (84 triples), scFv (9), peptide
antigen (17), sequence-mismatch target (9mz8), low-DockQ failures <0.05 (41).
The 3 anti-phosphoepitope targets (null DockQ) are excluded by construction.

Chain grids: tinyprot requires identical per-chain atom grids. Model and
native are residue-intersected by SEQUENCE ALIGNMENT (matching blocks, equal
residue names kept) — the same common-residue-universe semantics DockQ v2
implements (`make_compatible` in `scripts/abag_xm_dockq_xval.py`). Residues
dropped: 0 on 119/120 triples (see the one below).

## Sanity gate (1A2K)

DockQ's shipped example pair (1A2K r/l + model), both implementations:
DockQ 2.1.3 = 0.94253994, tinyprot = 0.94254001, |delta| = 5.6e-8 (< 1e-3
gate). Gate PASS before any campaign scoring.

## Results (120/120 triples scored, `docs/abag-xm-dockq-xval.csv`)

| metric | value |
|---|---|
| Pearson r (DockQ) | 0.99968 |
| Spearman rho | 0.99985 |
| median abs delta | 2.8e-8 |
| p95 abs delta | 1.2e-6 |
| mean signed delta (tinyprot - v2) | +0.00074 |
| max abs delta | 0.089 (one triple, explained below) |
| threshold flips @ 0.23 | 0 |
| threshold flips @ 0.8 | 0 |

119 of 120 triples agree to <= 2.4e-6 — float-level agreement between two
independent codebases on the same residue universe. There is no systematic
offset (mean signed delta +0.0007, driven entirely by the single deviation).

## The one deviation: 9q1l / opendde-abag / rank 12 (|delta| = 0.089)

v2 = 0.0786, tinyprot = 0.1676. Component split: fnat 0.0 vs 0.286 (the whole
delta is the fnat/3 term); iRMSD 7.47 vs 9.28; LRMSD 17.16 vs 17.45.

This triple exposed a real label bug — NOT in either DockQ implementation but
in the chain-assignment wrapper. 9q1l is a 2:2:2 assembly (two Fab copies, two
peptide-antigen copies); the ARK-declared interface is auth chain E (LIGHT
chain, copy 2) x auth F (antigen). The fold's model carries one Fab + antigen.
The shipped labels for opendde-abag AND protenix-v2 9q1l map native E to the
model's HEAVY chain (B) instead of its light chain (C) — a cross-type
assignment. Both implementations then score a light-vs-heavy Ig-framework
alignment (142 of 214 residues dropped by the sequence intersection, hence
the fnat disagreement) and both return garbage-failure values.

Root cause: `_build_seq_map`'s fallback in `abag_xm_dockq_interface.py`. Model
CIFs from OpenDDE/Protenix carry no `_entity_poly` sequences, so the wrapper
falls back to raw CA-atom-count proximity. Heavy (221 CA) and light (214 CA)
chains differ by ~3%, and native CA counts run high (219 for a 214-residue
chain), so the heavy chain wins the argmax for every antibody chain in the
native. Boltz2/esmfold2 CIFs carry real sequences and map correctly
(light x light; all 50 samples DockQ >= 0.23 in both).

Correct-interface score for this triple, recomputed with DockQ 2.1.3 on the
declared pair (native E x F vs model C x A): **DockQ = 0.9625, fnat = 1.0** —
a near-perfect fold mislabeled as a 0.079 failure. The threshold-flip count
above stays 0 only because both implementations agree on the (wrong) chain
assignment they were handed; against the TRUE interface this sample flips.

Blast radius and remediation: full-dataset chain-map audit in
`scripts/abag_xm_chainmap_audit.py`. A pre-patch audit of every target with
near-duplicate native chains (29 targets x both sequence-less-CIF generators,
`docs/abag-xm-chainmap-audit-atrisk.csv`) flagged exactly the two 9q1l folds.
Their labels were recomputed with DockQ 2.1.3 on the sequence-corrected
assignment through the existing label-patch machinery
(`scripts/abag_xm_chainmap_patch.py`; `_build_seq_map` now derives sequences
from `_atom_site` when `_entity_poly` is absent and matches unequal lengths by
alignment identity). The post-patch full-dataset audit
(`docs/abag-xm-chainmap-audit.csv`, 483 folds, 966 side-rows) finds zero
remaining cross-type assignments; the largest residual correspondence gap
anywhere is 1 residue. Post-patch both 9q1l folds score 50/50 DockQ >= 0.23
(opendde max 0.973, protenix-v2 max 0.987) — both generators had in fact
nailed the target. Single-copy folds reproduce shipped values bit-for-bit
(21du 0.658683, 9u5r 0.886619). See the dataset card "Label provenance"
section.

## Verdict

The DockQ 2.1.3 implementation used for every shipped label is numerically
validated against an independent implementation: r = 0.9997, zero threshold
flips, no systematic offset, and the single deviation fully explained to
component level (wrapper chain-map, not DockQ numerics). Labels regenerated
for the folds the audit flags are re-validated by the same harness.
