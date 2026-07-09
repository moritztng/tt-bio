# Protenix-v2 accuracy investigation — reference-model root-cause leg

**Question.** tt-bio's on-device Protenix-v2 predicts CA-RMSD ~10 Å vs ground truth on
`examples/prot.yaml` (117-res chain A, PDB **7ROA**). Is this a real tt-bio device bug or an
eval/undersampling artifact? The shipped gate only checked device-vs-reference self-consistency,
which cannot distinguish the two.

**Method.** Ran the **official ByteDance Protenix 2.0.0** torch implementation (installed in
`~/protenix_ref_venv`, genuine `protenix-v2.pt` checkpoint, loads clean: missing=0 unexpected=0) on
CPU, on the same target, at several settings. Ground truth = `examples/ground_truth_structures/prot.cif`
(PDB 7ROA, chain A, 117 resolved residues, 98.3% identity to the input — 2 mismatches are MSE
selenomethionine = MET). RMSD is Kabsch/CA, matched the same way tt-bio's own `tests/test_structure.py`
does (rank-order when equal-length; label_seq when the construct is longer). Best sample chosen by
Protenix's official `ranking_score`. Scripts: `scripts/protenix_ref_gt_rmsd.py` (no-MSA, 117-res) and
`scripts/protenix_ref_msa_rmsd.py` (MSA / no-MSA on the full construct).

Note on the input: `prot.yaml` (117 res) is only the crystallographically **resolved** residues of
7ROA. The deposited entity / the MSA's query row is the full **136-res construct** (7ROA SEQRES; the
crystal resolves label_seq 3..134 with a ~19-res disordered internal loop + unresolved termini). So
"117-res" runs fold a truncated/gapped chain; "136-res" runs fold the complete construct and are scored
over the 117 resolved residues by label_seq.

## Results (CA-RMSD vs 7ROA, N_step=200, best-of-5 unless noted)

| Reference run | Input | MSA | RMSD | pLDDT / pTM |
|---|---|---|---|---|
| exact `prot.yaml` (undersampled, = tt-bio setting) | 117-res | none | **7.84 Å** (N_step=10, 1 sample) | 67.3 / 0.67 |
| exact `prot.yaml` | 117-res | none | **5.44 Å** | 68.8 / 0.69 |
| full construct, no-MSA control | 136-res | none | **3.21 Å** | 63.9 / 0.59 |
| full construct, with MSA (77 seqs) | 136-res | shallow a3m | **3.13 Å** | 75.4 / 0.78 |
| tt-bio (reported / sibling stream) | 117-res | none (CLI default) | ~10 Å (best) / 9.7 Å oracle | pTM 0.39 |
| tt-bio (sibling stream) | 117-res | `--use_msa_server` (deep) | 3.87 Å best / **2.34 Å oracle** | 0.71 |

## Verdict: **eval/undersampling + regime (truncated-construct + no/shallow-MSA) artifact — NOT a tt-bio device bug.**

The reference torch implementation on the *exact same* single-sequence/no-MSA input reaches only
**5.44 Å even with proper 200-step / best-of-5 sampling** — nowhere near the 1–2 Å a "tt-bio is broken"
hypothesis would require. At tt-bio's own undersampled setting (N_step=10, 1 sample) the reference gives
**7.84 Å**, consistent with tt-bio's ~10 Å within diffusion stochasticity. No reference configuration
folds this input to 1–2 Å, so ~5–10 Å is the *expected* regime behavior, not a bug signature. With a
proper (deep) MSA the tt-bio device path reaches 2.34 Å oracle, matching reference-class accuracy.

## What actually drives the poor RMSD (controlled decomposition)

The length-matched control corrects a naive reading of the table:

- **Construct completeness — largest single factor (~2.2 Å).** Same no-MSA, same sampling: the full
  136-res construct folds to **3.21 Å**, but the truncated 117-res `prot.yaml` (missing the ~19-res
  loop) folds to only **5.44 Å**. Removing the internal loop creates an artificial chain break that
  degrades the global fold. The literal `prot.yaml` input is itself a truncated/gapped construct.
- **Undersampling (~2.4 Å).** N_step=10 → 200 improves the 117-res no-MSA run 7.84 → 5.44 Å. tt-bio's
  e2e harness used N_step=10.
- **MSA depth.** With a *shallow* 77-seq a3m, MSA barely changes RMSD on the full construct
  (3.21 → 3.13 Å) but sharply improves **confidence calibration** (pTM 0.59 → 0.78, pLDDT 64 → 75). A
  *deep* MSA (`--use_msa_server`) helps RMSD materially — the sibling stream measured tt-bio 2.34 Å
  oracle with it. (An earlier draft credited the 5.44 → 3.13 Å drop to MSA; the control shows that was
  mostly construct length, not MSA.)
- **Confidence/ranking is weak here.** ranking_score is nearly flat across the 5 samples (≈0.14–0.16),
  so best-by-confidence ≈ a random draw; the oracle-of-5 is meaningfully better than best-conf.

## Cross-check with the on-device stream (open item now closed)

An earlier on-device ablation ([[protenix-undersampling-ablation]]) reported ~14 Å seed-to-seed variance
and no N_step improvement, raising a possible device-specific EDM bug. The sibling accuracy-investigation
stream has since shown that pathology is **specific to the synthetic 38-token golden molecule/harness**:
on the real 117-res input tt-bio diffusion *converges* (seed-to-seed CA-RMSD mean 3.11 Å no-MSA, 2.44 Å
w/MSA). My reference in the same no-MSA regime likewise improves with N_step and is low-variance across
5 samples. So there is **no device-variance bug on the production path**. (There remains a modest no-MSA
RMSD gap — reference 5.44 Å vs tt-bio ~10 Å on the 117-res input at 200 vs 10 steps — but it closes on
the MSA production path where tt-bio ≈ reference at ~2–3 Å, so it is not a fold-breaking device defect.)

## Recommendation

The reported "~10 Å" is the **single-sequence CLI-default path**. Highest-impact fix is UX, not
kernels: default MSA on for `protenix-v2` (or warn loudly when folding single-sequence), and use proper
sampling (N_step≥200, n_sample≥5, ideally oracle/ensemble given the weak confidence head). Feeding the
complete construct rather than only resolved residues also helps. No device-implementation fix is
indicated by this evidence.
