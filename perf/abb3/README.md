# ABodyBuilder3 evaluation instrument

`tt_bio/antibody_rmsd.py` computes the per-region antibody Fv RMSDs that the ABodyBuilder3
reproduction is scored against. It is validated by recomputing Kenlay et al.'s own released
per-structure numbers from their own released structures, before it is ever pointed at a model of
ours. The scripts here are that validation.

Source: Kenlay et al., *Bioinformatics* 40(10):btae576, doi `10.1093/bioinformatics/btae576`.
Structures and per-structure evaluations at Zenodo `10.5281/zenodo.11354577` (CC-BY-4.0).

## Running it

Fetch and unpack `output.tar.gz` (440,793,236 B), then:

```
python perf/abb3/verify_instrument.py <output-dir>   # per-structure agreement, 1,236 evaluations
python perf/abb3/check_numbering.py   <output-dir>   # the region labels really are IMGT
python perf/abb3/paper_table.py       <output-dir>   # which directory is which published column
```

All three exit non-zero if they stop agreeing. No device, no card lease, CPU only.

## What the validation established

`verify_instrument.py` reproduces every released per-structure value across all four variants and
both refinement backends to **0.00843 A** worst case, against a 0.01 A bar. `base-loss` returns
2.714 A on CDR-H3 over the 250-structure set, which is the value the reproduction is
pre-registered against. 249 of 250 structures agree to 0.00001 A; the entire residual comes from
one structure, `5wn9_H0-H1`, whose light chain differs at the 5e-4 level for reasons that are not
on our side (the atom sets and the residue pairing are identical, so their scored light-chain
coordinates for that one entry simply differ from the ones they published).

Three conventions carry the result, and each of them silently corrupts every per-region number if
it is wrong.

**"Backbone" is N, CA, C, CB.** Upstream's `extract_backbone_coordinates` slices
`positions[:, :, :4]` and documents its input as "(B, n, 14/37, 3)". Under atom14 that slice is
N, CA, C, O; under atom37 it is N, CA, C, CB. The published evaluation used atom37, so the
carbonyl O is absent, CB is present, and glycine contributes three atoms rather than four.
Scoring the true backbone instead moves CDR-H3 by 0.07 A and the worst per-structure column by
0.64 A, which `verify_instrument.py` runs as a control: if that control ever passes, the
agreement above is insensitive to the atom set and proves nothing.

**Superposition is per chain, over that chain's whole backbone.** Heavy and light are superposed
onto the truth independently, and every region of a chain is measured under its own chain's
transform. Framework-only superposition, the other common convention, is not what they did.

**Residues pair through their index into the region list, not their residue number.** Their
inference stage numbers heavy `1..n_h` and light `501..`; the ABodyBuilder2 baseline structures
carry real IMGT numbering with insertion codes and restart the light chain at 1; and their PDB
fixer drops residues with no resolved atoms, so a chain can be a residue short. Matching on
residue number pairs the wrong residues, or none at all.

## Numbering

Regions are ANARCI IMGT numbering with IMGT region boundaries
(`stages/data/generate_data.py:192,395`). They cannot be regenerated: the pipeline calls
`exs.sabdab`, which is Exscientia-internal. They ship instead, per residue, in the released
`<variant>/plddt/<structure>.pt` and in the dataset's own per-structure files.

`check_numbering.py` confirms the scheme without that code. IMGT places CDR3 at positions 105-117,
so it is bracketed by the Cys at 104 and the Trp (heavy) or Phe (light) at 118; Kabat and Chothia
put those boundaries elsewhere. The anchors hold on 250 of 250 structures in both chains.

## Refinement, and which column is which

Their published numbers are post-refinement and there are two backends. `evaluate.csv` is OpenMM
and covers all 250; `evaluate_yasara.csv` is YASARA and covers only the 59 structures with
`refine_yasara/`. `pred/` is the raw prediction run through their PDB fixer, which adds hydrogens
and OXT but moves no heavy atom, so it scores identically to `pred_unfixed/`.

The directory-to-published-column mapping was not settled upstream, and `paper_table.py` settles
it: all 5 rows x 8 columns of the paper's Table 1 reproduce to the 2 decimal places it prints,
worst cell 0.0049 A, on the **59-structure** subset.

| Table 1 row | artefact |
|---|---|
| ABodyBuilder2 | `abodybuilder2-loss/evaluate.csv` |
| Baseline (OpenMM) | `base-loss/evaluate.csv` |
| Baseline (Yasara) | `base-loss/evaluate_yasara.csv` |
| ABodyBuilder3 | `plddt-loss/evaluate_yasara.csv` |
| ABodyBuilder3-LM | `language-loss/evaluate_yasara.csv` |

`abodybuilder2-loss/` is the ABodyBuilder2 tool itself, not a loss variant of ABodyBuilder3: its
`raw/` holds ImmuneBuilder's four-model ensemble output (`rank0_refined.pdb` and friends).

Two consequences for anything downstream. The paper's headline 2.42 A is **YASARA**-refined, and
YASARA is commercial, so that exact column is not reproducible by us; the OpenMM column is, and
that is what the reproduction is pre-registered against. And the 250-structure set is
validation(150) + test(100) combined, not a held-out test set, so a comparison on it is
like-for-like with their published numbers but is not a clean generalisation estimate.
