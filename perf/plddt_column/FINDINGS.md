# The reported plDDT and the B-factor column agree. The reader did not.

`k10-p2` flagged that tt-bio's reported plDDT was not the mean of the per-atom plDDT it writes
into its own B-factor column: 0.811365 against 0.859296 at 512 aa, 0.918296 against 0.910437 at
298 aa, sign flipping between the two sizes. The leading hypothesis was padding: `plddt_atom`
still carrying pad slots when `.mean()` is taken for the metric while the writer slices to the
real atoms.

That is not what happened. The engine's plDDT report path is exactly consistent, at both sizes,
in both model families, and it matches upstream's convention. The two numbers in the flag came
from different quantities because the fold harness that recorded one of them read the wrong
metrics key.

## Reproduction

Two Boltz-2 folds, tt-bio's torch CPU path, no device, 10 sampling steps and 0 recycles (the
defect class is indexing, so it does not need the production step count), at both sizes the flag
quoted:

| | 298 aa | 512 aa |
|---|---|---|
| `complex_plddt` | 0.444008 | 0.410867 |
| mean CA B-factor of the CIF that fold wrote | 0.444008 | 0.410867 |
| **gap** | **0.0** | **0.0** |
| mean all-atom B-factor of the same column | 0.446019 | 0.412284 |
| `confidence_score` | 0.465067 | 0.401478 |
| gap to the CA reading if a reader takes `confidence_score` instead | +0.021059 | **-0.009389** |
| `plddt` key in `metrics` before the fix | absent | absent |

`out/verify_b2_298.json`, `out/verify_b2_512.json`, 94.6 s and 451.8 s on pc. The reported number
and the column are the same number, exactly, at both sizes. What is 0.009 to 0.021 away is
`confidence_score` — and it flips sign between the two sizes inside this one pair of folds, which
is the signature the flag read as a size-dependent padding ratio.
`(4*0.444008 + 0.549303)/5 = 0.465067` reproduces it from `complex_plddt` and `ptm` exactly.

## The mechanism

Boltz-2 was the only model whose metrics dict had no `plddt` key. `tt_bio/main.py`'s
`write_result` emits `complex_plddt`, while `worker.py`'s `_row` emits both `plddt` and
`complex_plddt` for Protenix-v2 / OpenFold3 / OpenBind-0 / OpenDDE, and ESMFold2 and RF3 emit
only `plddt`. Six perf harnesses carry the identical line

    "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)

(`perf/k10_p2/fold_cpu_ref.py`, `perf/roof_shared/fold_shared.py`, `perf/b2z_levers/fold_ab.py`,
`perf/b2z2_cond/fold_cond.py`, `perf/b2z2_dp/dp_width.py`, `perf/b2z2_qchunk/qchunk_bh.py`), so
for every Boltz-2 fold the fallback fired and `confidence_score` went into the record under the
name `plddt`.

`confidence_score` is `0.8*complex_plddt + 0.2*iptm`, falling back to `ptm` when there is no
interface. On cdk2x2 that term sits ABOVE plDDT at 298 aa (pTM 0.9497) and BELOW it at 512 aa
(0.6130), which is the whole of the flagged 0.008/-0.050 discrepancy and the whole of the sign
flip. Backing the implied term out of the four committed k10-p2 arms gives 0.9497, 0.8264,
0.6130 and 0.4873 — all in range and ordered by arm quality, none of them a plDDT.

## Both families are internally consistent, and match upstream

Boltz-2 writes one plDDT per residue, broadcast to that residue's atoms, so its reported value is
the CA mean; the all-atom mean of the same column is atom-count weighted and sits 0.002 higher.
Protenix-v2 / OpenFold3 / OpenBind-0 / OpenDDE write a genuine per-atom column and report its
all-atom mean, 0.025 BELOW the CA mean on the same fixture. Reading the CA mean against a
per-atom report, or the reverse, is a 0.025 error on a correct fold.

| evidence | folds | reading | worst gap |
|---|---|---|---|
| upstream `boltz==2.2.1` reference fixtures, 4 targets | 27 | CA mean | 1e-6 |
| upstream Protenix-v2 reference fixtures | 12 | all-atom mean | 2e-6 |
| upstream OpenBind reference fixtures | 10 | all-atom mean | 1e-6 |
| tt-bio OpenFold3 device folds, 298 aa and 512 aa (`perf/of3_4xpd`) | 18 | all-atom mean | 0.0 |
| tt-bio Boltz-2 CPU folds, 298 aa and 512 aa (this pass) | 2 | CA mean | 0.0 |
| upstream Boltz-2 GPU folds, its own `complex_plddt` vs its own column (`perf/k10_anchor`) | 10 | CA mean | 1e-7 |

A further 11 upstream Protenix-v2 and OpenDDE reference CIFs carry a flat-zero B-factor column:
those stacks write no plDDT into the structure at all, where our port does.

## The fix

`write_result` mirrors `complex_plddt` into `plddt`, one line in the `_scalars` closure that both
the winner's metrics and every `all_runs` row already go through. That makes `plddt` the one key
every model reports the complex mean under, so all six harnesses read the right quantity without
being touched, and the 298 aa CIF is byte-identical before and after the change
(sha256 `17c8c4e71e8976bb`). `fold_cpu_ref.py` raises instead of substituting, because a harness
that silently returns a different quantity is the defect, not the missing key.

`perf/other512/cif_rmsd.py` gains `bfactor_plddt` (both readings of the column, one parser shared
with `read_atoms` and `mean_ca_bfactor`) and `plddt_column_check`, which matches a reported plDDT
against the closer of the two readings. Deliberately no model list: a report path that leaked
padding, re-ordered a tensor or picked up a different scalar matches NEITHER reading, so the
check catches the class without needing to be told about the next port
(memory `hardcoded-model-list-misses-new-port-recurring`).
`perf/b2z2_fusebias/score.py` emits it beside `plddt_cif`.

Negative control: re-scoring the committed 512 aa set is identical to `out/score_512.json` apart
from the new key, and `plddt_column` fires `ok=false` on all six tt-bio arms recorded through the
old reader (-0.033 to -0.058) while passing all five upstream ones at 0.0. A check that cannot
break on the data that motivated it proves nothing.

## Second finding in the same path: RF3 reported plDDT on the wrong scale

`worker.py`'s RF3 path reported `plddt` as `mean * 100`, justified by a comment that said this is
"the 0-100 scale every other model in this repo reports". That comment is false: Boltz-2 reports
0.444008, OpenFold3 0.577188, Protenix-v2 0.918534, OpenDDE 0.874895, ESMFold2 0.7658 — all 0..1.
RF3 was the only model on 0..100, and the inconsistency was inside a single results entry as well
as across models, since `--early_stop_plddt` is compared against the 0..1 value and the README
documents it that way (`--early_stop_plddt 0.5`). One harness, `perf/fused_sdpa`, recorded
`plddt=81.7155` for RF3 and `plddt=0.344919` for OpenFold3 under the same key.

RF3's metric was consistent with its own column (0.0 gap on both committed 298 aa CIFs, per-atom
reading) — it was the scale that disagreed with everything else. It now reports 0..1 like every
other model, so `mean(B-factor column) / 100` is one invariant with no per-model branch, and the
code matches the README instead of contradicting it. **This changes a user-facing number**: an
RF3 `results.json` entry that read `"plddt": 81.7155` now reads `"plddt": 0.817155`. No
coordinate moves.

## What this does not touch

No coordinate, no model output, no accuracy-gate digest. The standing CA-lDDT deficit k10-p2
measured is unrelated: it comes from coordinates scored against 1HCL, never from this reported
metric, and the plDDT rows in k10-p2's own tables were already `plddt_cif`, the column. Only the
paragraph reading the reported number was wrong, and it is corrected in place.

Boltz-2's `_sample_scalars` at `tt_bio/worker.py:865` is ESMFold2's, not Boltz-2's. It reports
`s.plddt.mean()` under `plddt`, so it has the key and cannot hit the fallback; ESMFold2 has no
CPU path, so its column was not re-measured here.

## Repro

No device. `PYTHONPATH` must point at this checkout, not the venv's installed package.

```
PYTHONPATH=$PWD python3 perf/plddt_column/verify_plddt_column.py \
    --out perf/plddt_column/out/verify_b2_298.json --models boltz2 \
    --fixtures cdk2x2_298 --steps 10 --recycles 0 --keep-cif perf/plddt_column/cif
```

Protenix-v2, OpenFold3, OpenBind-0, OpenDDE and RF3 are Tenstorrent-only (`load_model` refuses
them on a CPU worker), so their legs are committed device folds and upstream reference fixtures
rather than fresh folds. `--sweep` checks all of them plus the negative control in one pass, folds
nothing, and exits non-zero on a mismatch or on the check going blind:

```
PYTHONPATH=$PWD python3 perf/plddt_column/verify_plddt_column.py \
    --sweep --out perf/plddt_column/out/sweep.json
```

83 folds, 0 mismatches, worst gap 2e-6, and `confidence_score` rejected on all three folds it is
fed to. `out/sweep.json` carries the per-fold rows.
