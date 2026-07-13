# tt-bio Release-Gate Verification — 2026-07-13 (~20:30–21:30 UTC+2)

Branch: `wk/tt-bio-release-gate-verify` (off main tip at time of run).
Hardware: qb2, P300 Blackhole. Card 0 = accuracy gate + perf; card 1 = pytest recheck; card 2 = UX.
Goal: confirm main is release-gate-clean before Moritz cuts v0.2.6. **No version bump, no tag.**

## Verdict (this turn)

**NOT fully release-gate-clean.** 3 of 4 legs PASS; pytest has 1 real failure; UX final pending relaunch.

| Leg | Status | Numbers |
|-----|--------|---------|
| 1. Accuracy gate (`release_gate.py`) | **PASS** (exit 0) | see below |
| 2. Perf regression (`perf_regression.py`) | **PASS** (clean re-measure) | see below |
| 3. UX regression (`ux_regression.py`) | **IN PROGRESS** | restarted on card 2 after a wedge; reached boltz2 trpcage phase; final pending relaunch |
| 4. Pytest suite | **FAIL** (1 real) | `test_of3_pairformer_block0_on_device` FAIL (KeyError, gold-fixture); `test_protenix_seqfold` contention-only (PASSED on quiet card) |
| 5. `docs/pharma-benchmark.md` consistency | **PASS** | R/D/X spot-checks consistent; Status section has nothing pending |

**Blocker:** `tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device` fails
deterministically on main tip (reproduced on a quiet card 1, not contention). At line 71 the test
does `pickle.load(open(_GOLD,"rb"))["intermediates"]["pairformer_block0"]` and the loaded gold
pickle has no `pairformer_block0` key under `intermediates` → `KeyError`. This is a gold-fixture /
intermediate-recording mismatch (the `_GOLD` pickle exists and has `intermediates`, but the
`pairformer_block0` entry is absent). Regression status vs v0.2.5 was not determined this turn —
needs a `v0.2.5` checkout run of the same test to decide new-vs-preexisting. Flagged for Moritz /
orchestrator: either regenerate the gold fixture with the `pairformer_block0` intermediate, or fix
the intermediate-recording path, or mark the test xfail if the per-block gate is no longer the
intended signal.

## Leg 1 — Accuracy gate (`TT_VISIBLE_DEVICES=0 python scripts/release_gate.py`)

**GATE_EXIT=0 — PASS.** All fold models cleared parse + ground-truth floor; BoltzGen cleared
designability; ESMC cleared the fused-RoPE PCC floor.

### Fold (prot/7ROA, 200 steps / 5 samples, seed 0)
| model | RMSD (Å) | TM | floor | wall | result |
|-------|----------|------|-------|------|--------|
| boltz2 | 1.845 | 0.890 | <=3.0/>=0.75 | 75s | PASS |
| esmfold2 | 1.789 | 0.908 | <=4.0/>=0.65 | 72s | PASS |
| esmfold2-fast | 1.694 | 0.911 | <=4.5/>=0.6 | 30s | PASS |
| protenix-v2 | 1.428 | 0.947 | <=6.0/>=0.5 | 72s | PASS |

### BoltzGen designability (binder.yaml, 4 designs, protein-anything)
| model | scRMSD (Å) | pass rate | floor | wall | result |
|-------|-----------|-----------|-------|------|--------|
| boltzgen | 0.741 | 100% | <=2.0Å/>=50% | 244s | PASS |

### ESMC embedding parity (fused-RoPE shipped path, PCC floor 0.99)
| model | per-res PCC | pooled | logits | argmax | wall | result |
|-------|------------|--------|--------|--------|------|--------|
| esmc-300m | 0.99961 | 0.99993 | 0.99990 | 1.0000 | 5s | PASS |
| esmc-600m | 0.99964 | 0.99989 | 0.99996 | 1.0000 | 8s | PASS |

Note: the gate required a P300 mesh-graph-descriptor fix in `scripts/release_gate.py` and
`tt_bio/main.py`'s `gen` path (a lone P300 Blackhole chip is a custom topology; ttnn refuses to
open it without a 1x1 MGD). The in-process ESMC embed leg and the BoltzGen subprocess inherited the
env var set by the gate wrapper. Both fixes are committed on this branch.

## Leg 2 — Perf regression (`python scripts/perf_regression.py`)

**PASS (clean re-measure).** Per-model warm latency vs `docs/perf_baselines.json` (v0.2.5):

| model | latency (s) | Δ vs baseline | verdict |
|-------|------------|---------------|---------|
| boltz2 | 1.427 | -4.7% | within noise |
| esmfold2 | 2.267 | -6.1% | within noise |
| esmfold2-fast | 2.959 | -5.4% | within noise |
| protenix-v2 | 3.444 | -0.2% | within noise (clean re-measure) |
| opendde | 2.654 | -1.1% | within noise |
| esmc-600m | 24.78 | +12.1% | within ±15% band |

All six within the ±15% noise band. **Important caveat:** the first full concurrent run (4 gate
legs launched simultaneously across 4 cards) reported protenix-v2 at -64% — that was a
**contention artifact** (host CPU / disk / download contention from the 4-way concurrent launch),
not a real regression. A clean re-measure of protenix-v2 alone on a quiet card gave 3.444s (-0.2%).
The lesson: **do not launch perf_regression concurrently with other device jobs** — its numbers are
only valid on a quiet machine. (RELEASING.md should state this; flagged below.)

## Leg 3 — UX regression (`python scripts/ux_regression.py`)

**IN PROGRESS — final pending relaunch.** The first launch (concurrent with the other 3 legs on
card 2) stalled in a predict subprocess (card contention / wedge). It was killed and restarted
on card 2 alone. The restart has reached the boltz2 trpcage predict phase (recyc=2, steps=4,
samples=1). It still needs to complete the esmfold2 and protenix phases plus the CLI/help and
output-file-parse checks. A relaunch should let it finish (~10 min) and capture UX_EXIT. `ux_regression.py`
runs its folds via subprocess in unique temp dirs (`out_<model>/boltz_results_trpcage/`), so it does
not conflict with other runs — it does need a free device (it opens one for the predict folds).

## Leg 4 — Pytest suite

Ran the full suite concurrently first (contention). 2 failures:
- `tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device` — **FAIL (real).**
  Reproduced on a quiet card 1. `KeyError: 'pairformer_block0'` at line 71
  (`pickle.load(open(_GOLD,"rb"))["intermediates"]["pairformer_block0"]`). The gold-intermediates
  pickle is missing the `pairformer_block0` entry. Gold-fixture / intermediate-recording mismatch.
  This is the release-gate blocker.
- `tests/test_protenix_seqfold.py::test_protenix_sequence_to_structure` — **contention only.**
  Timed out at 600s in the concurrent run; on a quiet card 1 it **PASSED in 68.65s**. Not a real
  failure.

Recheck command (card 1): `TT_VISIBLE_DEVICES=1 python -m pytest
tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device
tests/test_protenix_seqfold.py::test_protenix_sequence_to_structure -v --tb=short`
→ `1 failed, 1 passed in 68.65s`, RECHECK_EXIT=1.

## Leg 5 — `docs/pharma-benchmark.md` consistency

**PASS.** Spot-checked R/D/X entries against committed data files (`boltz2.json`, `esmfold2.json`,
`protenix-v2.json`, `opendde.json`, `esmc-300m.json`, `esmc-600m.json`). Found and fixed 2 nits
on this branch:
1. OpenDDE production-settings entry in the Status section had R and X swapped — corrected.
2. ESMC-300m PCC range was "0.9988–0.9996"; tightened to "0.9987–0.9996" to match the committed
   data (0.9987 is the actual low). Applied in the Status section, the main results table, and the
   body text.

Status section now has nothing "in progress". `pharma_parity.py` was NOT re-run (per task).

## RELEASING.md note (flagged, not edited this turn)

While running, I noticed RELEASING.md does not warn that `perf_regression.py` must run on a quiet
machine (no concurrent device jobs) or its numbers are invalid. The -64% protenix false-regression
this turn came from exactly that. A one-line addition to the perf leg of the release checklist
would prevent a future false FAIL. Left for a follow-up edit (did not want to expand scope on a
turn that already found a real blocker).

## Fixes committed on this branch

- `scripts/release_gate.py`: set `TT_MESH_GRAPH_DESC_PATH` for P300 (in-process ESMC embed +
  BoltzGen subprocess inherit it).
- `tt_bio/main.py` `gen`: same P300 MGD setup so standalone `tt-bio gen` works on P300.
- `docs/pharma-benchmark.md`: OpenDDE R/X swap fix + ESMC-300m PCC range tightening.
- `docs/pharma-benchmark-data/release-gate-verify-2026-07-13.md`: this report.

## Relaunch instructions (to finish)

1. Let `ux_regression.py` finish on a quiet card 2; record UX_EXIT + per-model phase advances;
   update the Leg 3 row here.
2. Decide the pairformer blocker: checkout v0.2.5, run
   `test_of3_pairformer_block0_on_device` — if it also fails there, it's preexisting (test-infra,
   not a main regression); if it passes, main regressed it. Either way the gold fixture or the test
   needs fixing before v0.2.6.
3. Optionally add the "perf must run on a quiet machine" line to RELEASING.md.
4. Commit + push (orchestrator pushes this branch to origin).
