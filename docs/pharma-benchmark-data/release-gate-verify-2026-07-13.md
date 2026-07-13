# tt-bio Release-Gate Verification — 2026-07-13 (~20:30–22:40 UTC+2)

Branch: `wk/tt-bio-release-gate-verify` (off main tip at time of run).
Hardware: qb2, P300 Blackhole. Card 0 = accuracy gate + perf + clean UX re-run; card 1 = pytest recheck; card 2 = UX first attempt.
Goal: confirm main is release-gate-clean before Moritz cuts v0.2.6. **No version bump, no tag.**

## Verdict

**NOT release-gate-clean — one real blocker.** 4 of 5 legs PASS; the pytest leg has one real,
deterministic test failure (`test_of3_pairformer_block0_on_device`, a stale gold-fixture mismatch
regressed on main after v0.2.5). It is a test-infra issue, not an accuracy regression (the pairformer
port is verified by the stronger `prefix47` gate, which passes) — but it is still a failing test on
main and should be fixed before v0.2.6.

| Leg | Status | Numbers |
|-----|--------|---------|
| 1. Accuracy gate (`release_gate.py`) | **PASS** (exit 0) | boltz2 1.845Å/0.890, esmfold2 1.789/0.908, esmfold2-fast 1.694/0.911, protenix-v2 1.428/0.947, BoltzGen scRMSD 0.741Å/100%, ESMC-300m PCC 0.99961, ESMC-600m 0.99964 |
| 2. Perf regression (`perf_regression.py`) | **PASS** (clean) | boltz2 -4.7%, esmfold2 -6.1%, esmfold2-fast -5.4%, protenix-v2 -0.2%, opendde -1.1%, esmc-600m +12.1% — all within ±15% |
| 3. UX regression (`ux_regression.py`) | **PASS** (clean re-run) | all 6 surfaces clear: boltz2 11s, esmfold2 23s, esmfold2-fast 23s, protenix-v2 10s, opendde 12s, esmc-600m 6s (UX0_EXIT=0) |
| 4. Pytest suite | **FAIL (1 real)** | `test_of3_pairformer_block0_on_device` FAIL (KeyError `pairformer_block0`, gold-fixture, real — reproduced on quiet card 1); `test_protenix_seqfold` contention-only (PASSED 68.65s on quiet card) |
| 5. `docs/pharma-benchmark.md` consistency | **PASS** | R/D/X spot-checks consistent; Status section has nothing pending |

**The blocker (Leg 4):** `tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device`
fails deterministically on main tip (reproduced on a quiet card 1, not contention). At line 71 the
test does `pickle.load(open(_GOLD,"rb"))["intermediates"]["pairformer_block0"]` with
`_GOLD = ~/of3_ref_out.pkl` (a local, untracked file). The pickle loads and has an `intermediates`
dict, but the `pairformer_block0` entry is absent → `KeyError`.

**Root cause / regression status:** commit `9347d39` (2026-07-12, "port(openfold3): P4 —
real-distribution golden, honest stack re-gate, MSA-block remap") regenerated the gold pickle with a
real distribution and dropped/renamed the `pairformer_block0` intermediate that the older P0+P2
block0 test (line 71) still expects. `9347d39` is **after** the v0.2.5 tag (2026-07-11), so **v0.2.5
did not have this failure — main regressed it.** The pairformer port itself is verified by the
stronger `test_of3_pairformer_stack_prefix47_on_device` gate (s_pcc>0.98, z_pcc>0.97 on real input
through 47/48 blocks) added by P4/P5, which does not fail — so this is a stale-test-fixture
mismatch, not an accuracy regression. Fix options for Moritz: (a) update the block0 test to the new
intermediate key, (b) mark it xfail/skip if the per-block gate is superseded by the prefix47/real-
stack gates, or (c) have the golden script emit `pairformer_block0` again. Recommended: (b) — the
prefix47 gate is strictly stronger and the block0 test is now redundant.

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
open it without a 1x1 MGD). The in-process ESMC embed leg and the BoltzGen subprocess inherit the
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

All six within the ±15% noise band. **Caveat:** the first full concurrent run (4 gate legs launched
simultaneously across 4 cards) reported protenix-v2 at -64% — that was a **contention artifact**
(host CPU / disk / download contention from the 4-way concurrent launch), not a real regression. A
clean re-measure of protenix-v2 alone on a quiet card gave 3.444s (-0.2%). Lesson: perf_regression
numbers are only valid on a quiet machine — worth a one-line warning in RELEASING.md.

## Leg 3 — UX regression (`python scripts/ux_regression.py`)

**PASS (clean re-run, UX0_EXIT=0).** Two attempts:
- **Card 2 (first attempt, concurrent with other legs):** 5 of 6 surfaces PASS, but boltz2 predict
  timed out after 900s → UX_EXIT=1. The other 5 models all ran fast (esmfold2 23s, esmfold2-fast
  23s, protenix-v2 10s, opendde 11s, esmc-600m 6s). boltz2 was the first predict after a restart on
  a contended card — cold-start weight load/compile on a busy card hit the 900s timeout. Contention
  artifact, not a real UX regression.
- **Card 0 (clean re-run, quiet, boltz2 weights cached from the gate run):** all 6 surfaces PASS,
  UX0_EXIT=0. boltz2 11s, esmfold2 23s, esmfold2-fast 23s, protenix-v2 10s, opendde 12s, esmc-600m
  6s. CLI/help checks pass; output-file parse + results.json/manifest shape checks pass for every
  model.

`ux_regression.py` runs its folds via subprocess in unique temp dirs (`out_<model>/...`), so it does
not conflict with other runs — but its first-predict-per-model can stall on a contended card. Run it
on a quiet card with cached weights.

## Leg 4 — Pytest suite

Full suite ran concurrently first (contention). 2 failures observed:
- `tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device` — **FAIL (real).**
  Reproduced on a quiet card 1. `KeyError: 'pairformer_block0'` at line 71. Stale gold-fixture
  mismatch regressed on main by `9347d39` (P4, after v0.2.5). Port verified by the stronger
  `prefix47` gate (passes). **This is the release-gate blocker.** See Verdict.
- `tests/test_protenix_seqfold.py::test_protenix_sequence_to_structure` — **contention only.**
  Timed out at 600s in the concurrent run; on a quiet card 1 it **PASSED in 68.65s**.

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

## Fixes committed on this branch

- `scripts/release_gate.py`: set `TT_MESH_GRAPH_DESC_PATH` for P300 (in-process ESMC embed +
  BoltzGen subprocess inherit it).
- `tt_bio/main.py` `gen`: same P300 MGD setup so standalone `tt-bio gen` works on P300.
- `docs/pharma-benchmark.md`: OpenDDE R/X swap fix + ESMC-300m PCC range tightening.
- `docs/pharma-benchmark-data/release-gate-verify-2026-07-13.md`: this report.

## RELEASING.md note (flagged, not edited this turn)

Two operational caveats found while running the gate that deserve a one-line addition to the
release checklist:
1. `perf_regression.py` must run on a quiet machine (no concurrent device jobs) — its numbers are
   invalid under contention (a 4-way concurrent launch produced a false protenix-v2 -64% this run).
2. `ux_regression.py`'s first predict per model can stall on a contended card (boltz2 hit the 900s
   timeout on a busy card 2 but passed in 11s on a quiet card 0 with cached weights). Run UX on a
   quiet card.

Left for a follow-up edit to keep this verify task scoped to its real finding (the pairformer
blocker).

## Recommended fix for the blocker (for Moritz / a follow-up task)

Mark `test_of3_pairformer_block0_on_device` xfail (or skip) with a reason pointing at P4
`9347d39`: the per-block gate is superseded by `test_of3_pairformer_stack_prefix47_on_device`
(stronger: 47/48 blocks on real input, s_pcc>0.98 & z_pcc>0.97). Alternatively regenerate the gold
pickle to re-emit `pairformer_block0`. Either clears Leg 4 and makes main release-gate-clean.
