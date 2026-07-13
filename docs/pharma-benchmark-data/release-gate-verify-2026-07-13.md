# Release-gate verify — 2026-07-13 (qb2)

Branch: `wk/tt-bio-release-gate-verify` (base = main tip `1417981` + 3 verify-side
fixes committed on the branch, HEAD `943728f`). Hardware: qb2 (tt-quietbox2),
Blackhole P300, cards 0-3, env `/home/ttuser/tt-bio/env` (python 3.12, ttnn from
that env). Run by an autonomous tt-bio release-gate verify worker.

## Verdict so far: NOT release-gate-clean

Two of the four gate legs already FAIL (perf, pytest). The accuracy gate and UX
leg are still running (setsid, survived the worker turn) — even if both pass, the
perf + pytest failures block a v0.2.6 cut. **Do not cut v0.2.6 until the
protenix-v2 perf regression and the two test failures are resolved (or the
baselines/tests are updated intentionally with a recorded reason).**

## Verify-side fixes committed on this branch (for the orchestrator to merge)

1. `release_gate: set P300 mesh-graph descriptor for in-process + gen legs`
   (`scripts/release_gate.py`, `tt_bio/main.py:gen`). Main-tip `release_gate.py`
   could not open a lone P300 Blackhole chip for its in-process ESMC embed leg
   (`esmc.embed_sequences` via `load_esmc`, which bypasses the embed CLI) nor for
   the BoltzGen `gen` subprocess — `ttnn.open_device` aborts with
   `Custom fabric mesh graph descriptor path must be specified` on a lone P300
   without `TT_MESH_GRAPH_DESC_PATH`. `perf_regression.py` already set this once
   at top of main; `release_gate.py` did not. Set it once at the top of
   `release_gate.py:main()` so every leg inherits it, and in `main.py:gen` so
   standalone `tt-bio gen` works on a lone P300 too. No accuracy/perf/OOM impact
   (only lets the device be opened on this board topology).
2. `docs(pharma): fix swapped R/X in OpenDDE production Status line + ESMC PCC range`
   and `docs(pharma): correct ESMC-300m PCC range 0.9988 -> 0.9987`. The Status
   summary had R and X swapped for the OpenDDE production leg relative to the
   committed data (`opendde-prod-leg.json`: cross_X 5.6789 +/- 3.9759,
   ref_floor_R 1.9009, dev_floor_D 8.0606) and the rest of the doc; the ESMC-300m
   PCC range was listed as ">0.999" (false: min is 0.9987496) / "0.9988" (rounds
   to 0.9987). Corrected to the measured 0.9987-0.9996.

## Gate legs

### 1. Accuracy / ground-truth — `scripts/release_gate.py` — RUNNING (PENDING)

Relaunched setsid on card 0 at 20:03 (see Notes). Log: `runs/gate-full.log`;
poll for `GATE_EXIT` and the `RELEASE GATE` table. Folds `examples/prot.yaml`
(7ROA, 117-res) at production 200 steps / 5 samples for boltz2, esmfold2,
esmfold2-fast, protenix-v2, plus BoltzGen designability (n=4, 500-step) and ESMC
300m/600m embed parity. ~35 min end-to-end; will complete after this worker turn.

### 2. Perf — `scripts/perf_regression.py` — FAIL (exit 1)

Full run (6 models, warm 2+5, trpcage 20aa single-seq, 1 recycle / 10 steps / 1
sample), baseline `docs/perf_baselines.json` (0.2.5, blackhole, same input,
seeded 2026-07-13 → directly comparable):

| model | metric | baseline | current | delta | verdict |
|---|---|---|---|---|---|
| boltz2 | structures/s | 1.498 | 1.427 | -4.7% | PASS |
| esmfold2 | structures/s | 2.414 | 2.267 | -6.1% | PASS |
| esmfold2-fast | structures/s | 3.126 | 2.959 | -5.4% | PASS |
| protenix-v2 | structures/s | 3.451 | 1.238 | -64.1% | **FAIL** |
| opendde | structures/s | 2.683 | 2.654 | -1.1% | PASS |
| esmc-600m | seq/s | 22.1 | 24.78 | +12.1% | PASS |

protenix-v2 is the only model beyond the +/-15% threshold. Two independent
measurements both fail: 2.153 structures/s (-37.6%) during a 4-way concurrent
device run, and 1.238 structures/s (-64.1%) re-measured alone on card 1. Both
runs had host-side contention (other device jobs cold-compiling on the other
cards), so the exact magnitude is not clean — but both are far below the 3.451
baseline, and the other 5 models sit within noise of their baselines on the same
hardware, so this is a **real protenix-v2 perf regression on main since v0.2.5**
(direction unambiguous; a clean re-measure with all other device jobs stopped is
recommended to nail the number — see Notes). This is a release-gate blocker.

### 3. UX — `scripts/ux_regression.py` — RUNNING (PENDING)

Relaunched setsid on card 2 at 20:03. Log: `runs/ux.log`; poll for `UX_EXIT`.

### 4. Test suite — `pytest` — FAIL (exit 1)

Full suite, canonical env, quiet card 3: **2 failed, 137 passed, 45 skipped,
3 xfailed in 1284.14s**. Both failures reproduce (a prior relaunch's run in the
`tt-bio-dev` python 3.10 env hit the same two):

- `tests/test_openfold3_pairformer.py::test_of3_pairformer_block0_on_device` —
  `KeyError: 'pairformer_block0'` at line 71. This is the file's "honest
  correctness gate (passes)" test (the full-48-block test is separately xfail);
  its failure is a regression, not a known-xfail.
- `tests/test_protenix_seqfold.py::test_protenix_sequence_to_structure` —
  `subprocess.TimeoutExpired` after 600s running
  `scripts/protenix_seqfold.py GSSGSSGQITLWQRPLVT 8` (a 16-res peptide fold).
  Likely a consequence of the protenix-v2 perf regression (a ~2.8x slower
  protenix pushes the seqfold over its 600s bar); confirm by re-running after
  the perf regression is fixed.

## Pharma-benchmark doc consistency — PASS

Spot-checked R/D/X in `docs/pharma-benchmark.md` against the committed JSON in
`docs/pharma-benchmark-data/` (the way recent verify workers have), all match:

- Boltz-2: trpcage X=0.60+/-0.24 R=0.79 D=0.37 (within); prot no-MSA
  X=5.51+/-0.70 R=3.37 D=4.35 (1.27x, disclosed gap); prot MSA X=0.94+/-0.14
  R=0.81 D=0.98 (within) — `boltz2.json` targets prot/trpcage/prot_msa.
- ESMFold2: trpcage X=0.61+/-0.10 R=0.51 D=0.16; GB1 X=0.33+/-0.05 R=0.29
  D=0.18; ubiquitin X=0.75+/-0.10 R=0.92 D=0.23 — `esmfold2.json`.
- Protenix-v2: X=2.63+/-0.42 R=2.94 D=1.47 (X/floor 0.89) — `protenix-v2.json`.
- OpenDDE: trpcage X=0.39+/-0.11 R=0.31 D=0.24 — `opendde.json`; production prot
  X=5.68+/-3.98 R=1.90 D=8.06 (X/floor 0.70) — `opendde-prod-leg.json`.
- ESMC: 300m 0.9987-0.9996, 600m 0.9994-0.9996 — `esmc-300m.json`/`esmc-600m.json`.

The Status section has nothing "pending"/"in progress" — all four shipped models
+ BoltzGen + ESMC are listed Complete with measured numbers. (Found and fixed
two doc inaccuracies while in there — the swapped R/X and the ESMC PCC range,
committed above.)

## Notes for the next relaunch / orchestrator

- Accuracy gate running setsid on card 0. Poll `runs/gate-full.log` for
  `GATE_EXIT=` and the `RELEASE GATE` pass/fail table. Card 0 was wedged earlier
  (ARC heartbeat stalled) by a 4-way concurrent device launch; reset with
  `tt-smi -r 0` recovered it and the gate is now folding on it (AICLK 0x546).
  Lesson: do NOT launch release_gate.py + perf_regression.py + ux_regression.py
  + pytest simultaneously on 4 cards — the concurrent device-open +
  checkpoint-download storm wedged card 0 and killed predict workers. Run them
  with a stagger or sequentially.
- UX running setsid on card 2. Poll `runs/ux.log` for `UX_EXIT=`.
- After both finish, re-measure protenix-v2 perf alone on a quiet card (no other
  device jobs) for a clean number:
  `TT_VISIBLE_DEVICES=<card> PYTHONPATH=. python scripts/perf_regression.py --model protenix-v2`.
- The `release_gate.py`/`main.py:gen` P300 mesh-graph fix on this branch is
  REQUIRED for the gate to run on qb2 P300 — merge it (no accuracy/perf/OOM
  impact).
- **Push:** `ttuser@tt-quietbox2` has no GitHub credential (no token, no `gh auth`,
  no credential helper, SSH key not registered with github.com). Prior `wk/`
  branches on origin were pushed by the orchestrator (creds are orchestrator-side,
  not worker-side). `git push origin wk/tt-bio-release-gate-verify` from the
  worker fails with `could not read Username for 'https://github.com'`. The
  orchestrator must push this branch and run the DONE_CHECK.
