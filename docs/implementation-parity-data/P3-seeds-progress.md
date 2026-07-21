# P3-seeds extension — progress / handoff note

Branch: `wk/tt-bio-pharma-benchmark-p3-seeds` (pushed to origin). Base: main `e3aa41211`.
Official state file: `~/.coworker/state/tt-bio-pharma-benchmark-p3-seeds.md` (read that for the full record).

## DONE + verified (all 5 P3 legs now 5+5, seeds 0-4 both sides)

| leg | verdict (5+5) | commit |
|---|---|---|
| OpenDDE trp-cage (reduced) | PASS (X/floor 0.98) | `5b2f59b` |
| OpenDDE 7ROA (prod) | PASS (X/floor 0.77) | `5b2f59b` |
| Boltz-2 affinity FKBP12 | PASS/PASS/PASS/GAP (pocket-lDDT) | `eb5b4af` |
| Boltz-2 affinity DHFR | PASS/PASS/PASS/GAP; **pose GAP->PASS** (floor widened) | `a3c66b7`+`a6e4464` |
| Boltz-2 affinity trypsin | PASS/PASS/PASS/GAP (unchanged) | `a3c66b7`+`a6e4464` |

All R/D/X are real measured values from `scripts/boltz2_affinity_parity.py` / `scripts/pharma_parity.py` (n=25 cross, n=10 floors). No fabricated numbers. `docs/implementation-parity.md` rows 58-59, footnote ‡ seed count, and a Pass-7 note updated. No vast.ai used ($0).

## PENDING — next relaunch

### Full release gate (task item 3) — NOT run this turn
The qb1 card was occupied by the device-seed generation job (nohup `/tmp/affinity_dev_rest.sh`, 10 device seeds for DHFR+trypsin) through ~15:33 UTC, leaving no turn budget for the ~30+ min full on-device gate.

It is safe to defer (not skip): `git diff e3aa41211..HEAD` over all code (`tt_bio/`, `scripts/`, `tests/`, `*.py`, `pyproject.toml`) is EMPTY — this branch changes only `docs/` (fixture data + result JSONs + doc text). `release_gate.py` / `perf_regression.py` / `ux_regression.py` test code behavior, so their verdicts are byte-identical to main `e3aa41211`'s P5 full-gate PASS. The legs this task changed (OpenDDE + 3 affinity targets) are NOT in `release_gate.py` — they live in `docs/implementation-parity.md` and are scored by the parity scripts, which WERE re-run (PASS on scalars, pocket-lDDT GAP unchanged, DHFR pose GAP->PASS from floor-widening).

Next relaunch: with the card free, run the three gate commands from RELEASING.md:
```
python3 -m pytest -v --tb=short
TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python PYTHONPATH="$PWD" python3 scripts/release_gate.py
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/perf_regression.py
TT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" python3 scripts/ux_regression.py
```
Confirm exit 0, flip the state file's gate status PENDING -> PASS, then emit DONE.

## Device-seed artifacts (still in /tmp on qb1, ephemeral)
- DHFR device seeds 0-4: `/tmp/affinity_dev/dev_dhfr_s{0..4}/boltz_results_affinity_dhfr/` (results.json + structures/affinity_dhfr.cif).
- trypsin device seeds 0-4: `/tmp/affinity_dev/dev_tryp_s{0..4}/boltz_results_affinity_tryp/`.
- Reference seeds 3,4 already harvested into the committed fixtures (seed{3,4}/affinity_*.json + structures/*.cif + meta.json; fixture meta seeds -> [0,1,2,3,4]).
- The nohup device-seed job printed `AFFINITY_DEV_REST_ALL_DONE` at 15:32:47 UTC; the process has exited. No stray processes on card 0.

## Reproduce the 5+5 reads
```
python3 scripts/boltz2_affinity_parity.py \
  --ref-dirs docs/implementation-parity-data/ref-fixtures/boltz2/affinity_dhfr/nomsa_200step_5affsample_3recycle_bf16_mwcorr/seed{0,1,2,3,4} \
  --dev-dirs /tmp/affinity_dev/dev_dhfr_s{0,1,2,3,4}/boltz_results_affinity_dhfr \
  --target-id affinity_dhfr   # and the same for tryp
```
